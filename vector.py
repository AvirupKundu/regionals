
import os
import json
import hashlib
import httpx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# ----------------------------
# Local tiktoken cache
# ----------------------------
tiktoken_cache_dir = "tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
assert os.path.exists(os.path.join(tiktoken_cache_dir, "9b5ad71b2ce5302211f9c61530b329a4922fc6a4")), \
    "Missing tiktoken cache file in 'tiktoken_cache/'."

client = httpx.Client(verify=False)


# ----------------------------
# Token-level F1 helper
# ----------------------------
def _tokenize_text(s):
    return s.lower().split()

def f1_score_tokens(pred, gold):
    p_tokens = _tokenize_text(pred)
    g_tokens = _tokenize_text(gold)
    if not p_tokens and not g_tokens:
        return 1.0
    if not p_tokens or not g_tokens:
        return 0.0
    common = {}
    for t in p_tokens:
        common[t] = common.get(t, 0) + 1
    match = 0
    for t in g_tokens:
        if common.get(t, 0) > 0:
            match += 1
            common[t] -= 1
    precision = match / len(p_tokens)
    recall = match / len(g_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# ----------------------------
# 1) Chunking
# ----------------------------
def chunk_texts(documents, chunk_size=1000, chunk_overlap=200, keep_meta=True):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""])
    chunks = []
    for doc in documents:
        doc_id = doc.get("id") or hashlib.sha1(doc.get("text", "").encode()).hexdigest()
        text = doc.get("text", "")
        meta = doc.get("meta", {})
        split_texts = splitter.split_text(text)
        for i, t in enumerate(split_texts):
            chunk = {
                "chunk_id": f"{doc_id}_chunk_{i}",
                "doc_id": doc_id,
                "text": t,
                "meta": meta if keep_meta else {}
            }
            chunks.append(chunk)
    return chunks

# ----------------------------
# 2) Create embeddings
# ----------------------------
def create_embeddings(texts):
    embedder = OpenAIEmbeddings(
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("EMBEDDING_MODEL"),
        http_client=client
    )

    print("Embedding model initialized.")
    embeddings = embedder.embed_documents(texts)
    print(embeddings)
    print(f"Created {len(embeddings)} embeddings.")
    return embeddings, embedder


# ----------------------------
# 3) Store in Chroma
# ----------------------------

def store_in_chroma(collection_name, chunks, embeddings, persist_directory="./chromadb_store", distance_metric="cosine"):
    client = chromadb.PersistentClient(path=persist_directory)

    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        collection = client.create_collection(name=collection_name, metadata={"distance": distance_metric})

    ids = [c["chunk_id"] for c in chunks]
    metadatas = [{"doc_id": c["doc_id"], **(c.get("meta") or {})} for c in chunks]
    documents = [c["text"] for c in chunks]

    if len(documents) != len(embeddings):
        raise ValueError("Mismatch between documents and embeddings length")

    collection.upsert(ids=ids, metadatas=metadatas, documents=documents, embeddings=embeddings)
    return collection


# ----------------------------
# 4) Search
# ----------------------------

def search_vector_db(query, collection, embedder, k=5, filter_meta=None):
    q_emb = embedder.embed_query(query)
    if filter_meta:
        results = collection.query(query_embeddings=[q_emb], n_results=k, where=filter_meta)
    else:
        results = collection.query(query_embeddings=[q_emb], n_results=k)

    hits = []
    for idx in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][idx],
            "text": results["documents"][0][idx],
            "meta": results["metadatas"][0][idx],
            "distance": results["distances"][0][idx]
        })
    return hits


# ----------------------------
# 5) Retrieve KB result
# ----------------------------
def retrieve_kb_result(hits):
    parts = []
    for i, h in enumerate(hits):
        parts.append(f"Result {i+1} (id={h['id']})\n{h['text']}")
    return "\n\n---\n\n".join(parts)

# ----------------------------
# 6) Evaluation
# ----------------------------
def evaluate_retrieval_accuracy(retrieved_hits, relevant_doc_ids, k=5):
    found = 0
    retrieved_ids = [h["meta"].get("doc_id") for h in retrieved_hits]
    for rid in relevant_doc_ids:
        if rid in retrieved_ids[:k]:
            found += 1
    recall_at_k = found / max(1, len(relevant_doc_ids))
    return {"recall_at_k": recall_at_k, "retrieved_ids": retrieved_ids[:k]}

def evaluate_response_accuracy(response_text, gold_text, embedder):
    emb_resp = embedder.embed_query(response_text)
    emb_gold = embedder.embed_query(gold_text)
    cos_sim = float(cosine_similarity([emb_resp], [emb_gold])[0][0])
    token_f1 = f1_score_tokens(response_text, gold_text)
    return {"cosine_similarity": cos_sim, "token_f1": token_f1}

# ----------------------------
# 7) Main pipeline
# ----------------------------
def main_pipeline(docs, user_queries, collection_name="kb_collection", persist_directory="./chromadb_store", chunk_size=1000, chunk_overlap=200, top_k=5):
    client = httpx.Client(verify=False)

    # 1) Chunk
    chunks = chunk_texts(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        keep_meta=True
    )

    # 2) Embeddings
    texts = [c["text"] for c in chunks]
    embeddings, embedder = create_embeddings(texts)

    # 3) Store
    collection = store_in_chroma(
        collection_name,
        chunks,
        embeddings,
        persist_directory=persist_directory
    )
    print(f"Chroma collection '{collection_name}' ready with {len(chunks)} chunks.")

    # Optional LLM for generating answers
    llm = ChatOpenAI(
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("LLM_MODEL"),
        http_client=client
    )

    results = {"collection_name": collection_name, "num_chunks": len(chunks), "queries": []}

    for q in user_queries:
        query_text = q["query"]
        gold_doc_ids = q.get("gold_doc_ids", [])
        gold_answer = q.get("gold_answer")

        hits = search_vector_db(query_text, collection, embedder, k=top_k)
        context = retrieve_kb_result(hits)

        # Generate response using LLM
        response_text = llm.invoke(f"Use the following context to answer the question:\n{context}\nQuestion: {query_text}")

        retrieval_metrics = evaluate_retrieval_accuracy(hits, gold_doc_ids, k=top_k)
        response_metrics = {}
        if gold_answer:
            response_metrics = evaluate_response_accuracy(str(response_text), gold_answer, embedder)

        results["queries"].append({
            "query": query_text,
            "hits": [{"id": h["id"], "doc_id": h["meta"].get("doc_id"), "distance": h["distance"]} for h in hits],
            "retrieval_metrics": retrieval_metrics,
            "response_metrics": response_metrics,
            "response_text_snippet": str(response_text)[:1000]
        })
    return results

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    docs = [
        {"id": "doc1", "text": "LangChain is a library for building applications with LLMs. It provides chains, agents, and integrations.", "meta": {"title": "LangChain intro"}},
        {"id": "doc2", "text": "Chroma is an open-source embedding database designed for local and cloud usage. It supports simple Python API.", "meta": {"title": "Chroma intro"}}
    ]
    user_queries = [
        {"query": "What is LangChain used for?", "gold_doc_ids": ["doc1"], "gold_answer": "LangChain is a library for building applications with LLMs, providing chains and integrations."},
        {"query": "What is Chroma?", "gold_doc_ids": ["doc2"], "gold_answer": "Chroma is an open-source embedding database with a Python API for local and cloud use."}
    ]
    output = main_pipeline(docs, user_queries)
    print(json.dumps(output, indent=2))
