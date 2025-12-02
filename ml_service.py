"""
ML Service for Circuit Breaker Fault Detection
Uses Random Forest Classifier for fault prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
from datetime import datetime

class MLFaultDetector:
    def __init__(self):
        self.model = None
        self.model_path = 'models/fault_detector.pkl'
        self.training_data_path = 'data/training_data.csv'
        print(f"Model path: {self.model_path}")
        print(f"Training data path: {self.training_data_path}")

        self.model_stats = {}
        
        # Create directories if they don't exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        # Load model if it exists
        self.load_model()
    
    def save_training_data(self, df):
        """Save uploaded training data"""
        try:
            df.to_csv(self.training_data_path, index=False)
            return True
        except Exception as e:
            print(f"Error saving training data: {str(e)}")
            return False
    
    def train_model(self, test_size=0.2, n_estimators=100, max_depth=10, random_state=42):
        """Train Random Forest model for fault detection"""
        try:
            # Load training data
            if not os.path.exists(self.training_data_path):
                return {
                    "success": False,
                    "message": "No training data found. Please upload training data first."
                }
            
            df = pd.read_csv(self.training_data_path)
            
            # Validate required columns
            required_columns = ['voltage', 'current', 'temperature', 'operating_time', 
                              'vibration', 'insulation_resistance', 'oil_level', 'fault_status']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return {
                    "success": False,
                    "message": f"Missing required columns: {', '.join(missing_columns)}"
                }
            
            # Prepare features and target
            feature_columns = [col for col in required_columns if col != 'fault_status']
            X = df[feature_columns]
            y = df['fault_status']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Train Random Forest
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
            
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            # Feature importance
            feature_importance = [
                {"feature": feature, "importance": float(importance)}
                for feature, importance in zip(feature_columns, self.model.feature_importances_)
            ]
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            
            # Save model and stats
            self.save_model()
            self.model_stats = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "dataset_size": len(df),
                "training_date": datetime.now().isoformat(),
                "version": "1.0",
                "n_estimators": n_estimators,
                "max_depth": max_depth
            }
            
            return {
                "success": True,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": cm.tolist(),
                "feature_importance": feature_importance,
                "message": "Model trained successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Training error: {str(e)}"
            }
    
    def predict(self, input_data):
        """Predict fault from input parameters"""
        try:
            if self.model is None:
                return {
                    "success": False,
                    "message": "Model not trained yet"
                }
            
            # Prepare input features
            features = pd.DataFrame([{
                'voltage': input_data.get('voltage', 0),
                'current': input_data.get('current', 0),
                'temperature': input_data.get('temperature', 0),
                'operating_time': input_data.get('operating_time', 0),
                'vibration': input_data.get('vibration', 0),
                'insulation_resistance': input_data.get('insulation_resistance', 0),
                'oil_level': input_data.get('oil_level', 0)
            }])
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0]
            
            # Determine fault status and risk level
            if prediction == 0:
                fault_status = "No Fault"
                risk_level = "Low"
                explanation = "All parameters are within normal operating range."
                recommendations = "Continue regular monitoring and maintenance schedule."
            else:
                fault_status = "Fault Detected"
                confidence = max(probability) * 100
                
                # Determine risk level based on confidence
                if confidence > 90:
                    risk_level = "Critical"
                elif confidence > 70:
                    risk_level = "High"
                else:
                    risk_level = "Medium"
                
                # Generate explanation based on parameters
                issues = []
                if input_data.get('temperature', 0) > 80:
                    issues.append("high temperature")
                if input_data.get('vibration', 0) > 20:
                    issues.append("excessive vibration")
                if input_data.get('insulation_resistance', 0) < 100:
                    issues.append("low insulation resistance")
                if input_data.get('oil_level', 0) < 50:
                    issues.append("low oil level")
                
                explanation = f"Fault detected due to: {', '.join(issues) if issues else 'anomalous parameter combination'}."
                recommendations = "Immediate inspection recommended. Schedule maintenance within 24-48 hours."
            
            return {
                "success": True,
                "fault_detected": fault_status,
                "confidence": round(max(probability) * 100, 2),
                "risk_level": risk_level,
                "fault_type": "Operational Fault" if prediction == 1 else None,
                "explanation": explanation,
                "recommendations": recommendations,
                "probability": {
                    "no_fault": round(probability[0] * 100, 2),
                    "fault": round(probability[1] * 100, 2) if len(probability) > 1 else 0
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Prediction error: {str(e)}"
            }
    
    def save_model(self):
        """Save trained model to disk"""
        try:
            if self.model is not None:
                joblib.dump(self.model, self.model_path)
                return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self):
        """Load trained model from disk"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print("✅ Model loaded successfully")
                return True
        except Exception as e:
            print(f"⚠️ Could not load model: {str(e)}")
            return False
    
    def delete_model(self):
        """Delete trained model"""
        try:
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            self.model = None
            self.model_stats = {}
            return True
        except Exception as e:
            print(f"Error deleting model: {str(e)}")
            return False
    
    def get_model_stats(self):
        """Get model statistics"""
        if self.model is None:
            return {
                "model_trained": False,
                "message": "No model trained yet"
            }
        
        return {
            "model_trained": True,
            **self.model_stats
        }

# Global ML service instance
ml_service = MLFaultDetector()
