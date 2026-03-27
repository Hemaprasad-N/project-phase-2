import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError

class MLClassifier:
    def __init__(self, model_type="logreg"):
        self.model_type = model_type.lower()
        self.model_path = f"data/risk_model_{self.model_type}.joblib"
        self.pipeline = None

    def create_pipeline(self):
        # Base vectorizer with updated TF-IDF settings
        vectorizer = TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1,2), sublinear_tf=True)
        return Pipeline([
            ('tfidf', vectorizer)
        ])
        
    def get_classifier(self):
        if self.model_type == "svm":
            # LinearSVC decision_function can be calibrated
            base_svm = LinearSVC(C=1.0, random_state=42, class_weight='balanced')
            return CalibratedClassifierCV(estimator=base_svm)
        elif self.model_type == "rf":
            return RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        else:
            # Default to LogReg
            return LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced', solver='lbfgs', multi_class='multinomial')

    def set_trained_pipeline(self, pipeline):
        self.pipeline = pipeline

    def predict_clause(self, clause: str):
        if not self.pipeline:
            raise NotFittedError("Model is not loaded or trained yet.")
        
        prediction = self.pipeline.predict([clause])[0]
        
        # Get probability/confidence
        if hasattr(self.pipeline, "predict_proba"):
            probabilities = self.pipeline.predict_proba([clause])[0]
            confidence = max(probabilities)
        elif hasattr(self.pipeline, "decision_function"):
            decision = self.pipeline.decision_function([clause])[0]
            # Simple min-max scaling mock or fallback if no proba
            confidence = min(max((abs(decision.max()) / 2.0), 0.5), 1.0)
        else:
            confidence = 0.5
            
        return prediction, confidence

    def save_model(self):
        if self.pipeline:
            joblib.dump(self.pipeline, self.model_path)
            # Also save vectorizer separately if explicitly needed for anything, but pipeline covers it
        else:
            raise ValueError("No model to save.")

    def load_model(self) -> bool:
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
            return True
        return False
