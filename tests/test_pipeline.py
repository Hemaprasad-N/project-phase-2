from src.document_processor import DocumentProcessor
from src.text_preprocessor import TextPreprocessor
from src.risk_detector import RiskDetector
from src.risk_scorer import RiskScorer
from src.ml_classifier import MLClassifier
import os

def test_phase1_pipeline():
    sample_text = "This contract automatically renews. We are not responsible for damages. We will share your data with third parties. No refund."
    
    # Preprocessing
    sentences = TextPreprocessor.segment_sentences(sample_text)
    assert len(sentences) == 4, "Sentence segmentation failed"
    
    # Rule-Based Risk Detection
    risks = RiskDetector.detect_risks(sentences, use_ml=False)
    assert len(risks) == 4, f"Risk detection failed to catch all risks, caught {len(risks)}"
    
    # Advanced Scoring validation
    score, level = RiskScorer.score_risks(risks)
    assert 0 <= score <= 100, f"Score out of bounds: {score}"
    assert level in ["Low", "Medium", "High"], f"Invalid Risk level: {level}"
    
def test_ml_pipeline():
    # Only test if model exists
    if not os.path.exists("data/risk_model_logreg.joblib"):
        print("Skipping ML test, model not yet trained. Run train_model.py.")
        return
        
    classifier = MLClassifier("logreg")
    assert classifier.load_model() == True, "Failed to load ML model"
    
    sample_text = "By using this service, you consent to third-party data tracking."
    sentences = TextPreprocessor.segment_sentences(sample_text)
    
    risks = RiskDetector.detect_risks(sentences, use_ml=True, ml_classifier=classifier)
    assert len(risks) > 0, "ML Failed to detect obvious risk."
    assert "severity" in risks[0], "Severity not returned from ML."
    
    print("Testing ML Classification complete!")

if __name__ == "__main__":
    test_phase1_pipeline()
    test_ml_pipeline()
    print("All Pipeline components tested successfully!")
