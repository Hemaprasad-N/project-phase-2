import re
from src.ml_classifier import MLClassifier

class RiskDetector:
    # Basic rule-based dictionary for Phase 1
    RISK_KEYWORDS = {
        "Data Privacy Risk": ["share your data", "third party", "track your activities", "sell your information"],
        "Financial Risk": ["auto-renew", "automatically renews", "subscription", "hidden charge", "fee"],
        "Liability Risk": ["limitation of liability", "indemnify", "not responsible", "as is"],
        "Termination & Refund Risk": ["no refund", "terminate at any time", "sole discretion"],
        "Arbitration": ["class action waiver", "binding arbitration", "waive your right"]
    }

    @staticmethod
    def detect_risks(sentences: list[str], use_ml: bool = False, ml_classifier: MLClassifier = None) -> list[dict]:
        detected_risks = []
        for sentence in sentences:
            clause = sentence.strip()
            if not clause:
                continue

            if use_ml and ml_classifier:
                pred, conf = ml_classifier.predict_clause(clause)
                if pred != "No Risk":
                    detected_risks.append({
                        "type": pred,
                        "clause": clause,
                        "keyword_matched": "ML Detection",
                        "severity": round(conf, 2)
                    })
            else:
                # Fallback to rule-based detection
                for risk_type, keywords in RiskDetector.RISK_KEYWORDS.items():
                    found_risk = False
                    for keyword in keywords:
                        if re.search(r'\b' + re.escape(keyword) + r'\b', sentence, re.IGNORECASE):
                            detected_risks.append({
                                "type": risk_type,
                                "clause": clause,
                                "keyword_matched": keyword,
                                "severity": 0.5  # default rule-based severity
                            })
                            found_risk = True
                            break
                    if found_risk:
                        break # Limit to mapping one obvious risk type per sentence for simplicity
        return detected_risks
