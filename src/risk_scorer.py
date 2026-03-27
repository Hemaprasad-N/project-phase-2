class RiskScorer:
    @staticmethod
    def score_risks(detected_risks: list[dict]) -> tuple[int, str]:
        if not detected_risks:
            return 0, "Low"
        
        # Advanced ML-ready scoring: aggregate severity scores
        total_severity = sum(risk.get("severity", 0.5) for risk in detected_risks)
        
        # Let's say one severe risk equals 20 points
        base_score = int(min(total_severity * 20, 100))
        
        if base_score <= 30:
            level = "Low"
        elif base_score <= 60:
            level = "Medium"
        else:
            level = "High"
            
        return base_score, level
