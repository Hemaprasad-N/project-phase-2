import pandas as pd
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from src.ml_classifier import MLClassifier

def train_and_evaluate(model_type, X_train, X_test, y_train, y_test):
    print(f"\n--- Training {model_type.upper()} ---")
    classifier_wrapper = MLClassifier(model_type=model_type)
    
    # We will build a pipeline for GridSearchCV if it's logreg
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', sublinear_tf=True, max_features=3000)),
        ('clf', classifier_wrapper.get_classifier())
    ])
    
    if model_type == "logreg":
        param_grid = {
            'tfidf__ngram_range': [(1,1), (1,2)],
            'tfidf__min_df': [1, 2],
            'clf__C': [0.1, 1, 5, 10]
        }
        print("Running GridSearchCV for Logistic Regression...")
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    else:
        best_model = pipeline.fit(X_train, y_train)
        
    classifier_wrapper.set_trained_pipeline(best_model)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"F1-score (macro): {f1_macro:.2f}")
    print(f"F1-score (weighted): {f1_weighted:.2f}")
    
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm
    }
    
    print(f"Saving {model_type.upper()} model using joblib...")
    classifier_wrapper.save_model()
    return metrics

def main():
    print("Loading dataset...")
    df = pd.read_csv("data/risk_dataset.csv")

    X = df['clause'].values
    y = df['label'].values

    # Stratified split 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    all_metrics = {}
    for mt in ["logreg", "svm", "rf"]:
        metrics = train_and_evaluate(mt, X_train, X_test, y_train, y_test)
        all_metrics[mt] = metrics
        
    with open("data/model_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)
    print("\nAll models trained and metrics saved to data/model_metrics.json")

if __name__ == "__main__":
    main()
