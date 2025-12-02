import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from config import DEFAULT_STOCK_TICKERS
from paths import get_data_path, get_model_path

def train_model(ticker, horizons=[1, 7, 30]):
    """
    Train a predictive model for stock price direction for different horizons.
    """
    accuracies = {}
    for h in horizons:
        try:
            features_path = get_data_path(f"{ticker}_features_{h}d.csv")
            labels_path = get_data_path(f"{ticker}_labels_{h}d.csv")
            features = pd.read_csv(features_path)
            labels = pd.read_csv(labels_path)['label']
            
            # Adjusted minimum based on horizon
            min_required = 3 if h == 1 else (5 if h == 7 else 10)
            
            if features.empty or labels.empty or len(features) < min_required:
                print(f"⚠️ Insufficient data for {ticker} {h}-day (have {len(features)}, need {min_required}), skipping.")
                continue
            
            # Check for minimum samples
            if len(features) != len(labels):
                print(f"Feature/label mismatch for {ticker} {h}-day")
                continue
            
            # For very small datasets, adjust test_size
            if len(features) < 10:
                test_size = 0.25  # Use 25% for very small datasets
                print(f"⚠️ Training with only {len(features)} samples - predictions may be unreliable")
            else:
                test_size = 0.2
            
            # Check if we can stratify (need at least 2 samples per class)
            label_counts = labels.value_counts()
            can_stratify = len(label_counts) > 1 and label_counts.min() >= 2
            
            if not can_stratify:
                print(f"⚠️ Cannot stratify split for {ticker} {h}-day (class distribution: {label_counts.to_dict()})")
                # For very small datasets with imbalanced classes, just split without stratification
                if len(features) < 5:
                    # Use all data for training (no test split) for very small datasets
                    X_train, y_train = features, labels
                    X_test, y_test = features, labels  # Use same data for evaluation (not ideal but works)
                    print(f"⚠️ Using all {len(features)} samples for training (no test split)")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        features, labels, test_size=test_size, random_state=42, shuffle=True
                    )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=test_size, random_state=42, stratify=labels
                )

            # Adjust model complexity based on dataset size
            if len(features) < 10:
                n_estimators = 50  # Fewer trees for small datasets
                max_depth = 3      # Limit depth to prevent overfitting
            else:
                n_estimators = 100
                max_depth = None
            
            model = RandomForestClassifier(
                n_estimators=n_estimators, 
                max_depth=max_depth,
                random_state=42, 
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy for {ticker} {h}-day: {accuracy:.4f}")

            # Save model
            model_path = get_model_path(f"{ticker}_model_{h}d.pkl")
            joblib.dump(model, model_path)
            accuracies[h] = accuracy
            
        except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e:
            print(f"Data error for {ticker} {h}-day: {e}")
            continue
        except Exception as e:
            print(f"Error training model for {ticker} {h}-day: {e}")
            continue

    if accuracies:
        print(f"Training completed for {ticker}. Accuracies: {accuracies}")
    else:
        print(f"No models trained for {ticker}")
    return accuracies

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else input("Enter stock ticker: ")
    train_model(ticker)