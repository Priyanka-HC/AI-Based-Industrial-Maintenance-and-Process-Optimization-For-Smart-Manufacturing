import sys
import os
import joblib

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import preprocess_data

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ============================
# Load and preprocess data
# ============================
X_train, X_test, y_train, y_test, scaler = preprocess_data()

# ============================
# Logistic Regression
# ============================
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# ============================
# Random Forest
# ============================
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# ============================
# XGBoost
# ============================
xgb = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)

# ============================
# Model Comparison
# ============================
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(f"Logistic Regression Accuracy : {lr_acc:.4f}")
print(f"Random Forest Accuracy       : {rf_acc:.4f}")
print(f"XGBoost Accuracy             : {xgb_acc:.4f}")

# ============================
# Select Best Model
# ============================
models = {
    "Logistic Regression": (lr_acc, lr),
    "Random Forest": (rf_acc, rf),
    "XGBoost": (xgb_acc, xgb)
}

best_model_name = max(models, key=lambda x: models[x][0])
best_model = models[best_model_name][1]
best_accuracy = models[best_model_name][0]

print("\n" + "=" * 60)
print(f"BEST MODEL : {best_model_name}")
print(f"ACCURACY   : {best_accuracy:.4f}")
print("=" * 60)

# ============================
# Save Model and Scaler
# ============================
os.makedirs("saved_models", exist_ok=True)

joblib.dump(best_model, "saved_models/model.pkl")
joblib.dump(scaler, "saved_models/scaler.pkl")

print("\n✅ Model saved successfully!")
print("📁 saved_models/model.pkl")
print("📁 saved_models/scaler.pkl")