"""
generate_ml_data.py
────────────────────────────────────────────────────────
Trains ML models on synthetic Cairo traffic data and
exports ml_predictions.json for use in the web app.

Models used (scikit-learn):
  - GradientBoostingRegressor  → traffic multiplier (continuous)
  - RandomForestClassifier     → congestion level (low/medium/high)

Run:
  python generate_ml_data.py
"""

import json
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# STEP 1: Generate temporal training data
# ─────────────────────────────────────────────
def generate_training_data():
    hours = list(range(24))
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    records = []

    for day_i, day in enumerate(days):
        for hour in hours:
            # Realistic Cairo traffic model:
            # - Morning rush: 7–10am on weekdays
            # - Evening rush: 4–8pm (worse than morning in Cairo)
            # - Friday: ~30% lighter (shorter work week)
            # - Saturday: moderate (shopping, leisure)
            # - Late night: very light
            base = 1.0
            if hour in range(7, 10):   base = 2.5
            elif hour in range(16, 20): base = 2.8
            elif hour in range(0, 5):   base = 0.4
            elif hour in range(22, 24): base = 0.6
            elif hour in range(5, 7):   base = 0.8

            if day in ['Fri', 'Sat']:
                base *= 0.7

            # Add Gaussian noise to simulate real variance
            traffic = base + random.gauss(0, 0.15)
            traffic = max(0.3, min(3.5, traffic))

            congestion = ('high' if traffic > 2.0
                          else 'medium' if traffic > 1.2
                          else 'low')

            records.append({
                'hour': hour,
                'day_of_week': day_i,
                'is_weekend': 1 if day in ['Fri', 'Sat'] else 0,
                'traffic_multiplier': round(traffic, 3),
                'congestion_level': congestion
            })

    return records


# ─────────────────────────────────────────────
# STEP 2: Feature engineering
# ─────────────────────────────────────────────
def build_features(records):
    X, y_reg, y_cls = [], [], []
    for r in records:
        h = r['hour']
        X.append([
            h,
            r['day_of_week'],
            r['is_weekend'],
            np.sin(2 * np.pi * h / 24),   # cyclic hour encoding
            np.cos(2 * np.pi * h / 24),
        ])
        y_reg.append(r['traffic_multiplier'])
        y_cls.append(r['congestion_level'])
    return np.array(X), np.array(y_reg), y_cls


# ─────────────────────────────────────────────
# STEP 3: Train models
# ─────────────────────────────────────────────
def train_models(X, y_reg, y_cls):
    print("Training GradientBoostingRegressor...")
    reg = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=4)
    reg.fit(X, y_reg)
    cv_reg = cross_val_score(reg, X, y_reg, cv=5, scoring='r2')
    print(f"  R² scores (5-fold CV): {cv_reg.round(3)}")
    print(f"  Mean R²: {cv_reg.mean():.3f}")

    print("Training RandomForestClassifier...")
    le = LabelEncoder()
    y_enc = le.fit_transform(y_cls)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y_enc)
    cv_clf = cross_val_score(clf, X, y_enc, cv=5, scoring='accuracy')
    print(f"  Accuracy scores (5-fold CV): {cv_clf.round(3)}")
    print(f"  Mean Accuracy: {cv_clf.mean():.3f}")

    return reg, clf, le


# ─────────────────────────────────────────────
# STEP 4: Generate predictions for all hours/days
# ─────────────────────────────────────────────
def generate_predictions(reg, clf, le):
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    weekly = {}

    for day_i, day in enumerate(days):
        weekly[day] = []
        for hour in range(24):
            is_weekend = 1 if day in ['Fri', 'Sat'] else 0
            feat = np.array([[
                hour, day_i, is_weekend,
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24)
            ]])
            mult = float(reg.predict(feat)[0])
            cls_idx = int(clf.predict(feat)[0])
            cls = le.inverse_transform([cls_idx])[0]
            weekly[day].append({
                'hour': hour,
                'multiplier': round(mult, 3),
                'congestion': cls
            })

    # Today's predictions with confidence scores (assume Monday = weekday)
    today = []
    for hour in range(24):
        feat = np.array([[
            hour, 0, 0,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24)
        ]])
        mult = float(reg.predict(feat)[0])
        proba = clf.predict_proba(feat)[0]
        confidence = {
            le.classes_[i]: round(float(p) * 100, 1)
            for i, p in enumerate(proba)
        }
        today.append({
            'hour': hour,
            'multiplier': round(mult, 3),
            'confidence': confidence
        })

    return {
        'weekly_predictions': weekly,
        'today_predictions': today,
        'model_info': {
            'regressor': 'GradientBoostingRegressor',
            'classifier': 'RandomForestClassifier',
            'features': ['hour', 'day_of_week', 'is_weekend', 'sin_hour', 'cos_hour'],
            'n_estimators': 100,
            'training_samples': 168
        }
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("Cairo Traffic ML Pipeline")
    print("=" * 50)

    records = generate_training_data()
    print(f"\nGenerated {len(records)} training samples")

    with open('traffic_data.json', 'w') as f:
        json.dump(records, f, indent=2)
    print("Saved → traffic_data.json")

    X, y_reg, y_cls = build_features(records)
    print(f"Feature matrix: {X.shape}")

    reg, clf, le = train_models(X, y_reg, y_cls)

    predictions = generate_predictions(reg, clf, le)

    with open('ml_predictions.json', 'w') as f:
        json.dump(predictions, f, separators=(',', ':'))
    print("\nSaved → ml_predictions.json")

    print("\nSample predictions:")
    print(f"  Mon 8am  → {predictions['weekly_predictions']['Mon'][8]}")
    print(f"  Mon 5pm  → {predictions['weekly_predictions']['Mon'][17]}")
    print(f"  Fri 8am  → {predictions['weekly_predictions']['Fri'][8]}")
    print(f"  Sat 3am  → {predictions['weekly_predictions']['Sat'][3]}")
    print("\nDone! Embed ml_predictions.json in index.html to update the web app.")
