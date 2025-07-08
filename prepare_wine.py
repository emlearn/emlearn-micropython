#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load both red and white wine datasets
red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

red = pd.read_csv(red_url, sep=';')
white = pd.read_csv(white_url, sep=';')

# Add wine type feature (0=red, 1=white)
red['wine_type'] = 0
white['wine_type'] = 1

# Combine datasets
data = pd.concat([red, white], ignore_index=True)

# Convert quality to binary classification (good wine: quality >= 6)
X = data.drop('quality', axis=1).values
y = (data['quality'] >= 6).astype(int).values

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: 0=poor wine (<6), 1=good wine (>=6)")
print(f"Class distribution: {np.bincount(y)}")
print(f"Good wine ratio: {y.mean():.2f}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to int16 range (0-1000) for MicroPython
X_train_int = ((X_train_scaled + 3) / 6 * 1000).astype(np.int16)
X_test_int = ((X_test_scaled + 3) / 6 * 1000).astype(np.int16)

X_train_int = np.clip(X_train_int, 0, 1000)
X_test_int = np.clip(X_test_int, 0, 1000)

print(f"Train: {X_train_int.shape}, Test: {X_test_int.shape}")
print(f"Feature range: [{X_train_int.min()}, {X_train_int.max()}]")

# Sklearn baseline
clf = ExtraTreesClassifier(
    n_estimators=30,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42
)

clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
baseline_acc = accuracy_score(y_test, y_pred)

print(f"\nSklearn ExtraTrees baseline: {baseline_acc:.3f}")
print(classification_report(y_test, y_pred, target_names=['poor_wine', 'good_wine']))

# Feature importance
feature_names = list(data.columns[:-1])  # Exclude 'quality'
importances = clf.feature_importances_
top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
print(f"\nTop 5 features:")
for name, imp in top_features[:5]:
    print(f"  {name}: {imp:.3f}")

# Save data for MicroPython
np.save('X_train.npy', X_train_int)
np.save('y_train.npy', y_train.astype(np.int16))
np.save('X_test.npy', X_test_int)
np.save('y_test.npy', y_test.astype(np.int16))

print(f"\nSaved files:")
print(f"X_train.npy: {X_train_int.shape} int16")
print(f"y_train.npy: {y_train.shape} int16")
print(f"X_test.npy: {X_test_int.shape} int16")
print(f"y_test.npy: {y_test.shape} int16")
print(f"\nTarget accuracy: {baseline_acc:.3f}")
