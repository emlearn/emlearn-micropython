#!/usr/bin/env python3
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {np.unique(y)} (0=malignant, 1=benign)")
print(f"Class distribution: {np.bincount(y)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to int16 range (0-1000) for MicroPython compatibility
X_train_int = ((X_train_scaled + 3) / 6 * 1000).astype(np.int16)
X_test_int = ((X_test_scaled + 3) / 6 * 1000).astype(np.int16)

# Clip to valid range
X_train_int = np.clip(X_train_int, 0, 1000)
X_test_int = np.clip(X_test_int, 0, 1000)

print(f"Train: {X_train_int.shape}, Test: {X_test_int.shape}")
print(f"Feature range: [{X_train_int.min()}, {X_train_int.max()}]")

# Sklearn baseline
clf = ExtraTreesClassifier(
    n_estimators=20,
    max_depth=10, 
    min_samples_leaf=2,
    random_state=42
)

clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
baseline_acc = accuracy_score(y_test, y_pred)

print(f"\nSklearn ExtraTrees baseline: {baseline_acc:.3f}")
print(classification_report(y_test, y_pred, target_names=['malignant', 'benign']))

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
