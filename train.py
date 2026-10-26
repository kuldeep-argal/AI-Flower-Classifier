import joblib
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load dataset directly from scikit-learn
data = load_wine(as_frame=True)
X = data.data[['alcohol', 'malic_acid', 'color_intensity', 'flavanoids', 'proline']]
y = data.target
target_names = list(data.target_names)

# 2. Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=60, random_state=42)
clf.fit(X_train, y_train)

# 3. Evaluate
acc = accuracy_score(y_test, clf.predict(X_test))
print(f"Model trained with accuracy: {acc * 100:.2f}%")

# 4. Save model and metadata together
artifact = {
    "model": clf,
    "features": list(X.columns),
    "classes": target_names
}
joblib.dump(artifact, "wine_classifier.pkl")
print("Saved to wine_classifier.pkl")