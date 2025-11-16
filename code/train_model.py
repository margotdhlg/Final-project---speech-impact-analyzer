import pandas as pd
import numpy as np
import pickle
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from preprocess import clean_text
from features import rhetorical_features

# ================================
# LOAD DATASET
# ================================
print("🚀 Loading dataset...")
df = pd.read_csv("../merged_dataset_full.csv", low_memory=True)

print("Columns detected:", df.columns.tolist())

required = ["text", "likes", "retweets"]
for col in required:
    if col not in df.columns:
        raise KeyError(f"❌ Missing required column: {col}")

# ================================
# CREATE ENGAGEMENT LABEL
# ================================
print("\n📊 Creating engagement labels...")

df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0)
df["retweets"] = pd.to_numeric(df["retweets"], errors="coerce").fillna(0)

df["engagement_score"] = df["likes"] + df["retweets"]

df["engagement_level"] = pd.cut(
    df["engagement_score"],
    bins=[-1, 1, 20, 999999999],
    labels=["Low", "Medium", "High"]
)

print(df["engagement_level"].value_counts())

# ================================
# CLEAN TEXT
# ================================
print("\n🧹 Cleaning text...")
df = df.dropna(subset=["text"])
df["clean_text"] = df["text"].apply(clean_text)

# ================================
# RHETORICAL FEATURES
# ================================
print("\n🧠 Extracting rhetorical features...")
rhet_arr = np.vstack(df["clean_text"].apply(rhetorical_features))
X_rhet = csr_matrix(rhet_arr)

# ================================
# TF-IDF
# ================================
print("\n📚 Building TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df["clean_text"])

# Combine sparse matrices
print("\n🔗 Combining sparse features...")
X_full = hstack([X_tfidf, X_rhet])
y = df["engagement_level"].values

# ================================
# TRAIN / TEST SPLIT
# ================================
print("\n✂️ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# COMPUTE CLASS WEIGHTS MANUALLY
# ================================
print("\n⚖️ Computing balanced class weights...")

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weight_dict = {c: w for c, w in zip(classes, weights)}

print("Class weights:", class_weight_dict)

# ================================
# SGD LOGISTIC REGRESSION (WITH SAMPLE WEIGHTS)
# ================================
print("\n🔥 Training model with real-time progress...")

EPOCHS = 30
BATCH_SIZE = 2000

model = SGDClassifier(
    loss="log_loss",
    learning_rate="optimal",
    max_iter=1,
    tol=None
)

num_samples = X_train.shape[0]
num_batches = num_samples // BATCH_SIZE + 1

for epoch in range(1, EPOCHS + 1):
    print(f"\n📘 Epoch {epoch}/{EPOCHS}")

    idx = np.random.permutation(num_samples)

    for b in tqdm(range(num_batches), desc=f"Epoch {epoch}", ncols=100):
        start = b * BATCH_SIZE
        end = min(start + BATCH_SIZE, num_samples)

        batch_idx = idx[start:end]

        X_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]

        # Assign weight per sample
        sample_weight = np.array([class_weight_dict[label] for label in y_batch])

        if epoch == 1 and b == 0:
            model.partial_fit(X_batch, y_batch, classes=classes, sample_weight=sample_weight)
        else:
            model.partial_fit(X_batch, y_batch, sample_weight=sample_weight)

print("\n🎉 Training complete!")

# ================================
# EVALUATION
# ================================
print("\n📈 ===== CLASSIFICATION REPORT =====")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print("\n📉 ===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))

# ================================
# SAVE MODEL
# ================================
print("\n💾 Saving model + vectorizer...")
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n✅ Training finished successfully! Model ready for app.py")
