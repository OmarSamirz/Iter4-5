import time, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from pathlib import Path

DATA_PATH = Path("data")
TRAIN_VAL_DATA_PATH = DATA_PATH / "train_val.csv"
TEST_DATA_PATH = DATA_PATH / "test.csv"

# --- configure files / columns here ---
TRAIN = TRAIN_VAL_DATA_PATH
TEST  = TEST_DATA_PATH
TEXT_COL = "cleaned_item_name"  # column used for TF-IDF (must exist in both files)
ORIG_COL  = "Item_Name"         # original (pre-cleaning) name column in test file
LABEL_COL = "class"
OUT = "predictions-centroid.csv"
# --------------------------------------

tr = pd.read_csv(TRAIN)
te = pd.read_csv(TEST)

# fallback if chosen text col isn't present
if TEXT_COL not in tr.columns: TEXT_COL = tr.columns[0]
if TEXT_COL not in te.columns: raise SystemExit(f"{TEXT_COL} not in test file")

tr_text = tr[TEXT_COL].astype(str).tolist()
tr_lbl  = tr[LABEL_COL].astype(str).tolist()
te_text = te[TEXT_COL].astype(str).tolist()
te_lbl  = te[LABEL_COL].astype(str).tolist()
orig    = te[ORIG_COL].astype(str).tolist() if ORIG_COL in te.columns else te_text

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2, max_df=0.9,
                      sublinear_tf=True, dtype=np.float32, token_pattern=None)
Xtr = vec.fit_transform(tr_text)

classes = sorted(set(tr_lbl))
centroids = []
for c in classes:
    mask = [lbl==c for lbl in tr_lbl]
    Xc = Xtr[mask]
    centroids.append(np.asarray(Xc.mean(axis=0)).ravel() if Xc.shape[0] else np.zeros(Xtr.shape[1], dtype=np.float32))
C = np.vstack(centroids)

t0 = time.perf_counter()
Xte = vec.transform(te_text)
scores = cosine_similarity(Xte, C)
preds = [classes[i] for i in scores.argmax(axis=1)]
t_elapsed = time.perf_counter() - t0

try:
    f1 = f1_score(te_lbl, preds, average="weighted")
except:
    f1 = None

out = pd.DataFrame({
    "original_item_name": orig,
    "true_class": te_lbl,
    "predicted_class": preds,
})
out["match"] = (out["true_class"] == out["predicted_class"]).astype(int)
out["prediction_time_seconds"] = t_elapsed
out.to_csv(OUT, index=False, encoding="utf-8-sig")

print(f"Saved {len(out)} rows to {OUT} | F1={f1} | predict_time={t_elapsed:.4f}s")