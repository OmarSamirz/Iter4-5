import time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

DATA_PATH=Path("data"); TRAIN=DATA_PATH/"train_val.csv"; TEST=DATA_PATH/"test.csv"
TEXT_COL="cleaned_item_name"; ORIG_COL="Item_Name"; LABEL_COL="class"; OUT="predictions-logreg.csv"

tr=pd.read_csv(TRAIN); te=pd.read_csv(TEST)
if TEXT_COL not in tr.columns: TEXT_COL=tr.columns[0]
if TEXT_COL not in te.columns: raise SystemExit(f"{TEXT_COL} not in test file")

tr_text=tr[TEXT_COL].astype(str).tolist(); tr_lbl=tr[LABEL_COL].astype(str).tolist()
te_text=te[TEXT_COL].astype(str).tolist(); te_lbl=te[LABEL_COL].astype(str).tolist()
orig=te[ORIG_COL].astype(str).tolist() if ORIG_COL in te.columns else te_text

vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,max_df=0.9,sublinear_tf=True,dtype=np.float32,token_pattern=None)
Xtr=vec.fit_transform(tr_text); Xte=vec.transform(te_text)

clf=LogisticRegression(max_iter=200, n_jobs=-1)
t0=time.perf_counter(); clf.fit(Xtr, tr_lbl); preds=clf.predict(Xte); t=time.perf_counter()-t0
f1=f1_score(te_lbl, preds, average="weighted")

out=pd.DataFrame({"original_item_name":orig,"true_class":te_lbl,"predicted_class":preds})
out["match"]=(out.true_class==out.predicted_class).astype(int); out["prediction_time_seconds"]=t
out.to_csv(OUT,index=False,encoding="utf-8-sig")
print(f"Saved {len(out)} rows to {OUT} | F1={f1} | predict_time={t:.4f}s")