import os, csv, hashlib, pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

IN  = "artifacts/policy/dataset_pl2.csv"
OUTB= "artifacts/policy/policy.bin"
OUTJ= "artifacts/policy/policy.json"
os.makedirs("artifacts/policy", exist_ok=True)

rows=list(csv.reader(open(IN))); hdr=rows[0]; rows=rows[1:]
X=np.array([[float(v) for v in r[2:]] for r in rows], dtype=float)
y=np.array([int(r[1]) for r in rows], dtype=int)

Xtr,Xva,ytr,yva = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
pipe = Pipeline([("scaler",StandardScaler()), ("clf",LogisticRegression(max_iter=400,class_weight="balanced"))])
pipe.fit(Xtr,ytr)
auc = roc_auc_score(yva, pipe.predict_proba(Xva)[:,1])

raw = pickle.dumps({"type":"sklearn_pipeline","blob":pipe})
h = hashlib.sha256(raw).hexdigest()
open(OUTB,"wb").write(raw)
open(OUTJ,"w").write('{"name":"PL2-LogReg","auc":%.4f,"hash":"%s"}' % (auc,h))
print("policy.bin hash=",h," AUC=",round(auc,4))
