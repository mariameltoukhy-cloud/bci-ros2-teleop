import numpy as np, mne, joblib
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from features import bandpass, WIN, FS

SUBJECT = 7
mne.set_log_level("WARNING")

def load(run):
    f = eegbci.load_data(SUBJECT, [run])[0]
    raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
    eegbci.standardize(raw)
    return raw

def crops(raw, step=80):
    ev, _ = mne.events_from_annotations(raw, event_id={"T0": 1, "T1": 2, "T2": 3}, verbose=False)
    data = raw.get_data() * 1e6
    X, y = [], []
    for s, _, lab in ev:
        seg = data[:, s + int(0.5*FS): s + int(3.5*FS)]
        for off in range(0, seg.shape[1] - WIN + 1, step):
            X.append(bandpass(seg[:, off:off+WIN])); y.append(lab - 1)
    return np.array(X), np.array(y)

parts = [crops(load(r)) for r in (4, 8)]
Xtr = np.concatenate([p[0] for p in parts]); ytr = np.concatenate([p[1] for p in parts])
Xte, yte = crops(load(12))
print("train", Xtr.shape, "test", Xte.shape)

clf = make_pipeline(CSP(n_components=6, reg="oas", log=True), LDA())
clf.fit(Xtr, ytr)
print("held-out accuracy (run 12):", clf.score(Xte, yte))
joblib.dump(clf, "csp_lda.joblib")
