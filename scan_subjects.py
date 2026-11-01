import numpy as np, mne
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from features import bandpass, WIN, FS

mne.set_log_level("ERROR")

def load(subj, run):
    raw = mne.io.read_raw_edf(eegbci.load_data(subj, [run], update_path=False)[0],
                              preload=True, verbose=False)
    eegbci.standardize(raw)
    return raw

def crops(raw, step=80):
    ev, _ = mne.events_from_annotations(raw, event_id={"T1": 1, "T2": 2}, verbose=False)
    data = raw.get_data() * 1e6
    X, y = [], []
    for s, _, lab in ev:
        seg = data[:, s + int(0.5*FS): s + int(3.5*FS)]
        for off in range(0, seg.shape[1] - WIN + 1, step):
            X.append(bandpass(seg[:, off:off+WIN])); y.append(lab - 1)
    return np.array(X), np.array(y)

for subj in range(1, 21):
    try:
        p = [crops(load(subj, r)) for r in (4, 8)]
        Xtr = np.concatenate([a for a, _ in p]); ytr = np.concatenate([b for _, b in p])
        Xte, yte = crops(load(subj, 12))
        clf = make_pipeline(CSP(n_components=6, reg="oas", log=True), LDA()).fit(Xtr, ytr)
        print(f"subject {subj:2d}: {clf.score(Xte, yte):.2f}", flush=True)
    except Exception as e:
        print(f"subject {subj:2d}: failed ({e})", flush=True)
