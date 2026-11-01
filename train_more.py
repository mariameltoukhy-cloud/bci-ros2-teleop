import sys, numpy as np, mne, joblib
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from features import bandpass, WIN, FS

mne.set_log_level("ERROR")
SUBJ = [int(x) for x in sys.argv[1].split(",")]
RUNS = [int(x) for x in sys.argv[2].split(",")]
SAVE = len(sys.argv) > 3 and sys.argv[3] == "save"
TARGET, TEST_RUN = 7, 12
USE_CH = False   # set True to use the 17 motor channels
CH = ["FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
      "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4"]


def load(s, r):
    raw = mne.io.read_raw_edf(eegbci.load_data(s, [r])[0],
                              preload=True, verbose=False)
    eegbci.standardize(raw)
    return raw


names = load(TARGET, 4).ch_names
IDX = [names.index(c) for c in CH] if USE_CH else list(range(len(names)))


def crops(raw, step=80):
    ev, _ = mne.events_from_annotations(
        raw, event_id={"T0": 1, "T1": 2, "T2": 3}, verbose=False)
    data = raw.get_data()[IDX] * 1e6
    X, y = [], []
    for s, _, lab in ev:
        seg = data[:, s + int(0.5 * FS): s + int(3.5 * FS)]
        for off in range(0, seg.shape[1] - WIN + 1, step):
            X.append(bandpass(seg[:, off:off + WIN]))
            y.append(lab - 1)
    return X, y


Xtr, ytr = [], []
for s in SUBJ:
    for r in RUNS:
        try:
            X, y = crops(load(s, r))
            Xtr += X
            ytr += y
        except Exception as e:
            print("skip", s, r, e)
Xtr, ytr = np.array(Xtr), np.array(ytr)
X, y = crops(load(TARGET, TEST_RUN))
Xte, yte = np.array(X), np.array(y)
print("train", Xtr.shape, "test", Xte.shape)

clf = make_pipeline(CSP(n_components=6, reg="oas", log=True), LDA())
clf.fit(Xtr, ytr)
pred = clf.predict(Xte)
print("held-out accuracy:", round(float((pred == yte).mean()), 3))
cm = confusion_matrix(yte, pred)
print("confusion (rows=truth rest/left/right):")
print(cm)
print("recall per class:", np.round(cm.diagonal() / cm.sum(1), 2))
if SAVE:
    np.save("chan_idx.npy", np.array(IDX))
    joblib.dump(clf, "csp_lda.joblib")
    print("saved model + chan_idx.npy")
