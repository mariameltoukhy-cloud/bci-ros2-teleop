import time, mne, numpy as np
from mne.datasets import eegbci
from pylsl import StreamInfo, StreamOutlet

SUBJECT, RUN, CHUNK, FS = 7, 12, 16, 160
mne.set_log_level("WARNING")

raw = mne.io.read_raw_edf(eegbci.load_data(SUBJECT, [RUN])[0], preload=True, verbose=False)
eegbci.standardize(raw)
data = (raw.get_data() * 1e6).T.astype(np.float32)      # (n_samples, 64) in µV
print("loaded", data.shape, "streaming at", FS, "Hz")

info = StreamInfo("FakeEEG", "EEG", data.shape[1], FS, "float32", "sub1")
outlet = StreamOutlet(info, chunk_size=CHUNK)
mk = StreamOutlet(StreamInfo("Markers", "Markers", 1, 0, "string", "sub1_mk"))

onsets = {int(a["onset"] * FS): a["description"] for a in raw.annotations}

while True:                                              # loop the recording forever
    t0 = time.perf_counter()
    for i in range(0, len(data) - CHUNK, CHUNK):
        for j in range(i, i + CHUNK):
            if j in onsets:
                mk.push_sample([onsets[j]])
        outlet.push_chunk(data[i:i+CHUNK].tolist())
        time.sleep(max(0, t0 + (i + CHUNK) / FS - time.perf_counter()))
    print("looped")
