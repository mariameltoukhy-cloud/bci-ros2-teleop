import csv, numpy as np

rows = list(csv.DictReader(open("log.csv")))
M = {"T0": 0, "T1": 1, "T2": 2}
names = ["rest", "left", "right"]
t = np.array([float(r["t"]) for r in rows])
tr = [r["truth"] for r in rows]
turn = [float(r["turn"]) for r in rows]
pred = [0 if abs(w) < 1e-9 else (1 if w > 0 else 2) for w in turn]

# split into segments of constant truth label
segs, s = [], 0
for i in range(1, len(rows) + 1):
    if i == len(rows) or tr[i] != tr[s]:
        if tr[s] in M:
            segs.append((s, i))
        s = i

conf = np.zeros((3, 3), int)
all_ok, all_n, set_ok, set_n, lats = 0, 0, 0, 0, []
for a, b in segs:
    y = M[tr[a]]
    found = False
    for i in range(a, b):
        ok = pred[i] == y
        all_n += 1; all_ok += ok
        conf[y, pred[i]] += 1
        if t[i] - t[a] >= 1.0:          # after 1 s settling time
            set_n += 1; set_ok += ok
        if ok and not found:
            lats.append(t[i] - t[a]); found = True

print(f"decisions: {all_n}")
print(f"accuracy (all):          {all_ok / all_n:.2f}")
print(f"accuracy (after 1 s):    {set_ok / max(set_n, 1):.2f}")
print(f"median latency to first correct decision: {np.median(lats):.2f} s")
print("confusion (rows=truth, cols=decision):")
print("        " + "  ".join(f"{n:>5}" for n in names))
for k in range(3):
    print(f"{names[k]:>6}  " + "  ".join(f"{v:5d}" for v in conf[k]))
