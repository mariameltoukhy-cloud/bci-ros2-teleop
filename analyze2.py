import csv, sys, numpy as np
from math import log2

fn = sys.argv[1] if len(sys.argv) > 1 else "log.csv"
rows = list(csv.DictReader(open(fn)))
M = {"T0": 0, "T1": 1, "T2": 2}
NAMES = ["rest", "left", "right"]
t = np.array([float(r["t"]) for r in rows])
tr = [r["truth"] for r in rows]
turn = [float(r["turn"]) for r in rows]
pred = [0 if abs(w) < 1e-9 else (1 if w > 0 else 2) for w in turn]

# split into segments of constant truth label
segs, s = [], 0
for i in range(1, len(rows) + 1):
    if i == len(rows) or tr[i] != tr[s]:
        segs.append((s, i))
        s = i
# drop first/last segment (may be cut off by start/stop) and unlabeled ones
segs = [(a, b) for a, b in segs[1:-1] if tr[a] in M]

lat = {1: [], 2: []}
n = {1: 0, 2: 0}
miss = {1: 0, 2: 0}
wrong_first = {1: 0, 2: 0}
true_l, pred_l, durs = [], [], []

for a, b in segs:
    y = M[tr[a]]
    durs.append(t[b - 1] - t[a])
    late = [i for i in range(a, b) if t[i] - t[a] >= 1.0]
    if late:
        true_l.append(y)
        pred_l.append(int(np.bincount([pred[i] for i in late], minlength=3).argmax()))
    if y == 0:
        continue
    n[y] += 1
    hit = next((i for i in range(a, b) if pred[i] == y), None)
    first_turn = next((i for i in range(a, b) if pred[i] != 0), None)
    if hit is None:
        miss[y] += 1
    else:
        lat[y].append(t[hit] - t[a])
    if first_turn is not None and pred[first_turn] != y:
        wrong_first[y] += 1

print(f"file: {fn}   labeled segments used: {len(segs)}")
print("\nPer-cue results (left/right cues only):")
for y in (1, 2):
    L = lat[y]
    med = f"{np.median(L):.2f} s" if L else "n/a"
    p90 = f"{np.percentile(L, 90):.2f} s" if L else "n/a"
    print(f"  {NAMES[y]:>5}: cues={n[y]:2d}  detected={n[y]-miss[y]:2d}"
          f" ({(n[y]-miss[y])/max(n[y],1):.0%})  missed={miss[y]}"
          f"  wrong-direction-first={wrong_first[y]}"
          f"  latency median={med}  p90={p90}")

# trial-level accuracy and information transfer rate (Wolpaw)
P = float(np.mean(np.array(true_l) == np.array(pred_l)))
N = 3
bits = log2(N)
if 0 < P < 1:
    bits += P * log2(P) + (1 - P) * log2((1 - P) / (N - 1))
elif P == 0:
    bits += log2(1 / (N - 1)) * 0 + 0
T = float(np.median(durs)) + 0.3
print(f"\nTrial-level accuracy (majority vote after 1 s): {P:.2f} over {len(true_l)} trials")
print(f"Median trial duration: {T:.1f} s")
print(f"Bits per trial: {bits:.2f}   ITR: {bits * 60 / T:.1f} bits/min")
