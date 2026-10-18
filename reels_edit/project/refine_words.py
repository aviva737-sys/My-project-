"""Refine word onsets: inside (t-0.25, t] pick the 30fps frame where the author's
subtitle band changed most (band diff signal), then spread words that share a frame."""
import sys, json
raw, band, out = sys.argv[1:4]
diff = [float(l.split()[2]) for l in open(band)]
pages, cur = [], []
for line in open(raw, encoding="utf-8"):
    line = line.rstrip("\n")
    if not line or line.startswith("#"): continue
    if line == "--":
        pages.append(cur); cur = []; continue
    w, t = line.split("\t"); cur.append([w, float(t)])
pages.append(cur)
def refine(t):
    if t <= 0.01: return 0.0
    i1 = int(round(t * 30)); i0 = max(0, i1 - 7)
    best = max(range(i0 + 1, i1 + 1), key=lambda i: diff[i] if i < len(diff) else 0)
    return best / 30
res = []
for p in pages:
    words = []
    cache = {}
    for w, t in p:
        if t not in cache: cache[t] = refine(t)
        words.append([w, cache[t], t])
    # spread words sharing the same onset by 0.12 s
    for i in range(1, len(words)):
        if words[i][0] != "END" and abs(words[i][2] - words[i-1][2]) < 1e-6:
            words[i][1] = words[i-1][1] + 0.12
    res.append([[w, round(t, 3)] for w, t, _ in words])
json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
for p in res: print(" ".join(f"{w}@{t:.2f}" for w, t in p))
