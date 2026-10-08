"""Builds metrics/results.md and metrics/charts/*.png from the raw result files.

python3 usecase/report.py            tables and charts
python3 usecase/report.py --no-png   no Chrome needed
"""
import collections, hashlib, html, json, math, os, random, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.environ.get("RAW_DIR") or os.path.join(HERE, "..", "metrics", "raw")
OUT = os.environ.get("OUT_DIR") or os.path.join(HERE, "..", "metrics")
CHARTS = os.path.join(OUT, "charts")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

TEAMS = ["Credit reporting", "Debt collection", "Mortgage", "Credit card", "Bank account",
         "Student loan", "Vehicle loan", "Personal loan", "Money transfer"]
NAME = {"n": "Nimble 9B", "q": "Qwen3.5-9B, reasoning off", "t": "Qwen3.5-9B, reasoning on"}
SHORT = {"n": "Nimble 9B", "q": "Qwen, reasoning off", "t": "Qwen, reasoning on"}
COL = {"n": "#2a78d6", "q": "#eb6834", "t": "#1baf7a"}
PAPER, INK, SOFT, FAINT, GRID = "#f7f2e7", "#1b1e2b", "#4a4e5e", "#80859a", "#1b1e2b22"
FOOT = "CFPB consumer complaints · MacBook Pro 14-inch, Apple M1 Pro, 32 GB · Ollama 0.35.1 · both models 8-bit"


def load(name):
    path = os.path.join(RAW, name)
    return {r["id"]: r for r in map(json.loads, open(path))} if os.path.exists(path) else {}


N, Q, T = load("res-nimble.jsonl"), load("res-qwen3.5_9b-q8_0.jsonl"), load("res-qwen3.5_9b-q8_0-think.jsonl")
RES = {"n": N, "q": Q, "t": T}
LET = load("res-qwen3.5_9b-q8_0-letters.jsonl")                  # Qwen answering with one letter per team
ALL = [i for i in N if i in Q]
T_ERR = [i for i in T if T[i]["pred"] is None]                   # failed reasoning requests
SUB = [i for i in ALL if i in T and T[i]["pred"] is not None]    # answered with reasoning on
FULL = bool(SUB) and len(SUB) == len(ALL)
MODELS = "nqt" if FULL else "nq"


def inside(v, lo, hi, what):
    """Fail if a value is outside the axis."""
    if not lo <= v <= hi:
        raise ValueError(f"{what}: {v} is outside the axis range {lo}..{hi}")
    return v


def ok(r):
    return r["pred"] == r["gold"]


def acc(R, ids):
    k = sum(ok(R[i]) for i in ids)
    p = k / len(ids)
    return k, p, 1.96 * math.sqrt(p * (1 - p) / len(ids))


def mcnemar(A, B, ids):
    a = sum(ok(A[i]) and not ok(B[i]) for i in ids)
    b = sum(ok(B[i]) and not ok(A[i]) for i in ids)
    d = a + b
    p = min(1.0, 2 * sum(math.comb(d, k) for k in range(min(a, b) + 1)) / 2 ** d) if d else 1.0
    return a, b, p


def curve(R, ids, key=lambda r: -r["p"]):
    """(share routed, accuracy) after the k most confident answers."""
    rs = sorted((R[i] for i in ids), key=key)
    pts, good = [], 0
    for k, r in enumerate(rs, 1):
        good += ok(r)
        pts.append((k / len(rs), good / k))
    return pts


def coverage_at(pts, target, min_k=30):
    best = 0.0
    for k, (c, a) in enumerate(pts, 1):
        if k >= min_k and a >= target:
            best = c
    return best


def boot(R, ids, target, n=2000):
    """95% interval of the routed share, from resampling the complaints."""
    rng = random.Random(1)
    xs = sorted(coverage_at(curve(R, [rng.choice(ids) for _ in ids]), target) for _ in range(n))
    return xs[int(.025 * n)], xs[int(.975 * n)]


def auroc(R, ids):
    """Chance that a right answer has a higher probability than a wrong one."""
    right = [R[i]["p"] for i in ids if ok(R[i])]
    wrong = [R[i]["p"] for i in ids if not ok(R[i])]
    return sum((a > b) + .5 * (a == b) for a in right for b in wrong) / (len(right) * len(wrong))


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def secs(R, ids):
    return sorted(R[i]["sec"] for i in ids)


def dur(s):
    return f"{s / 60:.0f} min" if s < 5400 else f"{int(s // 3600)} h {int(s % 3600 // 60):02d} min"


def head(models):
    return "| " + " | ".join(NAME[m] for m in models) + " |", "|---" * len(models) + "|"


def fit(R, ids):
    """Least-squares line sec = a + b * tokens, without cache hits."""
    pts = [(R[i]["in_tok"], R[i]["sec"]) for i in ids if R[i]["sec"] > 1]
    mx, my = statistics.mean(x for x, _ in pts), statistics.mean(y for _, y in pts)
    b = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
    return my - b * mx, b, sum(R[i]["sec"] <= 1 for i in ids)


def cascade(first, cut, ids):
    """First model answers all; below the cut-off the reasoning model answers again."""
    F = RES[first]
    up = {i for i in ids if F[i]["p"] < cut}
    good = sum(ok(T[i]) if i in up else ok(F[i]) for i in ids)
    return {"up": up, "acc": good / len(ids), "sec": sum(F[i]["sec"] for i in ids) + sum(T[i]["sec"] for i in up)}


def cache_rows():
    rows = []
    if not os.path.exists(os.path.join(RAW, "cache-test.txt")):
        return rows
    for line in open(os.path.join(RAW, "cache-test.txt")):
        if "median" in line and line[0] in "ABCDEFGHIJ" and line[1] == " ":
            label, rest = line[2:].split("median")
            rows.append((line[0], label.strip(), float(rest.split()[0]), float(rest.split("tokens")[1].split()[0])))
    return rows


# ---------------------------------------------------------------- tables

def results_md():
    L = ["# Results", ""]
    L += [f"## 1. Routing accuracy, {len(ALL)} complaints", "", "| Model | Correct | Accuracy | 95% margin | Invalid answers |", "|---|---|---|---|---|"]
    for m in "nq":
        k, p, ci = acc(RES[m], ALL)
        L.append(f"| {NAME[m]} | {k} / {len(ALL)} | {pct(p)} | ±{pct(ci)} | {sum(RES[m][i]['pred'] is None for i in ALL)} |")
    a, b, p = mcnemar(N, Q, ALL)
    both = sum(ok(N[i]) and ok(Q[i]) for i in ALL)
    neither = [i for i in ALL if not ok(N[i]) and not ok(Q[i])]
    same = [i for i in neither if N[i]["pred"] == Q[i]["pred"]]
    L += ["", f"Paired: both right {both}, only Nimble {a}, only Qwen {b}, both wrong {len(neither)} "
              f"(same wrong answer from both: {len(same)}). McNemar exact p = {p:.3f}.", ""]

    h, sep = head(MODELS)
    L += [f"## 2. Per team ({len(ALL) // len(TEAMS)} complaints each)", "", "| Team " + h, "|---" + sep]
    for t in TEAMS:
        ids = [i for i in ALL if N[i]["gold"] == t]
        L.append(f"| {t} | " + " | ".join(str(sum(ok(RES[m][i]) for i in ids)) for m in MODELS) + " |")

    L += ["", "Times each model answered Credit reporting (100 complaints belong there): "
          + ", ".join(f"{NAME[m]} {sum(RES[m][i]['pred'] == 'Credit reporting' for i in ALL)}" for m in MODELS) + "."]

    L += ["", "## 3. Shared misses (both models gave the same wrong team)", "", "| Labelled as | Both models said | Complaints |", "|---|---|---|"]
    for (g, pr), k in collections.Counter((N[i]["gold"], N[i]["pred"]) for i in same).most_common(8):
        L.append(f"| {g} | {pr} | {k} |")

    agree = [i for i in ALL if N[i]["pred"] == Q[i]["pred"]]
    sure = [i for i in ALL if N[i]["p"] >= .95]
    both_ = [i for i in sure if i in set(agree)]
    L += ["", f"Nimble and Qwen (reasoning off) gave the same answer for {len(agree)} complaints, {pct(acc(N, agree)[1])} of them right. "
              f"Nimble at 0.95 or more: {pct(len(sure) / len(ALL), 0)} of complaints, {pct(acc(N, sure)[1])} right. "
              f"Nimble at 0.95 or more and Qwen agrees: {pct(len(both_) / len(ALL), 0)}, {pct(acc(N, both_)[1])} right."]

    L += ["", "## 4. Share routed automatically at a required accuracy", "", "Most confident answers first, counted from 30 answers upward.", "",
          "| Required accuracy " + h, "|---" + sep]
    for target in (0.85, 0.90, 0.95):
        L.append(f"| {pct(target, 0)} | " + " | ".join(pct(coverage_at(curve(RES[m], ALL), target), 0) for m in MODELS) + " |")

    L += ["", "## 5. Stated probability and hit rate", "", "Share correct, number of answers in brackets.", "",
          "| Stated probability " + h, "|---" + sep]
    for lo, up in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01), (0.99, 1.01)):
        cells = []
        for m in MODELS:
            b_ = [ok(RES[m][i]) for i in ALL if lo <= RES[m][i]["p"] < up]
            cells.append(f"{pct(sum(b_) / len(b_), 0)} ({len(b_)})" if b_ else "none")
        L.append(f"| {f'{lo:.2f} or more' if lo == 0.99 else f'{lo:.1f} to {min(up, 1.0):.1f}'} | " + " | ".join(cells) + " |")

    L += ["", f"## 6. Speed, {len(ALL)} complaints", "", "| Model | Median s | Slowest 5% s | Total min | Median input tokens |", "|---|---|---|---|---|"]
    for m in "nq":
        s = secs(RES[m], ALL)
        L.append(f"| {NAME[m]} | {statistics.median(s):.2f} | {s[int(len(s) * .95)]:.2f} | {sum(s) / 60:.1f} | "
                 f"{statistics.median(RES[m][i]['in_tok'] for i in ALL):.0f} |")

    L += ["", "Time against input tokens, fitted line:", "",
          "| Model | Fixed part s | Reading rate tokens/s | Requests under 1 s (cache hits) |", "|---|---|---|---|"]
    for m in "nq":
        a0, b0, hits = fit(RES[m], ALL)
        L.append(f"| {NAME[m]} | {a0:.2f} | {1 / b0:.0f} | {hits} |")

    L += [""]

    if SUB:
        scope = f"all {len(SUB)} complaints" if FULL else f"first {len(SUB)} complaints"
        L += [f"## 7. With reasoning: {scope}", "",
              "| Model | Correct | Accuracy | 95% margin | Median s | Mean s | Longest s | Total |", "|---|---|---|---|---|---|---|---|"]
        for m in "nqt":
            k, p, ci = acc(RES[m], SUB)
            s = secs(RES[m], SUB)
            L.append(f"| {NAME[m]} | {k} / {len(SUB)} | {pct(p)} | ±{pct(ci)} | {statistics.median(s):.1f} | {statistics.mean(s):.1f} | {s[-1]:.0f} | {dur(sum(s))} |")
        for x, y in (("t", "n"), ("t", "q"), ("n", "q")):
            a, b, p = mcnemar(RES[x], RES[y], SUB)
            L += ["", f"{NAME[x]} vs {NAME[y]}: only first right {a}, only second right {b}, McNemar exact p = {p:.3f}."]
        st = secs(T, SUB)
        tot = {m: sum(secs(RES[m], SUB)) for m in "nqt"}
        right = [T[i]["sec"] for i in SUB if ok(T[i])]
        wrong = [T[i]["sec"] for i in SUB if not ok(T[i])]
        slow = max(SUB, key=lambda i: T[i]["sec"])
        L += ["", f"Total time with reasoning on is {tot['t'] / tot['n']:.0f} times Nimble's and {tot['t'] / tot['q']:.0f} times that of reasoning off.",
              "", f"Reasoning time per complaint: fastest {st[0]:.0f} s, median {statistics.median(st):.0f} s, slowest 5% above {st[int(len(st) * .95)]:.0f} s, longest {st[-1]:.0f} s. "
                  f"Over 1 minute: {sum(x > 60 for x in st)}. Over 3 minutes: {sum(x > 180 for x in st)}. Over 5 minutes: {sum(x > 300 for x in st)}.",
              "", f"Reasoning time when the answer was right: median {statistics.median(right):.0f} s. When wrong: median {statistics.median(wrong):.0f} s. "
                  f"Median output: {statistics.median(T[i]['out_tok'] for i in SUB):.0f} tokens. "
                  f"Longest: {T[slow]['out_tok']} tokens, {T[slow]['sec']:.0f} s, answer {'right' if ok(T[slow]) else 'wrong'}.",
              "", f"Run from {min(T[i]['at'] for i in SUB if T[i].get('at'))} to {max(T[i]['at'] for i in SUB if T[i].get('at'))}. Failed requests: {len(T_ERR)}.", ""]
        L += ["Share routed automatically. Last column: reasoning model ranked by reasoning length, shortest first.", "",
              "| Required accuracy " + head("nqt")[0] + " Reasoning on, ranked by length |", "|---" * 5 + "|"]
        by_len = curve(T, SUB, key=lambda r: r["out_tok"])
        for target in (0.85, 0.90, 0.95):
            L.append(f"| {pct(target, 0)} | " + " | ".join(pct(coverage_at(curve(RES[m], SUB), target), 0) for m in "nqt")
                     + f" | {pct(coverage_at(by_len, target), 0)} |")
        L.append("")
        t_all = sum(T[i]["sec"] for i in SUB)
        L += ["Cascade: the first model answers everything, answers below the cut-off go to the reasoning model. Time includes the first model.", "",
              "| First model | Escalate below | Escalated | Accuracy | Time | Time vs reasoning on everything |", "|---|---|---|---|---|---|"]
        for first in "nq":
            for cut in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
                c = cascade(first, cut, SUB)
                L.append(f"| {NAME[first]} | {'never' if not cut else f'{cut:.2f}'} | {pct(len(c['up']) / len(SUB), 0)} | {pct(c['acc'])} | {dur(c['sec'])} | {pct(c['sec'] / t_all, 0)} |")
        L += [f"| Reasoning on for every complaint | | 100% | {pct(acc(T, SUB)[1])} | {dur(t_all)} | 100% |", ""]
        up = [i for i in SUB if N[i]["p"] < .95]
        L += [f"The {len(up)} complaints where Nimble is below 0.95: reasoning on right {acc(T, up)[0]}, Nimble right {acc(N, up)[0]}. "
              f"The other {len(SUB) - len(up)}: reasoning on right {acc(T, [i for i in SUB if i not in set(up)])[0]}, "
              f"Nimble right {acc(N, [i for i in SUB if i not in set(up)])[0]}.", ""]

    L += ["## 8. What a cache hit changes (8 complaints per case, 20 for the short input)", "", "| Case | Median s | Tokens |", "|---|---|---|"]
    for key, label, s, tok in cache_rows():
        who = "Nimble" if key < "H" else "Qwen"
        L.append(f"| {who}: {label} | {s:.2f} | {tok:.0f} |")
    L.append("")

    L += ["## 9. Machine time", "", "Sum of request times per result file.", "",
          "| Run | Requests | Time |", "|---|---|---|"]
    total = 0.0
    for label, rel in (("Nimble 9B", "raw/res-nimble.jsonl"), ("Qwen3.5-9B, reasoning off", "raw/res-qwen3.5_9b-q8_0.jsonl"),
                       ("Qwen3.5-9B, reasoning on", "raw/res-qwen3.5_9b-q8_0-think.jsonl"),
                       ("Qwen3.5-9B, reasoning off, letter answers", "raw/res-qwen3.5_9b-q8_0-letters.jsonl")):
        path = os.path.join(RAW, "..", rel)
        if os.path.exists(path):
            rows = [json.loads(l) for l in open(path)]
            t = sum(r["sec"] or 0 for r in rows)
            total += t
            L.append(f"| {label} | {len(rows)} | {dur(t)} |")
    L += [f"| All runs | | {dur(total)} |", ""]

    ids = [i for i in ALL if i in LET and LET[i]["pred"] is not None]
    if ids:
        k, p, ci = acc(LET, ids)
        a, b, pv = mcnemar(N, LET, ids)
        L += [f"## 10. Qwen with a probability for all nine teams, {len(ids)} complaints", "",
              "Reasoning off. Qwen answers with one letter per team. The probabilities of the nine letters are scaled to add up to 1.", "",
              f"Accuracy {k} / {len(ids)}, {pct(p)} (±{pct(ci)}). Against Nimble 9B: only Nimble right {a}, only Qwen right {b}, McNemar exact p = {pv:.3f}.", "",
              f"Share of the probability that was on the nine letters before scaling: median {pct(statistics.median(LET[i]['found'] for i in ids), 0)}.", "",
              "| Required accuracy | Nimble 9B | Qwen, team name | Qwen, letter |", "|---|---|---|---|"]
        for target in (0.85, 0.90, 0.95):
            L.append(f"| {pct(target, 0)} | " + " | ".join(pct(coverage_at(curve(R, ids), target), 0) for R in (N, Q, LET)) + " |")
        L += ["", "The same shares with a 95% interval, from 2,000 resamples of the complaints:", "",
              "| Required accuracy | Nimble 9B | Qwen, team name | Qwen, letter |", "|---|---|---|---|"]
        for target in (0.90, 0.95):
            L.append(f"| {pct(target, 0)} | " + " | ".join("{} to {}".format(*(pct(x, 0) for x in boot(R, ids, target))) for R in (N, Q, LET)) + " |")
        rng = random.Random(1)
        draws = [[rng.choice(ids) for _ in ids] for _ in range(2000)]
        ahead = sum(coverage_at(curve(N, s), .95) > coverage_at(curve(LET, s), .95) for s in draws)
        L += ["", f"Nimble routes more than Qwen, letter, at 95% in {pct(ahead / len(draws), 0)} of the resamples."]
        raw = {i: dict(LET[i], p=LET[i]["p"] * LET[i]["found"]) for i in ids}
        L += ["", "Qwen, letter, without scaling to 1: " + ", ".join(f"{pct(coverage_at(curve(raw, ids), t), 0)} at {pct(t, 0)}" for t in (0.85, 0.90, 0.95)) + ".", "",
              "Chance that a right answer has a higher probability than a wrong one (0.5 is no better than a coin): "
              + ", ".join(f"{label} {auroc(R, ids):.2f}" for label, R in (("Nimble 9B", N), ("Qwen, team name", Q), ("Qwen, letter", LET), ("Qwen, reasoning on", T)) if R) + ".", ""]
        top = sorted(ids, key=lambda i: -Q[i]["p"])[:round(coverage_at(curve(Q, ids), .95) * len(ids))]
        L += [f"Qwen, team name: the {len(top)} most confident answers by team: "
              + ", ".join(f"{t} {k}" for t, k in collections.Counter(Q[i]["pred"] for i in top).most_common(3)) + ". Highest probability "
              + f"{max(Q[i]['p'] for i in ids):.3f}."]
        L += ["", "| Stated probability | Nimble 9B | Qwen, letter |", "|---|---|---|"]
        for lo, up in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01), (0.99, 1.01)):
            cells = []
            for R in (N, LET):
                b_ = [ok(R[i]) for i in ids if lo <= R[i]["p"] < up]
                cells.append(f"{pct(sum(b_) / len(b_), 0)} ({len(b_)})" if b_ else "none")
            L.append(f"| {f'{lo:.2f} or more' if lo == 0.99 else f'{lo:.1f} to {min(up, 1.0):.1f}'} | " + " | ".join(cells) + " |")
        L.append("")
    open(os.path.join(OUT, "results.md"), "w").write("\n".join(L))


# ---------------------------------------------------------------- charts
# canvas 1200 x 750, exported at 2x

W, H = 1200, 750
LEFT, RIGHT = 56, W - 56


def esc(s):
    return html.escape(str(s))


def text(x, y, s, size=21, fill=INK, anchor="start", weight=400, family="t", extra=""):
    fam = {"t": "'Schibsted Grotesk', sans-serif", "d": "'Fraunces', serif"}[family]
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" {extra}>{esc(s)}</text>'


def frame(title, subtitle, body, legend=(), foot=FOOT):
    parts = [f'<rect width="{W}" height="{H}" fill="{PAPER}"/>',
             f'<rect width="{W}" height="{H}" fill="url(#dots)"/>',
             text(LEFT, 80, title, 40, family="d", weight=600, extra='letter-spacing="-0.5"'),
             text(LEFT, 116, subtitle, 22, SOFT)]
    x = LEFT
    for key in legend:
        parts.append(f'<circle cx="{x + 8}" cy="153" r="8" fill="{COL[key]}"/>')
        parts.append(text(x + 25, 160, NAME[key], 20, INK, weight=600))
        x += 56 + 10.8 * len(NAME[key])
    parts.append(f'<line x1="{LEFT}" y1="{H - 56}" x2="{RIGHT}" y2="{H - 56}" stroke="{INK}" stroke-width="2"/>')
    parts.append(text(LEFT, H - 28, foot, 17, FAINT, weight=500, extra='letter-spacing="0.2"'))
    return ('<!doctype html><html><head><meta charset="utf-8"><style>'
            '@import url("https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Schibsted+Grotesk:wght@400;500;600;700&display=swap");'
            f'html,body{{margin:0;background:{PAPER}}}</style></head><body>'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
            f'<defs><pattern id="dots" width="30" height="30" patternUnits="userSpaceOnUse"><circle cx="1.5" cy="1.5" r="1.5" fill="{INK}" opacity="0.07"/></pattern></defs>'
            + "".join(parts) + body + "</svg></body></html>")


def hbar(x0, y, w, color, thick=22):
    """Horizontal bar, rounded at the data end."""
    w = max(w, 5)
    return f'<path d="M{x0:.1f},{y:.1f} h{w - 4:.1f} a4,4 0 0 1 4,4 v{thick - 8} a4,4 0 0 1 -4,4 h-{w - 4:.1f} z" fill="{color}"/>'


def vgrid(x, y_top, y_bottom, label):
    return (f'<line x1="{x:.1f}" y1="{y_top}" x2="{x:.1f}" y2="{y_bottom}" stroke="{GRID}" stroke-width="1"/>'
            + text(x, y_bottom + 28, label, 20, FAINT, "middle"))


def hgrid(y, x_left, x_right, label):
    return (f'<line x1="{x_left}" y1="{y:.1f}" x2="{x_right}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>'
            + text(x_left - 14, y + 7, label, 20, FAINT, "end"))


def chart_accuracy():
    x0, x1 = 380, 940
    sx = lambda p: x0 + (x1 - x0) * p
    panels = [(f"All {len(ALL)} complaints", ALL, MODELS)]
    if SUB and not FULL:
        panels.append((f"The {len(SUB)} complaints also run with reasoning on", SUB, "nqt"))
    rows_total = sum(len(o) for _, _, o in panels)
    bottom = 196 + 46 * len(panels) + 50 * rows_total + 26 * (len(panels) - 1)
    b = [text(sx(p), bottom + 28, pct(p, 0), 20, FAINT, "middle") for p in (0, .25, .5, .75, 1)]
    y = 196
    for head, ids, order in panels:
        b.append(text(LEFT, y + 16, head, 21, SOFT, weight=600))
        y += 46
        for p in (0, .25, .5, .75, 1):
            b.append(f'<line x1="{sx(p):.1f}" y1="{y - 10}" x2="{sx(p):.1f}" y2="{y + 50 * len(order) - 18}" stroke="{GRID}" stroke-width="1"/>')
        for m in order:
            k, p, ci = acc(RES[m], ids)
            b.append(text(x0 - 18, y + 18, NAME[m], 21, INK, "end", 500))
            b.append(hbar(x0, y, sx(p) - x0, COL[m]))
            lo, hi = sx(p - ci), sx(min(1, p + ci))
            b.append(f'<line x1="{lo:.1f}" y1="{y + 11}" x2="{hi:.1f}" y2="{y + 11}" stroke="{INK}" stroke-width="2"/>')
            for xx in (lo, hi):
                b.append(f'<line x1="{xx:.1f}" y1="{y + 4}" x2="{xx:.1f}" y2="{y + 18}" stroke="{INK}" stroke-width="2"/>')
            b.append(text(hi + 14, y + 19, pct(p), 23, INK, weight=700))
            b.append(text(hi + 92, y + 19, f"{k} of {len(ids)}", 19, FAINT))
            y += 50
        y += 26
    return frame("Routing accuracy", "Share of complaints sent to the team they were filed under. Whiskers: 95% margin.", "".join(b))


def chart_coverage():
    x0, x1, y0, y1 = 120, 880, 596, 236
    RC = dict(RES, q=LET) if LET else RES              # Qwen line: the run with letter answers
    curves = {m: curve(RC[m], ALL) for m in MODELS}
    shown = {m: [(c, a) for k, (c, a) in enumerate(curves[m], 1) if k >= len(ALL) // 10] for m in MODELS}
    amin = min(.70, math.floor(min(a for m in MODELS for _, a in shown[m]) * 20) / 20)
    sx = lambda c: x0 + (x1 - x0) * c
    sy = lambda a: y0 + (y1 - y0) * (inside(a, amin, 1.0, "coverage chart accuracy") - amin) / (1.0 - amin)
    b = [hgrid(sy(amin + k * .05), x0, x1, pct(amin + k * .05, 0)) for k in range(int(round((1 - amin) / .05)) + 1)]
    b += [text(sx(c), y0 + 28, pct(c, 0), 20, FAINT, "middle") for c in (0, .2, .4, .6, .8, 1)]
    b.append(text((x0 + x1) / 2, y0 + 62, "Share of complaints routed without a person", 20, SOFT, "middle", 500))
    b.append(text(LEFT, 206, "Accuracy of the routed share", 20, SOFT, weight=500))
    for m in MODELS:
        d = " ".join(f"{sx(c):.1f},{sy(a):.1f}" for c, a in shown[m])
        b.append(f'<polyline points="{d}" fill="none" stroke="{COL[m]}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
        c, a = shown[m][-1]
        b.append(f'<circle cx="{sx(c):.1f}" cy="{sy(a):.1f}" r="7" fill="{COL[m]}" stroke="{PAPER}" stroke-width="2"/>')
    ends = {m: acc(RC[m], ALL)[1] for m in MODELS}
    y_last = None
    for m in sorted(MODELS, key=lambda m: -ends[m]):       # keep end labels apart
        y = sy(ends[m]) + 7 if y_last is None else max(sy(ends[m]) + 7, y_last + 28)
        b.append(text(x1 + 16, y, f"{SHORT[m]}  {pct(ends[m])}", 20, INK, weight=600))
        y_last = y
    for target in (.95, .90):
        placed = []
        for m in sorted(MODELS, key=lambda m: coverage_at(curves[m], target)):
            cov = coverage_at(curves[m], target)
            if not cov:
                continue                                   # never reached
            b.append(f'<line x1="{sx(cov):.1f}" y1="{sy(target):.1f}" x2="{sx(cov):.1f}" y2="{y0}" stroke="{COL[m]}" stroke-width="2" opacity="0.5"/>')
            b.append(f'<circle cx="{sx(cov):.1f}" cy="{sy(target):.1f}" r="8" fill="{COL[m]}" stroke="{PAPER}" stroke-width="2"/>')
            below = bool(placed) and cov - placed[-1] < .07   # label to the right if markers are close
            b.append(text(sx(cov) + (12 if below else 0), sy(target) - 16, pct(cov, 0), 22, INK, "start" if below else "middle", 700))
            placed.append(cov)
    return frame("How much can be routed automatically",
                 "Most confident answers first: the share that can skip review at a required accuracy.",
                 "".join(b), legend=MODELS)


def chart_time():
    x0, x1 = 370, 1130
    all_s = [RES[m][i]["sec"] for m in "nqt" for i in SUB]
    # axis ends follow the data
    smin, smax = (1.0 if min(all_s) >= 1.0 else 0.1), max(400.0, 1.15 * max(all_s))
    lo, hi = math.log10(smin), math.log10(smax)
    sx = lambda s: x0 + (x1 - x0) * (math.log10(s) - lo) / (hi - lo)
    b = [vgrid(sx(s), 190, 596, lab) for s, lab in ((0.2, "0.2 s"), (0.5, "0.5 s"), (1, "1 s"), (2, "2 s"), (5, "5 s"), (10, "10 s"), (30, "30 s"), (60, "1 min"), (120, "2 min"), (300, "5 min"), (600, "10 min")) if smin < s <= smax and (s >= 2 or smin < 1)]
    b.append(text(x1, 660, "Time per complaint, logarithmic scale", 20, SOFT, "end", 500))
    many = len(SUB) > 200
    r_ok, r_no, alpha, ring = (4.5, 3.5, 0.55, 1) if many else (6.5, 5.5, 0.8, 2)
    for row, m in enumerate("nqt"):
        yc = 256 + row * 138
        s = secs(RES[m], SUB)
        b.append(text(LEFT, yc - 10, NAME[m].split(",")[0], 23, INK, weight=600))
        b.append(text(LEFT, yc + 18, (NAME[m].split(", ")[1] if "," in NAME[m] else "single pass"), 20, SOFT))
        note = f"median {statistics.median(s):.1f} s" if s[-1] < 10 else f"median {statistics.median(s):.0f} s, longest {s[-1] / 60:.1f} min"
        b.append(text(LEFT, yc + 44, note, 19, FAINT))
        for i in SUB:
            r = RES[m][i]
            inside(r["sec"], smin, smax, f"time chart, {NAME[m]}, complaint {i}")
            jit = (int(hashlib.md5(f"{m}{i}".encode()).hexdigest()[:4], 16) / 65535 - .5) * 84
            if ok(r):
                b.append(f'<circle cx="{sx(r["sec"]):.1f}" cy="{yc + jit:.1f}" r="{r_ok}" fill="{COL[m]}" fill-opacity="{alpha}" stroke="{PAPER}" stroke-width="{ring}"/>')
            else:
                b.append(f'<circle cx="{sx(r["sec"]):.1f}" cy="{yc + jit:.1f}" r="{r_no}" fill="{PAPER}" stroke="{COL[m]}" stroke-width="{ring + 1}"/>')
    for row, m in enumerate("nqt"):                      # label the cached repeat
        fast = [RES[m][i]["sec"] for i in SUB if RES[m][i]["sec"] < 1]
        if fast:
            b.append(text(sx(min(fast)), 256 + row * 138 + 44, "repeated complaint, from cache", 18, SOFT, "middle", 500))
    b.append(f'<circle cx="880" cy="153" r="6.5" fill="{SOFT}"/>' + text(896, 160, "right team", 20, SOFT))
    b.append(f'<circle cx="1022" cy="153" r="5.5" fill="{PAPER}" stroke="{SOFT}" stroke-width="3"/>' + text(1038, 160, "wrong team", 20, SOFT))
    return frame("Time per complaint", f"One dot per complaint, the same {len(SUB)} complaints for each model.", "".join(b))


def chart_tokens():
    pts = {m: [(RES[m][i]["in_tok"], RES[m][i]["sec"]) for i in ALL] for m in "nq"}
    # axis ends follow the data
    xmax = 200 * math.ceil(max(t for m in pts for t, _ in pts[m]) / 200)
    ymax = math.ceil(max(s for m in pts for _, s in pts[m]))
    x0, x1, y0, y1 = 120, 1130, 596, 236
    sx = lambda t: x0 + (x1 - x0) * t / xmax
    sy = lambda s: y0 + (y1 - y0) * s / ymax
    b = [hgrid(sy(s), x0, x1, f"{s} s") for s in range(0, ymax + 1)]
    b += [text(sx(t), y0 + 28, f"{t:,}", 20, FAINT, "middle") for t in range(0, xmax + 1, 200)]
    b.append(text((x0 + x1) / 2, y0 + 62, "Input tokens per request: complaint, question and team descriptions", 20, SOFT, "middle", 500))
    b.append(text(LEFT, 206, "Seconds per complaint", 20, SOFT, weight=500))
    for m in "qn":
        for t, s in pts[m]:
            inside(t, 0, xmax, f"token chart, {NAME[m]}, tokens")
            inside(s, 0, ymax, f"token chart, {NAME[m]}, seconds")
            b.append(f'<circle cx="{sx(t):.1f}" cy="{sy(s):.1f}" r="4.5" fill="{COL[m]}" fill-opacity="0.45"/>')
    (an, bn, _), (aq, bq, _) = fit(N, ALL), fit(Q, ALL)
    b.append(text(x0 + 24, sy(ymax * 0.93), f"Nimble reads about {1 / bn:.0f} tokens per second, Qwen about {1 / bq:.0f}.", 21, INK, weight=600))
    b.append(text(x0 + 24, sy(ymax * 0.93) + 30, f"Fixed part per request: Nimble {an:.2f} s, Qwen {aq:.2f} s (it writes its answer).", 20, SOFT))
    for m in "nq":
        for t, s in pts[m]:
            if s < 1:
                b.append(f'<circle cx="{sx(t):.1f}" cy="{sy(s):.1f}" r="8" fill="{COL[m]}" stroke="{PAPER}" stroke-width="2"/>')
                b.append(text(sx(t) + 16, sy(s) - 12, f"repeated request, answered from cache in {s:.2f} s", 20, INK, weight=600))
    return frame("Latency follows the length of the input", "Each dot is one complaint. Time grows with the number of tokens the model has to read.", "".join(b), legend="nq")


def chart_teams():
    x0, x1 = 290, 1110
    per = {t: [i for i in ALL if N[i]["gold"] == t] for t in TEAMS}
    share = {t: {m: sum(ok(RES[m][i]) for i in per[t]) / len(per[t]) for m in MODELS} for t in TEAMS}
    lo = math.floor(min(v for t in TEAMS for v in share[t].values()) * 10) / 10
    sx = lambda p: x0 + (x1 - x0) * (inside(p, lo, 1.0, "team chart") - lo) / (1.0 - lo)
    ticks = [round(lo + k / 10, 1) for k in range(int(round((1 - lo) * 10)) + 1)]
    b = [vgrid(sx(p), 190, 600, pct(p, 0)) for p in ticks]
    for k, t in enumerate(sorted(TEAMS, key=lambda t: -share[t]["n"])):
        y = 216 + k * 45
        b.append(text(x0 - 22, y + 7, t, 21, INK, "end", 500))
        b.append(f'<line x1="{sx(min(share[t].values())):.1f}" y1="{y}" x2="{sx(max(share[t].values())):.1f}" y2="{y}" stroke="{FAINT}" stroke-width="3"/>')
        for m in reversed(MODELS):
            b.append(f'<circle cx="{sx(share[t][m]):.1f}" cy="{y}" r="10" fill="{COL[m]}" stroke="{PAPER}" stroke-width="2"/>')
    return frame("Accuracy by team", f"{len(ALL) // len(TEAMS)} complaints per team.", "".join(b), legend=MODELS)


def chart_total():
    tot = {m: sum(secs(RES[m], SUB)) for m in "nqt"}
    hmax = 4 * math.ceil(tot["t"] / 3600 / 4)          # hours, multiple of 4
    x0, x1 = 330, 980
    sx = lambda s: x0 + (x1 - x0) * inside(s / 3600, 0, hmax, "total time chart") / hmax
    b = [vgrid(sx(h * 3600), 200, 470, f"{h} h") for h in range(0, hmax + 1, 4)]
    for row, m in enumerate("nqt"):
        y = 236 + row * 80
        b.append(text(x0 - 20, y + 18, SHORT[m], 21, INK, "end", 500))
        b.append(hbar(x0, y, sx(tot[m]) - x0, COL[m]))
        b.append(text(sx(tot[m]) + 14, y + 19, dur(tot[m]), 23, INK, weight=700))
    st = secs(T, SUB)
    b.append(text(LEFT, 560, f"Reasoning on took {tot['t'] / tot['n']:.0f} times as long as Nimble 9B and {tot['t'] / tot['q']:.0f} times as long as the same model with reasoning off.", 21, INK, weight=600))
    b.append(text(LEFT, 592, f"Per complaint with reasoning on: median {statistics.median(st):.0f} s, longest {st[-1] / 60:.1f} min. Single pass: about 2 s.", 20, SOFT))
    return frame(f"Time to process {len(SUB)} complaints", "Sum of the time per complaint. Same complaints, same machine, one request at a time.", "".join(b))


def chart_pattern():
    """Concept diagram, no data."""
    box = lambda x, y, w, h, col: f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{PAPER}" stroke="{col}" stroke-width="3"/>'
    arrow = lambda d: f'<path d="{d}" fill="none" stroke="{INK}" stroke-width="2.5" marker-end="url(#tip)"/>'
    b = ['<defs><marker id="tip" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">'
         f'<path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker></defs>']

    def card(x, y, w, col, head, *lines):
        b.append(box(x, y, w, 58 + 28 * len(lines), col))
        b.append(text(x + 18, y + 36, head, 22, INK, weight=700))
        for j, ln in enumerate(lines):
            b.append(text(x + 18, y + 66 + 28 * j, ln, 20, SOFT))

    card(250, 196, 612, COL["n"], "Act automatically", "route, tag, approve: nobody waits")
    card(56, 400, 150, INK, "Incoming", "ticket,", "request")
    card(262, 400, 230, COL["n"], "Decision model", "fast and cheap", "answer + probability")
    card(632, 400, 230, INK, "Larger model", "slower, costs more", "can explain itself")
    card(1000, 400, 144, INK, "Person", "final call")
    b.append(arrow("M206,457 H256"))
    b.append(arrow("M492,457 H626"))
    b.append(arrow("M862,457 H994"))
    b.append(arrow("M377,400 V288"))
    b.append(arrow("M747,400 V288"))
    b.append(text(559, 442, "not confident", 18, INK, "middle", 600))
    b.append(text(928, 442, "fails the check", 18, INK, "middle", 600))
    b.append(text(391, 350, "confident", 18, INK, weight=600))
    b.append(text(761, 350, "passes a check", 18, INK, weight=600))
    b.append(text(LEFT, 600, "Each step handles what it is sure about and passes the rest on.", 21, INK, weight=600))
    b.append(text(LEFT, 632, "The cut-offs are set on a labelled sample of your own data, at the accuracy the automatic path has to reach.", 20, SOFT))
    b.append(text(LEFT, 660, "Decisions made by people become new labels for checking those cut-offs.", 20, SOFT))
    return frame("The decision model as a first gate", "An escalation ladder: the cheapest step sees everything, the most expensive sees the least.", "".join(b), foot="A possible design · escalation by confidence")


def chart_cache():
    label = {"A": "New complaint, one question", "D": "Short input, three teams", "B": "Identical request again",
             "E": "One sentence added at the end", "C": "Same complaint, different question", "G": "Three questions in one request"}
    rows = {r[0]: r for r in cache_rows()}
    smax = math.ceil(max(rows[k][2] for k in label))
    x0, x1 = 430, 990
    sx = lambda s: x0 + (x1 - x0) * inside(s, 0, smax, "cache chart") / smax
    b = [vgrid(sx(s), 190, 560, f"{s} s") for s in range(0, smax + 1)]
    for k, key in enumerate("ADBECG"):
        y = 208 + k * 60
        b.append(text(x0 - 20, y + 18, label[key], 21, INK, "end", 500))
        b.append(hbar(x0, y, sx(rows[key][2]) - x0, COL["n"]))
        b.append(text(sx(rows[key][2]) + 14, y + 19, f"{rows[key][2]:.2f} s", 23, INK, weight=700))
    return frame("When the cache helps", "Nimble 9B, median time per case. Only an identical request is answered from cache.", "".join(b))


def charts(png=True):
    os.makedirs(CHARTS, exist_ok=True)
    todo = [("pattern", chart_pattern), ("accuracy", chart_accuracy), ("coverage", chart_coverage), ("tokens", chart_tokens), ("teams", chart_teams)]
    if SUB:
        todo += [("time", chart_time), ("total", chart_total)]
    if os.path.exists(os.path.join(RAW, "cache-test.txt")):
        todo.append(("cache", chart_cache))
    for name, fn in todo:
        path = os.path.join(CHARTS, name + ".html")
        open(path, "w").write(fn())
        if png:
            subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},{H}",
                            "--force-device-scale-factor=2", "--virtual-time-budget=8000",
                            f"--screenshot={os.path.join(CHARTS, name + '.png')}", "file://" + os.path.abspath(path)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("chart", name)


if __name__ == "__main__":
    results_md()
    print("wrote metrics/results.md")
    charts(png="--no-png" not in sys.argv)
