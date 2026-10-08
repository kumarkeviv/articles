"""Route CFPB complaints to one of nine product teams.

python3 usecase/cfpb.py sample 70               -> metrics/raw/cfpb_eval.json
python3 usecase/cfpb.py extend 100              -> top every team up to 100
python3 usecase/cfpb.py nimble nimble           -> metrics/raw/res-nimble.jsonl
python3 usecase/cfpb.py chat qwen3.5:9b-q8_0    -> metrics/raw/res-qwen3.5_9b-q8_0.jsonl
"""
import collections, json, math, os, random, sys, time, urllib.request

# data files live in metrics/raw
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "metrics", "raw"))

HOST = "http://localhost:11434"
DS = "https://datasets-server.huggingface.co"
REPO = "BEE-spoke-data/consumer-finance-complaints"

TEAMS = {
    "Credit reporting": "Credit reports, credit scores, credit bureaus, inquiries, credit repair",
    "Debt collection": "A collector or collection agency pursuing a debt",
    "Mortgage": "Home loans, mortgage servicing, escrow, foreclosure",
    "Credit card": "Credit cards and prepaid cards: charges, fees, disputes, rewards",
    "Bank account": "Checking or savings accounts, deposits, withdrawals, overdrafts",
    "Student loan": "Federal or private student loans and their servicing",
    "Vehicle loan": "Car or other vehicle loans and leases",
    "Personal loan": "Payday, title, instalment or personal loans",
    "Money transfer": "Money transfers, payment apps, virtual currency",
}
# CFPB product name -> team
MAP = {
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Debt collection": "Debt collection",
    "Mortgage": "Mortgage",
    "Credit card or prepaid card": "Credit card",
    "Credit card": "Credit card",
    "Checking or savings account": "Bank account",
    "Student loan": "Student loan",
    "Vehicle loan or lease": "Vehicle loan",
    "Payday loan, title loan, or personal loan": "Personal loan",
    "Payday loan, title loan, personal loan, or advance loan": "Personal loan",
    "Money transfer, virtual currency, or money service": "Money transfer",
}
QUESTION = "Which product team should handle this complaint?"
NIMBLE = "nimble"            # 8-bit
QWEN = "qwen3.5:9b-q8_0"     # base model of Nimble, also 8-bit


def get(url, tries=4):
    for _ in range(tries):
        try:
            return json.load(urllib.request.urlopen(url, timeout=60))
        except Exception:
            time.sleep(3)
    return {}


def post(path, body):
    req = urllib.request.Request(HOST + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter()
    out = json.load(urllib.request.urlopen(req, timeout=600))
    return out, time.perf_counter() - t


def build_sample(per_class):
    n = get(f"{DS}/size?dataset={REPO}&config=has-text")["size"]["config"]["num_rows"]
    rnd = random.Random(2026)
    offsets = rnd.sample(range(0, n - 100, 100), 400)
    pool = collections.defaultdict(dict)
    for k, off in enumerate(offsets):
        rows = get(f"{DS}/rows?dataset={REPO}&config=has-text&split=train&offset={off}&length=100").get("rows", [])
        for r in rows:
            r = r["row"]
            text, team = r["Consumer complaint narrative"] or "", MAP.get(r["Product"])
            if team and r["Date received"] >= "2018-01-01" and 200 <= len(text) <= 2000:
                pool[team][r["Complaint ID"]] = {"id": r["Complaint ID"], "date": r["Date received"], "team": team,
                                                 "product": r["Product"], "issue": r["Issue"], "text": text}
        if k % 20 == 19:
            print(k + 1, {t: len(v) for t, v in pool.items()}, flush=True)
        if len(pool) == len(TEAMS) and min(len(v) for v in pool.values()) >= per_class:
            break
    out = []
    for team in TEAMS:
        items = sorted(pool[team].values(), key=lambda x: x["id"])
        out += rnd.sample(items, min(per_class, len(items)))
    rnd.shuffle(out)
    json.dump(out, open("cfpb_eval.json", "w"), indent=1)
    print(len(out), collections.Counter(x["team"] for x in out))


def extend_sample(per_class):
    """Top every team up to per_class. Existing complaints stay, new ones are appended."""
    data = json.load(open("cfpb_eval.json"))
    have_ids, have_text = {c["id"] for c in data}, {c["text"] for c in data}
    need = {t: per_class - sum(c["team"] == t for c in data) for t in TEAMS}
    n = get(f"{DS}/size?dataset={REPO}&config=has-text")["size"]["config"]["num_rows"]
    rnd = random.Random(2027)
    pool = collections.defaultdict(dict)
    for k, off in enumerate(rnd.sample(range(0, n - 100, 100), 400)):
        rows = get(f"{DS}/rows?dataset={REPO}&config=has-text&split=train&offset={off}&length=100").get("rows", [])
        for r in rows:
            r = r["row"]
            text, team = r["Consumer complaint narrative"] or "", MAP.get(r["Product"])
            if (team and r["Date received"] >= "2018-01-01" and 200 <= len(text) <= 2000
                    and r["Complaint ID"] not in have_ids and text not in have_text):
                have_text.add(text)  # skip duplicate texts
                pool[team][r["Complaint ID"]] = {"id": r["Complaint ID"], "date": r["Date received"], "team": team,
                                                 "product": r["Product"], "issue": r["Issue"], "text": text}
        if k % 20 == 19:
            print(k + 1, {t: len(v) for t, v in pool.items()}, flush=True)
        if all(len(pool[t]) >= need[t] for t in TEAMS):
            break
    extra = []
    for team in TEAMS:
        extra += rnd.sample(sorted(pool[team].values(), key=lambda x: x["id"]), max(0, need[team]))
    rnd.shuffle(extra)
    json.dump(data + extra, open("cfpb_eval.json", "w"), indent=1)
    print(len(data + extra), collections.Counter(x["team"] for x in data + extra))


def ask_nimble(model, c):
    raw, dt = post("/v1/systemone", {
        "model": model, "keep_alive": "30m",
        "state": {"complaint": c["text"]},
        "questions": {"team": {"type": "choice", "instructions": QUESTION, "criteria": TEAMS}},
    })
    a = raw["answers"]["team"]
    probs = {k: float(v) for k, v in a["probabilities"].items()}
    return {"pred": a["choice"], "p": probs[a["choice"]], "probs": probs, "confidence": a.get("confidence"),
            "in_tok": raw.get("usage", {}).get("input_tokens")}, dt


def ask_chat(model, c):
    prompt = QUESTION + "\n" + "\n".join(f"- {k}: {v}" for k, v in TEAMS.items()) + f"\n\nComplaint: {c['text']}"
    raw, dt = post("/api/chat", {
        "model": model, "keep_alive": "30m", "stream": False, "think": False,
        "options": {"temperature": 0}, "logprobs": True,
        "format": {"type": "object", "properties": {"team": {"type": "string", "enum": list(TEAMS)}}, "required": ["team"]},
        "messages": [{"role": "user", "content": prompt}],
    })
    content = raw["message"]["content"]
    try:
        pred = json.loads(content)["team"]
    except Exception:
        pred = None
    # probability of the answer = product of its token probabilities
    p = None
    if pred and raw.get("logprobs"):
        start = content.find(pred, content.find('"team"') + 6)
        pos, lp = 0, 0.0
        for t in raw["logprobs"]:
            a, b = pos, pos + len(t["token"])
            pos = b
            if b > start and a < start + len(pred):
                lp += t["logprob"]
        p = math.exp(lp)
    return {"pred": pred, "p": p, "raw": content if pred is None else None,
            "in_tok": raw.get("prompt_eval_count"), "out_tok": raw.get("eval_count")}, dt


def run(mode, model):
    data = json.load(open("cfpb_eval.json"))
    path = f"res-{model.replace(':', '_')}.jsonl"
    done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
    ask = ask_nimble if mode == "nimble" else ask_chat
    # warm-up, not recorded. Not a sample complaint: a repeated request is answered from cache.
    ask(model, {"text": "Warm-up request to load the model. Not part of the test."})
    with open(path, "a") as f:
        for i, c in enumerate(data):
            if c["id"] in done:
                continue
            try:
                r, dt = ask(model, c)
            except Exception as e:
                r, dt = {"pred": None, "p": None, "error": str(e)[:200]}, None
            r.update(id=c["id"], gold=c["team"], sec=dt)
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 50 == 0:
                print(i, flush=True)
    print("done", path)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sample":
        build_sample(int(sys.argv[2]))
    elif cmd == "extend":
        extend_sample(int(sys.argv[2]))
    else:
        run(cmd, sys.argv[2])
