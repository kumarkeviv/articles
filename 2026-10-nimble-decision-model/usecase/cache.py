"""Cache test: 8 complaints per case, 20 for the short input. Writes metrics/raw/cache-test.txt.

python3 usecase/cache.py
"""
import json, statistics, time, urllib.request
import cfpb

WARM = {"text": "Warm-up request to load the model. Not part of the test."}
OUT = []


def post(path, body):
    req = urllib.request.Request(cfpb.HOST + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    return r, time.perf_counter() - t


def show(key, label, xs):
    line = f"{key} {label:56s} median {statistics.median(x[0] for x in xs):6.3f} s   tokens {statistics.median(x[1] for x in xs):.0f}"
    print(line, flush=True)
    OUT.append(line)


def nimble(state, questions):
    r, dt = post("/v1/systemone", {"model": cfpb.NIMBLE, "keep_alive": "10m", "state": state, "questions": questions})
    return dt, r["usage"]["input_tokens"]


def qwen(prompt):
    r, dt = post("/api/chat", {
        "model": cfpb.QWEN, "keep_alive": "10m", "stream": False, "think": False, "options": {"temperature": 0},
        "format": {"type": "object", "properties": {"team": {"type": "string", "enum": list(cfpb.TEAMS)}}, "required": ["team"]},
        "messages": [{"role": "user", "content": prompt}]})
    return dt, r.get("prompt_eval_count", 0)


data = json.load(open("cfpb_eval.json"))
first, second = data[100:108], data[120:128]
TEAM = {"team": {"type": "choice", "instructions": cfpb.QUESTION, "criteria": cfpb.TEAMS}}
FRAUD = {"fraud": {"type": "noul", "instructions": "Does the consumer say they are a victim of fraud or identity theft?"}}
SHORT = {"team": {"type": "choice", "instructions": cfpb.QUESTION,
                  "criteria": {k: cfpb.TEAMS[k] for k in ("Mortgage", "Credit card", "Bank account")}}}
LOSS = {"loss": {"type": "noul", "instructions": "Does the consumer describe a financial loss?"}}

OUT.append(f"# usecase/cache.py, {time.strftime('%Y-%m-%d')}. 8 real complaints per case, 20 for case D.")
OUT.append(f"NIMBLE ({cfpb.NIMBLE})")
cfpb.ask_nimble(cfpb.NIMBLE, WARM)
A, B, C, D, E, F, G = [], [], [], [], [], [], []
for c in first:
    s = {"complaint": c["text"]}
    A.append(nimble(s, TEAM))                                              # new complaint
    B.append(nimble(s, TEAM))                                              # identical request again
    C.append(nimble(s, FRAUD))                                             # same complaint, new question
    E.append(nimble({"complaint": c["text"] + " Please respond."}, TEAM))  # same start, changed end
    F.append(nimble({"complaint": "Note: " + c["text"]}, TEAM))            # changed start
for c in second:
    G.append(nimble({"complaint": c["text"]}, {**TEAM, **FRAUD, **LOSS}))  # three questions, one request
for c in data[140:160]:
    D.append(nimble({"complaint": c["text"][:100]}, SHORT))                # short input, new text each time
show("A", "new complaint, one question", A)
show("B", "identical request again", B)
show("C", "same complaint, a different question", C)
show("D", "first 100 characters of a new complaint, three teams", D)
show("E", "same complaint with a sentence added at the end", E)
show("F", "same complaint with a word added at the start", F)
show("G", "new complaint, three questions in one request", G)
post("/api/generate", {"model": cfpb.NIMBLE, "keep_alive": 0})

OUT.append(f"QWEN, reasoning off ({cfpb.QWEN})")
teams = cfpb.QUESTION + "\n" + "\n".join(f"- {k}: {v}" for k, v in cfpb.TEAMS.items())
cfpb.ask_chat(cfpb.QWEN, WARM)
H = [qwen(teams + f"\n\nComplaint: {c['text']}") for c in first]            # instructions first, new complaint each time
I = [qwen(teams + f"\n\nComplaint: {first[-1]['text']}") for _ in range(4)]  # the last request again
J = [qwen(f"Complaint: {c['text']}\n\n" + teams) for c in second]           # complaint first, new complaint each time
show("H", "instructions first, then a new complaint", H)
show("I", "identical request again", I)
show("J", "complaint first, then instructions", J)
post("/api/generate", {"model": cfpb.QWEN, "keep_alive": 0})

open("cache-test.txt", "w").write("\n".join(OUT) + "\n")
