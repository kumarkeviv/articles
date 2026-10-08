"""Qwen3.5-9B with reasoning on, whole sample. About a minute per complaint.

python3 usecase/think.py    # can be stopped and restarted
"""
import collections, json, math, os, time, urllib.request
import cfpb

PATH = f"res-{cfpb.QWEN.replace(':', '_')}-think.jsonl"


def ask(c):
    prompt = cfpb.QUESTION + "\n" + "\n".join(f"- {k}: {v}" for k, v in cfpb.TEAMS.items()) + f"\n\nComplaint: {c['text']}"
    body = {"model": cfpb.QWEN, "keep_alive": "30m", "stream": False, "think": True, "logprobs": True, "options": {"temperature": 0},
            "format": {"type": "object", "properties": {"team": {"type": "string", "enum": list(cfpb.TEAMS)}}, "required": ["team"]},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(cfpb.HOST + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    dt = time.perf_counter() - t
    pred = json.loads(r["message"]["content"]).get("team")
    # the answer tokens come last, after the reasoning
    p, tail = None, (r.get("logprobs") or [])[-60:]
    joined = "".join(x["token"] for x in tail)
    start = joined.rfind(pred) if pred else -1
    if start >= 0:
        pos, lp = 0, 0.0
        for x in tail:
            a, b = pos, pos + len(x["token"])
            pos = b
            if b > start and a < start + len(pred):
                lp += x["logprob"]
        p = math.exp(lp)
    return {"pred": pred, "p": p, "out_tok": r.get("eval_count"), "in_tok": r.get("prompt_eval_count"),
            "done": r.get("done_reason"), "thinking": r["message"].get("thinking")}, dt


def main():
    data = json.load(open("cfpb_eval.json"))
    seen, first, rest = collections.Counter(), [], []
    for c in data:
        if seen[c["team"]] < 10:
            seen[c["team"]] += 1
            first.append(c)
        else:
            rest.append(c)
    done = {json.loads(l)["id"] for l in open(PATH)} if os.path.exists(PATH) else set()
    ask({"text": "Warm-up request to load the model. Not part of the test."})
    with open(PATH, "a") as f:
        for n, c in enumerate(first + rest, 1):
            if c["id"] in done:
                continue
            try:
                rec, dt = ask(c)
            except Exception as e:  # recorded, not retried
                rec, dt = {"pred": None, "p": None, "error": str(e)[:200]}, None
            rec.update(id=c["id"], gold=c["team"], sec=dt, at=time.strftime("%Y-%m-%d %H:%M:%S"))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(time.strftime("%H:%M"), n, "of", len(data), c["team"], "->", rec["pred"], f"{dt:.0f} s" if dt else "ERROR", flush=True)
    print("done")


if __name__ == "__main__":
    main()
