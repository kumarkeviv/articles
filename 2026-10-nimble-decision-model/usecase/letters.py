"""Qwen3.5-9B, reasoning off, answering with one letter per team. Gives a probability for all nine teams.

python3 usecase/letters.py    # can be stopped and restarted
"""
import json, math, os, time
import cfpb

PATH = f"res-{cfpb.QWEN.replace(':', '_')}-letters.jsonl"
LETTERS = dict(zip("ABCDEFGHI", cfpb.TEAMS))


def ask(c):
    prompt = (cfpb.QUESTION + "\n" + "\n".join(f"{k}. {t}: {cfpb.TEAMS[t]}" for k, t in LETTERS.items())
              + f"\n\nComplaint: {c['text']}\n\nAnswer with the letter of the team.")
    raw, dt = cfpb.post("/api/chat", {
        "model": cfpb.QWEN, "keep_alive": "30m", "stream": False, "think": False,
        "options": {"temperature": 0}, "logprobs": True, "top_logprobs": 20,
        "format": {"type": "object", "properties": {"team": {"type": "string", "enum": list(LETTERS)}}, "required": ["team"]},
        "messages": [{"role": "user", "content": prompt}],
    })
    letter = json.loads(raw["message"]["content"])["team"]
    # the token that carries the letter, and the other letters the model considered there
    tok = next(t for t in raw["logprobs"] if t["token"].strip(' "') == letter)
    mass = dict.fromkeys(LETTERS, 0.0)
    for alt in tok["top_logprobs"]:
        k = alt["token"].strip(' "')
        if k in mass:
            mass[k] += math.exp(alt["logprob"])
    total = sum(mass.values())
    probs = {LETTERS[k]: v / total for k, v in mass.items()}
    return {"pred": LETTERS[letter], "p": probs[LETTERS[letter]], "probs": probs, "found": total,
            "in_tok": raw.get("prompt_eval_count"), "out_tok": raw.get("eval_count")}, dt


def main():
    data = json.load(open("cfpb_eval.json"))
    done = {json.loads(l)["id"] for l in open(PATH)} if os.path.exists(PATH) else set()
    ask({"text": "Warm-up request to load the model. Not part of the test."})
    with open(PATH, "a") as f:
        for n, c in enumerate(data, 1):
            if c["id"] in done:
                continue
            try:
                r, dt = ask(c)
            except Exception as e:
                r, dt = {"pred": None, "p": None, "error": str(e)[:200]}, None
            r.update(id=c["id"], gold=c["team"], sec=dt)
            f.write(json.dumps(r) + "\n")
            f.flush()
            if n % 50 == 0:
                print(n, flush=True)
    print("done", PATH)


if __name__ == "__main__":
    main()
