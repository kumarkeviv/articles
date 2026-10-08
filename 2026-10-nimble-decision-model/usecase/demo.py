"""One complaint, asked three ways.

python3 usecase/demo.py [complaint_id]
"""
import json, sys, textwrap, time, urllib.request
import cfpb

DIM, BOLD, OFF, CLR = "\033[2m", "\033[1m", "\033[0m", "\033[K"
WIDTH = 92


def stream(path, body):
    req = urllib.request.Request(cfpb.HOST + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if line.strip():
                yield json.loads(line)


def row(name, answer, detail, sec):
    print(f"\r{CLR}{name:<28}{BOLD}{answer:<18}{OFF}{detail:<22}{sec:>6.1f} s")


def main():
    data = {c["id"]: c for c in json.load(open("cfpb_eval.json"))}
    c = data[int(sys.argv[1])] if len(sys.argv) > 1 else data[5983778]
    print(f"{DIM}CFPB complaint {c['id']}, filed under: {c['product']}{OFF}")
    for line in textwrap.wrap(c["text"].replace("\n", " "), WIDTH)[:5]:
        print(line)
    print(f"\n{DIM}{cfpb.QUESTION}  ({len(cfpb.TEAMS)} teams){OFF}\n")

    other = next(x for x in data.values() if x["id"] != c["id"])  # warm up with a different text

    # 1. decision model
    print(f"{DIM}loading nimble{OFF}", end="", flush=True)
    cfpb.ask_nimble("nimble", other)
    print(f"\r{CLR}Nimble 9B", end="", flush=True)
    r, dt = cfpb.ask_nimble("nimble", c)
    row("Nimble 9B", r["pred"], f"p {r['p']:.2f}", dt)

    # 2. base model, reasoning off, JSON schema
    print(f"{DIM}loading {cfpb.QWEN}{OFF}", end="", flush=True)
    cfpb.ask_chat(cfpb.QWEN, other)
    print(f"\r{CLR}Qwen3.5-9B, reasoning off", end="", flush=True)
    r, dt = cfpb.ask_chat(cfpb.QWEN, c)
    row("Qwen3.5-9B, reasoning off", r["pred"], f"p {r['p']:.2f}", dt)

    # 3. base model, reasoning on, streamed
    prompt = cfpb.QUESTION + "\n" + "\n".join(f"- {k}: {v}" for k, v in cfpb.TEAMS.items()) + f"\n\nComplaint: {c['text']}"
    body = {"model": cfpb.QWEN, "keep_alive": "30m", "stream": True, "think": True, "options": {"temperature": 0},
            "format": {"type": "object", "properties": {"team": {"type": "string", "enum": list(cfpb.TEAMS)}}, "required": ["team"]},
            "messages": [{"role": "user", "content": prompt}]}
    t0, thinking, content, tokens = time.perf_counter(), "", "", 0
    print()
    for chunk in stream("/api/chat", body):
        m = chunk.get("message", {})
        thinking += m.get("thinking") or ""
        content += m.get("content") or ""
        tokens = chunk.get("eval_count", tokens)
        tail = " ".join(thinking.split())[-(WIDTH - 4):]
        print(f"\033[1A\r{CLR}{'Qwen3.5-9B, reasoning on':<28}{'thinking':<18}{'':<22}{time.perf_counter() - t0:>6.1f} s\n\r{CLR}{DIM}  {tail}{OFF}", end="", flush=True)
    dt = time.perf_counter() - t0
    print(f"\033[1A", end="")
    row("Qwen3.5-9B, reasoning on", json.loads(content)["team"], f"{tokens} tokens written", dt)
    print(f"\r{CLR}")


if __name__ == "__main__":
    main()
