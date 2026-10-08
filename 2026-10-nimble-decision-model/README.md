# Nimble 9B and its base model Qwen3.5-9B

Code and results for the article "Evaluating a decision model: Nimble 9B and its base model Qwen3.5-9B".

900 CFPB consumer complaints, each routed to one of nine teams. Run on a MacBook Pro 14-inch (M1 Pro, 32 GB) with Ollama 0.35.1.

## Run

Needs Ollama 0.35 or later and Python 3. Charts also need Google Chrome.

```bash
ollama pull nimble && ollama pull qwen3.5:9b-q8_0

python3 usecase/cfpb.py nimble nimble            # about 35 minutes
python3 usecase/cfpb.py chat qwen3.5:9b-q8_0     # about 33 minutes
python3 usecase/think.py                         # reasoning on, about 21 hours
python3 usecase/letters.py                       # about 33 minutes
python3 usecase/cache.py                         # about 3 minutes
python3 usecase/report.py                        # tables and charts
```

The runs write to `metrics/raw`. The three long ones can be stopped and restarted. Delete a result file to run it from the start.

## Files

- `usecase/` the scripts
- `metrics/raw/cfpb_eval.json` the 900 complaints
- `metrics/raw/res-*.jsonl` one line per complaint and model
- `metrics/results.md` all tables
- `metrics/charts/` the charts

Data: CFPB Consumer Complaint Database, public domain.
