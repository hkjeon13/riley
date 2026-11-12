#!/usr/bin/env python3
"""GSM8K accuracy of an OpenAI-compatible server.

usage: gsm8k.py ADDRESS MODEL_ID DATA_DIR OUT.json [--limit N]
DATA_DIR holds openai/gsm8k `main` test.jsonl and train.jsonl (the parquet
files converted to one JSON object per line). The four first
train examples form a 4-shot prompt; greedy completions of up to 256 tokens
stop at "Question:"; the last number of the reply is compared with the answer
after "####". Requests go out 32 at a time.
"""
import argparse
import concurrent.futures
import json
import re
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("address")
parser.add_argument("model")
parser.add_argument("data")
parser.add_argument("out")
parser.add_argument("--limit", type=int, default=0)
args = parser.parse_args()


def rows(split):
    return [json.loads(line) for line in open(f"{args.data}/{split}.jsonl")]


def number(text):
    found = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return found[-1].replace(",", "").rstrip(".") if found else None


shots = "".join(f"Question: {r['question']}\nAnswer: {r['answer']}\n\n" for r in rows("train")[:4])
test = rows("test")
if args.limit:
    test = test[:args.limit]


def ask(row):
    body = json.dumps({"model": args.model, "prompt": shots + f"Question: {row['question']}\nAnswer:",
                       "max_tokens": 256, "temperature": 0, "stop": ["Question:"]}).encode()
    request = urllib.request.Request(f"http://{args.address}/v1/completions", body,
                                     {"Content-Type": "application/json"})
    text = json.load(urllib.request.urlopen(request, timeout=600))["choices"][0]["text"]
    gold = row["answer"].split("####")[-1].strip().replace(",", "")
    predicted = number(text.split("####")[-1]) if "####" in text else number(text)
    try:
        correct = predicted is not None and float(predicted) == float(gold)
    except ValueError:
        correct = False
    return {"correct": correct, "text": text}


with concurrent.futures.ThreadPoolExecutor(32) as pool:
    results = list(pool.map(ask, test))
accuracy = sum(r["correct"] for r in results) / len(results)
json.dump({"accuracy": accuracy, "questions": len(results), "results": results}, open(args.out, "w"))
print(json.dumps({"accuracy": round(100 * accuracy, 2), "questions": len(results)}))
