import argparse
import json
from collections import Counter

import  tensorflow.io.gfile as gfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--format", choices=["vqa2", "doc_qa"], default="vqa2")
    args = parser.parse_args()

    with gfile.GFile(args.source, "r") as f:
        data = json.load(f)

    converted = []
    for ex in data:
        pred = ex["prediction"].strip()
        if "\n" in pred:
            preds = [" ".join(x.strip().split()) for x in pred.split("\n")]
            counts = Counter(preds)
            max_count = max(counts.values())
            pred = [x for x in preds if counts[x] == max_count][0]
        if args.format == "vqa2":
            converted.append(dict(answer=pred, question_id=ex["example_id"]))
        else:
            converted.append(dict(answer=pred, questionId=ex["example_id"]))

    print(len(converted))
    with gfile.GFile(args.output, "w") as f:
        json.dump(converted, f)


if __name__ == '__main__':
    main()