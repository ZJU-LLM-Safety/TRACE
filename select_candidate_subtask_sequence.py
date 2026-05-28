import argparse
import json
import os

from env_utils import ensure_parent_dir


def select_fewest_harmful_sub_prompts(items, threshold):
    return min(items, key=lambda x: sum(1 for score in x["harmful_scores"] if score > threshold))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the least harmful candidate subtask sequence per id.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file containing candidate subtask sequences.")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file for selected sequences.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold above which a sub-prompt is treated as harmful.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    existing_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                existing_ids.add(item["id"])

    item_dict = {}

    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["id"] in existing_ids:
                print(f"skipping {item['id']} because it already exists")
                continue
            item_id = item["id"]
            if item_id not in item_dict:
                item_dict[item_id] = []
            item_dict[item_id].append(item)

    ensure_parent_dir(args.output_file)
    with open(args.output_file, "a", encoding="utf-8") as f:
        for item_id, items in item_dict.items():
            selected_item = select_fewest_harmful_sub_prompts(items, args.threshold)
            f.write(json.dumps(selected_item) + "\n")
