import argparse
import json

import tiktoken

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def main(dataset_path: str, fields: set[str]) -> None:
    row_count = 0
    token_count = 0

    with open(dataset_path) as f:
        for line in f:
            obj = json.loads(line)
            filtered_obj = {k: v for k, v in obj.items() if k in fields}

            row_str = json.dumps(filtered_obj)
            row_with_sep = row_str if row_count == 0 else "\n" + row_str
            token_count += _count_tokens(row_with_sep)
            row_count += 1

    print(f"Row count: {row_count}")
    print(f"Token count: {token_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--fields", nargs="+", required=True)
    args = parser.parse_args()

    main(args.dataset_path, set(args.fields))
