"""Reconstruct customer-service dialogs from the Kaggle Customer Support on Twitter
dataset, following the method described in the TweetSumm paper.

Reference: Feigenblat, G., Gunasekara, C., Sznajder, B., Joshi, S., Konopnicki, D.,
and Aharonov, R. (2021). "TweetSumm - A Dialog Summarization Dataset for Customer
Service." Section 2. https://aclanthology.org/2021.findings-emnlp.24
"""

import argparse
import csv
import json
from datetime import datetime
from typing import TypedDict

TWITTER_TS_FORMAT = "%a %b %d %H:%M:%S %z %Y"

MIN_UTTERANCES = 6
MAX_UTTERANCES = 20

Tweet = dict[str, str]


class DialogRecord(TypedDict):
    dialog_id: str
    company_id: str
    dialog: str


def _parse_created_at(value: str) -> datetime:
    return datetime.strptime(value, TWITTER_TS_FORMAT)


def _parse_response_ids(value: str) -> list[str]:
    if not value:
        return []
    return [rid.strip() for rid in value.split(",") if rid.strip()]


def _load_tweets(csv_path: str) -> dict[str, Tweet]:
    tweets: dict[str, Tweet] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tweets[row["tweet_id"]] = row
    return tweets


def _find_roots(tweets: dict[str, Tweet]) -> list[Tweet]:
    roots: list[Tweet] = []
    for tweet in tweets.values():
        parent_id = tweet["in_response_to_tweet_id"]
        if parent_id not in tweets:
            roots.append(tweet)
    return roots


def _sort_key(tweet: Tweet) -> datetime:
    return _parse_created_at(tweet["created_at"])


def _build_dialog(root: Tweet, tweets: dict[str, Tweet]) -> list[Tweet]:
    dialog: list[Tweet] = []
    visited: set[str] = set()
    stack = [root]

    while stack:
        tweet = stack.pop()
        tweet_id = tweet["tweet_id"]
        if tweet_id in visited:
            continue
        visited.add(tweet_id)
        dialog.append(tweet)

        children = [
            tweets[rid]
            for rid in _parse_response_ids(tweet["response_tweet_id"])
            if rid in tweets and rid not in visited
        ]
        # Sort siblings by timestamp; reverse so the stack pops them in order.
        children.sort(key=_sort_key, reverse=True)
        stack.extend(children)

    return dialog


def _reconstruct_dialogs(csv_path: str) -> list[list[Tweet]]:
    tweets = _load_tweets(csv_path)
    roots = _find_roots(tweets)
    roots.sort(key=_sort_key)
    return [_build_dialog(root, tweets) for root in roots]


def _format_time(value: str) -> str:
    return _parse_created_at(value).strftime("%Y-%m-%d %H:%M:%S")


def _company_id(dialog: list[Tweet]) -> str:
    # Assumes a dialog that passed _is_valid_dialog, i.e. exactly one agent.
    agents = {t["author_id"] for t in dialog if t["inbound"] == "False"}
    (company_id,) = agents
    return company_id


def _is_valid_dialog(dialog: list[Tweet]) -> bool:
    if not MIN_UTTERANCES <= len(dialog) <= MAX_UTTERANCES:
        return False
    customers = {t["author_id"] for t in dialog if t["inbound"] == "True"}
    agents = {t["author_id"] for t in dialog if t["inbound"] == "False"}
    return len(customers) == 1 and len(agents) == 1


def _dialog_to_record(dialog: list[Tweet]) -> DialogRecord:
    lines = [
        f"{t['author_id']} [{_format_time(t['created_at'])}]: {t['text']}"
        for t in dialog
    ]
    root = dialog[0]
    return {
        "dialog_id": root["tweet_id"],
        "company_id": _company_id(dialog),
        "dialog": "\n\n".join(lines),
    }


def main(dataset_path: str, output_dataset_path: str) -> None:
    dialogs = _reconstruct_dialogs(dataset_path)

    kept = 0
    with open(output_dataset_path, "w", encoding="utf-8") as f:
        for dialog in dialogs:
            if not _is_valid_dialog(dialog):
                continue
            record = _dialog_to_record(dialog)
            f.write(json.dumps(record) + "\n")
            kept += 1

    print(f"Wrote {kept} of {len(dialogs)} reconstructed dialogs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct dialogs from the Twitter customer support CSV, keeping only "
            f"single-customer/single-agent dialogs with {MIN_UTTERANCES}-"
            f"{MAX_UTTERANCES} utterances."
        )
    )
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dataset_path", type=str, required=True)
    args = parser.parse_args()

    main(args.dataset_path, args.output_dataset_path)
