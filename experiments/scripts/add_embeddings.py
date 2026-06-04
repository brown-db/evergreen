import argparse
import json
from pathlib import Path

from tqdm import tqdm

from evergreen.common.constants import (
    CACHE_DIR_ROOT,
    EMBEDDING_FIELD_SUFFIX,
    SENTENCE_EMBEDDINGS_FIELD_SUFFIX,
)
from evergreen.model.config import CortexModelConfig
from experiments.common import CONNECTION_NAME, EMBEDDING_MODEL


def main(
    dataset_path: str, fields: tuple[str, ...], output_dataset_path: str | None
) -> None:
    input_path = Path(dataset_path)

    if output_dataset_path:
        output_path = Path(output_dataset_path)
    else:
        output_path = input_path.with_suffix(".embed.jsonl")

    cache_dir = CACHE_DIR_ROOT / "add_embeddings"

    model_config = CortexModelConfig(
        language_models=[""],
        embedding_model=EMBEDDING_MODEL,
        connection_name=CONNECTION_NAME,
    )
    model = model_config.create_embedding_model(cache_dir=cache_dir)

    with open(input_path) as in_f:
        objs = [json.loads(line) for line in in_f]

    for field in tqdm(fields, desc="Adding embeddings"):
        results = model.embed_with_sentences_batch([obj[field] for obj in objs])
        for obj, (doc_embedding, sentence_embeddings) in zip(
            objs, results, strict=True
        ):
            obj[field + EMBEDDING_FIELD_SUFFIX] = doc_embedding
            obj[field + SENTENCE_EMBEDDINGS_FIELD_SUFFIX] = sentence_embeddings

    with open(output_path, "w") as out_f:
        for obj in objs:
            out_f.write(json.dumps(obj) + "\n")

    if not output_dataset_path:
        input_path.unlink()
        output_path.rename(input_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--fields", nargs="+", required=True)
    parser.add_argument("--output_dataset_path", type=str)
    args = parser.parse_args()

    main(args.dataset_path, tuple(args.fields), args.output_dataset_path)
