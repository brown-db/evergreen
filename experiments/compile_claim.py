import json
from datetime import datetime
from pathlib import Path

from evergreen.catalog.schema import Schema
from evergreen.claim_compiler import ClaimCompiler
from evergreen.common.constants import JSON_INDENT
from evergreen.model.config import CortexModelConfig
from experiments.common import (
    CONNECTION_NAME,
    DEFAULT_LANGUAGE_MODEL,
    RESULTS_DIR,
    TIMESTAMP_FORMAT,
)


def compile_claim(name: str, claim: str, agg_result_path: Path, schema: Schema) -> None:
    with open(agg_result_path) as f:
        agg_result = json.load(f)

    agg_metadata = agg_result["metadata"]
    agg_name = agg_metadata["name"]
    agg_prompt = agg_metadata["prompt"]

    dataset_dir_name = agg_result_path.parent.parent.name
    dataset_name = agg_result_path.parent.name

    results_dir = (
        RESULTS_DIR / "claim_compiler" / dataset_dir_name / dataset_name / agg_name
    )

    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)

    results_file = results_dir / f"{name}_{timestamp}.json"

    model_config = CortexModelConfig(
        language_models=[DEFAULT_LANGUAGE_MODEL],
        embedding_model="",
        connection_name=CONNECTION_NAME,
    )
    language_model = model_config.create_language_model(cache_dir=None)

    compiler = ClaimCompiler(language_model)
    result = compiler.compile(agg_prompt, schema, claim)

    obj = {
        "metadata": {
            "name": name,
            "timestamp": timestamp,
            "agg_result_path": str(agg_result_path),
            "agg_prompt": agg_prompt,
            "schema": schema.to_dict(),
            "claim": claim,
        },
        "claim_compilation_result": result.to_dict(),
    }

    with open(results_file, "w") as f:
        json.dump(obj, f, indent=JSON_INDENT)
