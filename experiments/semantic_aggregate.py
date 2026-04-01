import json
import logging
from datetime import datetime
from pathlib import Path

from snowflake.snowpark import Session

from evergreen.claim_decomposer import ClaimDecomposer
from evergreen.common.constants import EMBEDDING_FIELD_SUFFIX, JSON_INDENT
from evergreen.model.config import CortexModelConfig
from experiments.common import (
    CONNECTION_NAME,
    DEFAULT_LANGUAGE_MODEL,
    LOGS_DIR,
    RESULTS_DIR,
    TIMESTAMP_FORMAT,
    setup_logging,
)

logger = logging.getLogger(__name__)


class SemanticAggregate:
    def __init__(
        self, name: str, dataset_path: Path, expr: str, prompt: str, language_model: str
    ) -> None:
        self._name = name
        self._dataset_path = dataset_path
        self._expr = expr
        self._prompt = prompt
        self._language_model = language_model

        self._timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)

        dataset_dir_name = self._dataset_path.parent.name
        self._dataset_name = self._dataset_path.stem

        agg_subdir = Path("semantic_aggregate") / dataset_dir_name / self._dataset_name
        logs_dir = LOGS_DIR / agg_subdir
        results_dir = RESULTS_DIR / agg_subdir

        for d in (logs_dir, results_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._log_file = logs_dir / f"{self._name}_{self._timestamp}.log"
        self._results_file = results_dir / f"{self._name}_{self._timestamp}.json"

        setup_logging(self._log_file)

    def execute(self) -> None:
        logger.debug("Starting semantic aggregate")

        aggregate = self._aggregate()
        claims = self._decompose(aggregate)

        self._write_aggregation_result(aggregate, claims)

        logger.debug("Semantic aggregate completed")

    def _aggregate(self) -> str:
        session = Session.builder.config("connection_name", CONNECTION_NAME).create()

        schema: list[str] = []
        rows: list[list[object]] = []
        with open(self._dataset_path) as f:
            for line in f:
                obj = json.loads(line)
                filtered_obj = {
                    k: v
                    for k, v in obj.items()
                    if not k.endswith(EMBEDDING_FIELD_SUFFIX)
                }
                if not schema:
                    schema = list(filtered_obj.keys())
                rows.append([filtered_obj[key] for key in schema])

        df = session.create_dataframe(rows, schema=schema)  # type: ignore

        df.write.save_as_table(
            self._dataset_name, mode="errorifexists", table_type="temporary"
        )
        query = session.sql(
            f"SELECT AI_AGG({self._expr}, '{self._prompt}', "
            f"{{'model': '{self._language_model}'}}) "
            f"FROM {self._dataset_name}"
        )
        aggregate = str(query.collect()[0][0])  # type: ignore

        logger.debug(f"Aggregate: {aggregate}")

        return aggregate

    def _decompose(self, aggregate: str) -> list[str]:
        logger.debug("Decomposing...")

        model_config = CortexModelConfig(
            language_models=[DEFAULT_LANGUAGE_MODEL],
            embedding_model="",
            connection_name=CONNECTION_NAME,
        )
        language_model = model_config.create_language_model(cache_dir=None)

        claim_decomposer = ClaimDecomposer(language_model)
        claims = claim_decomposer.decompose(aggregate)

        return claims

    def _write_aggregation_result(self, aggregate: str, claims: list[str]) -> None:
        logger.debug("Writing aggregation result...")

        results = {
            "metadata": {
                "name": self._name,
                "timestamp": self._timestamp,
                "dataset_path": str(self._dataset_path),
                "expr": self._expr,
                "prompt": self._prompt,
                "log_file": str(self._log_file),
                "language_model": self._language_model,
            },
            "aggregation_result": {
                "aggregate": aggregate,
                "claims": claims,
            },
        }

        with open(self._results_file, "w") as f:
            json.dump(results, f, indent=JSON_INDENT)
