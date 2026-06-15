import argparse
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from itertools import product
from pathlib import Path

import numpy as np

from evergreen.catalog.schema import Schema
from evergreen.claim_compiler import ClaimCompiler
from evergreen.common.constants import (
    JSON_INDENT,
    SENTENCE_EMBEDDINGS_FIELD_SUFFIX,
)
from evergreen.core.session_context import SessionContext
from evergreen.data_frame import DataFrame
from evergreen.model.config import CortexModelConfig
from evergreen.planner.logical.expr import Alias, Expr, Prompt, col
from evergreen.planner.logical.optimizer import InsertSimilarityFilter
from evergreen.planner.logical.plan import Filter, LogicalPlan, Projection
from evergreen.provenance import Monomial
from evergreen.storage.row import Row
from evergreen.storage.row_id import RowId
from experiments.baselines import rag_agent, reasoning_model, rlm
from experiments.common import (
    CHECKPOINTS_DIR,
    CONNECTION_NAME,
    DEFAULT_LANGUAGE_MODEL,
    EMBEDDING_MODEL,
    ENSEMBLE_LANGUAGE_MODELS,
    EVALUATION_LANGUAGE_MODELS,
    LOGS_DIR,
    RESULTS_DIR,
    TIMESTAMP_FORMAT,
    setup_logging,
)
from experiments.types import EvaluationResult

logger = logging.getLogger(__name__)


class Implementation(Enum):
    # Baseline implementations
    BASE_RM = "base_rm"  # Reasoning model
    RAG_AGENT = "rag_agent"  # RAG agent
    RLM = "rlm"  # RLM

    # Evergreen-based implementations
    EVG_REF = "evg_ref"  # Reference
    EVG_OPT = "evg_opt"  # Optimized with compiled query
    EVG_UNOPT = "evg_unopt"  # Unoptimized with compiled query
    EVG_OPT_REF_QUERY = "evg_opt_ref_query"  # Optimized with reference query
    EVG_UNOPT_REF_QUERY = "evg_unopt_ref_query"  # Unoptimized with reference query

    # Evergreen ablations
    EVG_ABL_NO_ES = "evg_abl_no_es"  # Early stopping
    EVG_ABL_NO_RS = "evg_abl_no_rs"  # Relevance sorting
    EVG_ABL_NO_ECS = "evg_abl_no_ecs"  # Estimation with confidence sequences
    EVG_ABL_NO_OF = "evg_abl_no_of"  # Operator fusion
    EVG_ABL_NO_SF = "evg_abl_no_sf"  # Similarity filtering
    EVG_ABL_NO_PC = "evg_abl_no_pc"  # Prompt caching


@dataclass(frozen=True)
class OptimizationConfig:
    early_stop: bool = True
    relevance_sort: bool = True
    estimation: bool = True
    fusion: bool = True
    similarity_filter: bool = True
    cache: bool = True


OPTIMIZATION_CONFIGS = {
    Implementation.EVG_ABL_NO_ES: OptimizationConfig(
        early_stop=False, relevance_sort=False, estimation=False
    ),
    Implementation.EVG_ABL_NO_RS: OptimizationConfig(relevance_sort=False),
    Implementation.EVG_ABL_NO_ECS: OptimizationConfig(estimation=False),
    Implementation.EVG_ABL_NO_OF: OptimizationConfig(fusion=False),
    Implementation.EVG_ABL_NO_SF: OptimizationConfig(similarity_filter=False),
    Implementation.EVG_ABL_NO_PC: OptimizationConfig(cache=False),
}


class CheckpointType(Enum):
    # Prior to semantic operations
    PRE_SEM_OP = "pre_sem_op"
    # After semantic operations
    POST_SEM_OP = "post_sem_op"


def parse_claim_evaluator_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--impls",
        nargs="*",
        choices=[i.value for i in Implementation],
        default=[],
    )
    parser.add_argument(
        "--lms",
        nargs="*",
        choices=EVALUATION_LANGUAGE_MODELS,
        default=[],
    )
    parser.add_argument("--eval_sim_filter", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--trial_count", type=int, default=3)
    return parser.parse_args()


class ClaimEvaluator(ABC):
    NAME: str
    CLAIM: str
    HINTS: str = ""
    SCHEMA: Schema
    TEXT_FIELD_NAME: str
    AGG_RESULT_PATH: Path
    RANDOM_SEED: int = 42
    CACHE_ID: str | None = None

    def __init__(self) -> None:
        with open(self.AGG_RESULT_PATH) as f:
            agg_result = json.load(f)

        agg_metadata = agg_result["metadata"]
        agg_name = agg_metadata["name"]
        self._dataset_path = agg_metadata["dataset_path"]
        self._agg_prompt = agg_metadata["prompt"]

        dataset_dir_name = self.AGG_RESULT_PATH.parent.parent.name
        dataset_name = self.AGG_RESULT_PATH.parent.name

        claim_subdir = (
            Path("claim_evaluator")
            / dataset_dir_name
            / dataset_name
            / agg_name
            / self.NAME
        )
        self._logs_dir = LOGS_DIR / claim_subdir
        self._results_dir = RESULTS_DIR / claim_subdir
        self._checkpoints_dir = CHECKPOINTS_DIR / claim_subdir

        self._timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)

        self._post_sem_op_reference_df_path = self._checkpoint_df_path(
            CheckpointType.POST_SEM_OP,
            Implementation.EVG_REF,
            ENSEMBLE_LANGUAGE_MODELS,
            trial_id=0,
        )

    @abstractmethod
    def reference_query(
        self,
        df: DataFrame,
        impl: Implementation,
        language_models: tuple[str, ...],
        trial_id: int,
    ) -> DataFrame:
        pass

    def _reference_plan(self) -> LogicalPlan:
        df = SessionContext().read_rows([], self.SCHEMA)
        return self.reference_query(
            df, Implementation.EVG_REF, ENSEMBLE_LANGUAGE_MODELS, 0
        ).logical_plan()

    def _semantic_filter_prompt_str(self) -> str | None:
        prompts = [
            node.predicate.prompt_str
            for node in self._reference_plan().walk()
            if isinstance(node, Filter) and isinstance(node.predicate, Prompt)
        ]
        assert len(prompts) <= 1
        return prompts[0] if prompts else None

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        names: list[str] = []
        for node in self._reference_plan().walk():
            if isinstance(node, Projection):
                for expr in node.exprs:
                    if (
                        isinstance(expr, Alias)
                        and isinstance(expr.expr, Prompt)
                        and expr.name not in names
                    ):
                        names.append(expr.name)
        # `walk()` is top-down, so the outermost (last-applied) map is seen
        # first; reverse to recover the order the maps appear in the query.
        return tuple(col(name) for name in reversed(names))

    def _log_file_path(
        self,
        impl: Implementation,
        language_models: tuple[str, ...],
        trial_id: int,
    ) -> Path:
        return (
            self._logs_dir / f"{impl.value}_{'_'.join(language_models)}"
            f"_{self._timestamp}_{trial_id}.log"
        )

    def _checkpoint_df_path(
        self,
        checkpoint_type: CheckpointType,
        impl: Implementation,
        language_models: tuple[str, ...],
        trial_id: int,
    ) -> Path:
        return (
            self._checkpoints_dir / f"{checkpoint_type.value}_{impl.value}"
            f"_{'_'.join(language_models)}_{trial_id}.pkl"
        )

    def _results_file_path(
        self,
        impl: Implementation,
        language_models: tuple[str, ...],
        trial_id: int,
    ) -> Path:
        return (
            self._results_dir / f"{impl.value}_{'_'.join(language_models)}"
            f"_{self._timestamp}_{trial_id}.json"
        )

    def _read_pickle(self, path: Path) -> DataFrame:
        return SessionContext().read_pickle(str(path))

    def _compiled_query_path(self, trial_id: int) -> Path:
        return self._results_dir / f"compiled_query_{trial_id}.json"

    def _load_compiled_query(self, trial_id: int) -> str:
        with open(self._compiled_query_path(trial_id)) as f:
            return json.load(f)["claim_compilation_result"]["query"]

    def evaluate(self, args: argparse.Namespace) -> None:
        if args.compile:
            self.compile_claim_to_query(args.trial_count)
            return

        impls = {Implementation(i) for i in args.impls}
        language_models = tuple(args.lms)
        trial_count = args.trial_count

        if Implementation.BASE_RM in impls:
            for language_model, trial_id in product(
                language_models, range(trial_count)
            ):
                self.evaluate_reasoning_model(language_model, trial_id)

        if Implementation.RAG_AGENT in impls:
            for language_model, trial_id in product(
                language_models, range(trial_count)
            ):
                self.evaluate_rag_agent(language_model, trial_id)

        if Implementation.RLM in impls:
            for language_model, trial_id in product(
                language_models, range(trial_count)
            ):
                self.evaluate_rlm(language_model, trial_id)

        if Implementation.EVG_REF in impls:
            self.evaluate_reference_query(trial_id=0)

        if Implementation.EVG_OPT in impls:
            for language_model, trial_id in product(
                language_models, range(trial_count)
            ):
                self.evaluate_optimized_query(
                    Implementation.EVG_OPT,
                    language_model,
                    trial_id,
                    use_reference_query=False,
                )

        if Implementation.EVG_UNOPT in impls:
            for language_model, trial_id in product(
                language_models, range(trial_count)
            ):
                self.evaluate_unoptimized_query(
                    Implementation.EVG_UNOPT,
                    language_model,
                    trial_id,
                    use_reference_query=False,
                )

        if Implementation.EVG_OPT_REF_QUERY in impls:
            for language_model, trial_id in product(
                language_models, range(trial_count)
            ):
                self.evaluate_optimized_query(
                    Implementation.EVG_OPT_REF_QUERY,
                    language_model,
                    trial_id,
                    use_reference_query=True,
                )

        if Implementation.EVG_UNOPT_REF_QUERY in impls:
            for language_model, trial_id in product(
                language_models, range(trial_count)
            ):
                self.evaluate_unoptimized_query(
                    Implementation.EVG_UNOPT_REF_QUERY,
                    language_model,
                    trial_id,
                    use_reference_query=True,
                )

        for impl in OPTIMIZATION_CONFIGS:
            if impl in impls:
                for language_model, trial_id in product(
                    language_models, range(trial_count)
                ):
                    self.evaluate_query_with_config(impl, language_model, trial_id)

        if args.eval_sim_filter:
            self.evaluate_sim_filter(trial_count)

    def compile_claim_to_query(self, trial_count: int) -> None:
        for trial_id in range(trial_count):
            model_config = CortexModelConfig(
                language_models=[DEFAULT_LANGUAGE_MODEL],
                embedding_model="",
                connection_name=CONNECTION_NAME,
            )
            language_model = model_config.create_language_model(cache_dir=None)

            compiler = ClaimCompiler(language_model)
            result = compiler.compile(
                self._agg_prompt, self.SCHEMA, self.CLAIM, self.HINTS
            )

            obj = {
                "metadata": {
                    "name": self.NAME,
                    "timestamp": self._timestamp,
                    "claim": self.CLAIM,
                    "hints": self.HINTS,
                    "schema": self.SCHEMA.to_dict(),
                    "agg_prompt": self._agg_prompt,
                },
                "claim_compilation_result": result.to_dict(),
            }

            compiled_query_path = self._compiled_query_path(trial_id)
            compiled_query_path.parent.mkdir(parents=True, exist_ok=True)
            with open(compiled_query_path, "w") as f:
                json.dump(obj, f, indent=JSON_INDENT)

    def evaluate_reasoning_model(self, language_model: str, trial_id: int) -> None:
        setup_logging(
            self._log_file_path(Implementation.BASE_RM, (language_model,), trial_id)
        )

        logger.debug("Evaluating reasoning model")

        evaluation_result = reasoning_model.evaluate_claim(
            self.CLAIM,
            self.HINTS,
            self._agg_prompt,
            self._dataset_path,
            self.SCHEMA,
            language_model,
        )

        self._write_evaluation_result(
            evaluation_result, Implementation.BASE_RM, (language_model,), trial_id
        )

        logger.debug("Reasoning model evaluation completed")

    def evaluate_rag_agent(self, language_model: str, trial_id: int) -> None:
        setup_logging(
            self._log_file_path(Implementation.RAG_AGENT, (language_model,), trial_id)
        )

        logger.debug("Evaluating RAG model")

        evaluation_result = rag_agent.evaluate_claim(
            self.CLAIM,
            self.HINTS,
            self._agg_prompt,
            self._dataset_path,
            self.SCHEMA,
            self.TEXT_FIELD_NAME,
            language_model,
        )

        self._write_evaluation_result(
            evaluation_result,
            Implementation.RAG_AGENT,
            (language_model,),
            trial_id,
        )

        logger.debug("RAG model evaluation completed")

    def evaluate_rlm(self, language_model: str, trial_id: int) -> None:
        setup_logging(
            self._log_file_path(Implementation.RLM, (language_model,), trial_id)
        )

        logger.debug("Evaluating RLM")

        evaluation_result = rlm.evaluate_claim(
            self.CLAIM,
            self.HINTS,
            self._agg_prompt,
            self._dataset_path,
            self.SCHEMA,
            self.TEXT_FIELD_NAME,
            language_model,
        )

        self._write_evaluation_result(
            evaluation_result,
            Implementation.RLM,
            (language_model,),
            trial_id,
        )

        logger.debug("RLM evaluation completed")

    def evaluate_reference_query(self, trial_id: int) -> None:
        setup_logging(
            self._log_file_path(
                Implementation.EVG_REF, ENSEMBLE_LANGUAGE_MODELS, trial_id
            )
        )

        logger.debug("Evaluating reference query")

        ctx = SessionContext()
        ctx.enable_batching(batch_size=2048)
        ctx.enable_minimal_provenance()
        # We can add cache for reference, since we do not care about cost and
        # latency here.
        ctx.enable_cache(f"{Implementation.EVG_REF.value}_cache")
        ctx.register_model_config(
            CortexModelConfig(
                ENSEMBLE_LANGUAGE_MODELS,
                EMBEDDING_MODEL,
                CONNECTION_NAME,
            )
        )

        if self._post_sem_op_reference_df_path.exists():
            logger.debug(
                f"{self._post_sem_op_reference_df_path} exists, "
                "skipping reference query evaluation"
            )
            return

        df = ctx.read_json(self._dataset_path, self.SCHEMA.key)
        df = self.reference_query(
            df, Implementation.EVG_REF, ENSEMBLE_LANGUAGE_MODELS, trial_id
        )
        result = df.collect()

        verification_result, provenance = (
            self._extract_verification_result_and_provenance(result.rows, df.schema())
        )
        prov_tokens = {token: True for token in provenance}

        evaluation_result = EvaluationResult(
            verification_result=verification_result,
            query_metrics=result.metrics,
            filter_metrics=None,
            map_metrics=None,
            prov_tokens=prov_tokens,
            reasoning=None,
        )

        self._write_evaluation_result(
            evaluation_result,
            Implementation.EVG_REF,
            ENSEMBLE_LANGUAGE_MODELS,
            trial_id,
        )

        logger.debug("Reference query evaluation completed")

    def evaluate_optimized_query(
        self,
        impl: Implementation,
        language_model: str,
        trial_id: int,
        use_reference_query: bool,
    ) -> None:
        setup_logging(self._log_file_path(impl, (language_model,), trial_id))

        logger.debug("Evaluating optimized query (%s)", impl.value)

        ctx = SessionContext.create_optimized(
            random_seed=self.RANDOM_SEED + trial_id,
            cache_id=f"{self.CACHE_ID}_{impl.value}_{language_model}_{trial_id}"
            if self.CACHE_ID
            else None,
        )
        ctx.register_model_config(
            CortexModelConfig(
                [language_model],
                EMBEDDING_MODEL,
                CONNECTION_NAME,
            )
        )

        self._evaluate_query(
            ctx, impl, (language_model,), trial_id, use_reference_query
        )

        logger.debug("Optimized query evaluation completed (%s)", impl.value)

    def evaluate_unoptimized_query(
        self,
        impl: Implementation,
        language_model: str,
        trial_id: int,
        use_reference_query: bool,
    ) -> None:
        setup_logging(self._log_file_path(impl, (language_model,), trial_id))

        logger.debug("Evaluating unoptimized query (%s)", impl.value)

        ctx = SessionContext()
        ctx.enable_batching()
        ctx.enable_minimal_provenance()
        ctx.register_model_config(
            CortexModelConfig(
                [language_model],
                EMBEDDING_MODEL,
                CONNECTION_NAME,
            )
        )

        if use_reference_query:
            post_sem_op_unopt_df_path = self._checkpoint_df_path(
                CheckpointType.POST_SEM_OP,
                impl,
                (language_model,),
                trial_id,
            )
            if post_sem_op_unopt_df_path.exists():
                logger.debug(
                    f"{post_sem_op_unopt_df_path} exists, "
                    "skipping unoptimized query evaluation"
                )
                return

        self._evaluate_query(
            ctx, impl, (language_model,), trial_id, use_reference_query
        )

        logger.debug("Unoptimized query evaluation completed (%s)", impl.value)

    def evaluate_query_with_config(
        self, impl: Implementation, language_model: str, trial_id: int
    ) -> None:
        setup_logging(self._log_file_path(impl, (language_model,), trial_id))

        logger.debug("Evaluating query with config")

        config = OPTIMIZATION_CONFIGS[impl]

        ctx = SessionContext()
        ctx.enable_batching()
        ctx.enable_minimal_provenance()

        if config.early_stop:
            ctx.enable_early_stop()
        if config.relevance_sort:
            ctx.enable_relevance_sort()
        if config.estimation:
            ctx.enable_estimation(random_seed=self.RANDOM_SEED + trial_id)
        if config.fusion:
            ctx.enable_fusion()
        if config.similarity_filter:
            ctx.enable_similarity_filter()
        if config.cache:
            ctx.enable_cache(
                f"{self.CACHE_ID}_{impl.value}_{language_model}_{trial_id}"
                if self.CACHE_ID
                else None
            )

        ctx.register_model_config(
            CortexModelConfig([language_model], EMBEDDING_MODEL, CONNECTION_NAME)
        )

        self._evaluate_query(
            ctx, impl, (language_model,), trial_id, use_reference_query=True
        )

        logger.debug("Query with config evaluation completed")

    def _evaluate_query(
        self,
        ctx: SessionContext,
        impl: Implementation,
        language_models: tuple[str, ...],
        trial_id: int,
        use_reference_query: bool,
    ) -> None:
        df = ctx.read_json(self._dataset_path, self.SCHEMA.key)
        if use_reference_query:
            df = self.reference_query(df, impl, language_models, trial_id)
        else:
            df = ClaimCompiler.build_query(df, self._load_compiled_query(trial_id))
        result = df.collect()

        if not use_reference_query:
            verification_result, provenance = (
                self._extract_verification_result_and_provenance(
                    result.rows, df.schema()
                )
            )
            self._write_evaluation_result(
                EvaluationResult(
                    verification_result=verification_result,
                    query_metrics=result.metrics,
                    filter_metrics=None,
                    map_metrics=None,
                    prov_tokens={token: None for token in provenance},
                    reasoning=None,
                ),
                impl,
                language_models,
                trial_id,
            )
            return

        verification_result, provenance = (
            self._extract_verification_result_and_provenance(result.rows, df.schema())
        )

        filter_metrics = None
        map_metrics = None
        prov_tokens = {token: False for token in provenance}

        post_sem_op_df_path = self._checkpoint_df_path(
            CheckpointType.POST_SEM_OP, impl, language_models, trial_id
        )
        if (
            self._post_sem_op_reference_df_path.exists()
            and post_sem_op_df_path.exists()
        ):
            post_sem_op_reference_df = self._read_pickle(
                self._post_sem_op_reference_df_path
            )
            post_sem_op_df = self._read_pickle(post_sem_op_df_path)

            if df.logical_plan().is_or_above(Filter):
                pre_sem_op_df_path = self._checkpoint_df_path(
                    CheckpointType.PRE_SEM_OP, impl, language_models, trial_id
                )
                if pre_sem_op_df_path.exists():
                    pre_sem_op_df = self._read_pickle(pre_sem_op_df_path)
                    filter_metrics = post_sem_op_reference_df.evaluate_filter(
                        pre_sem_op_df, post_sem_op_df
                    )
                else:
                    logger.warning(
                        "%s does not exist, skipping filter evaluation",
                        pre_sem_op_df_path,
                    )

            map_metrics = post_sem_op_reference_df.evaluate_map(
                post_sem_op_df, self.semantic_map_columns()
            )

            prov_tokens = post_sem_op_reference_df.evaluate_tokens(provenance)
        else:
            logger.warning(
                "%s or %s do not exist, skipping filter, map, and provenance "
                "evaluation",
                self._post_sem_op_reference_df_path,
                post_sem_op_df_path,
            )

        evaluation_result = EvaluationResult(
            verification_result=verification_result,
            query_metrics=result.metrics,
            filter_metrics=filter_metrics,
            map_metrics=map_metrics,
            prov_tokens=prov_tokens,
            reasoning=None,
        )

        self._write_evaluation_result(
            evaluation_result, impl, language_models, trial_id
        )

    def _verdict_index(self, schema: Schema) -> int:
        # `check` always appends the verdict as the final column.
        return len(schema.fields) - 1

    def _extract_verification_result_and_provenance(
        self, rows: list[Row], schema: Schema
    ) -> tuple[bool, Monomial]:
        assert len(rows) == 1
        verification_result = rows[0][self._verdict_index(schema)]
        assert isinstance(verification_result, bool)
        monomials = rows[0].get_prov_monomials(self._verdict_index(schema))
        assert len(monomials) == 1
        return verification_result, monomials[0]

    def _write_evaluation_result(
        self,
        evaluation_result: EvaluationResult,
        impl: Implementation,
        language_models: tuple[str, ...],
        trial_id: int,
    ) -> None:
        results = {
            "metadata": {
                "name": self.NAME,
                "implementation": impl.value,
                "timestamp": self._timestamp,
                "trial_id": trial_id,
                "dataset_path": self._dataset_path,
                "log_file": str(self._log_file_path(impl, language_models, trial_id)),
                "random_seed": self.RANDOM_SEED,
            },
            "evaluation_result": evaluation_result.to_dict(),
        }

        results_file_path = self._results_file_path(impl, language_models, trial_id)
        results_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file_path, "w") as f:
            json.dump(results, f, indent=JSON_INDENT)

    def evaluate_sim_filter(self, trial_count: int) -> None:
        filter_prompt_str = self._semantic_filter_prompt_str()
        if filter_prompt_str is None:
            return

        # The sensitivity analysis only depends on the (dataset, filter prompt)
        # pair, so skip claims that share a filter prompt already analyzed.
        for existing_path in RESULTS_DIR.rglob("sim_filter_sensitivity_analysis.json"):
            with open(existing_path) as f:
                existing_metadata = json.load(f)["metadata"]
            if (
                existing_metadata["filter_prompt"] == filter_prompt_str
                and existing_metadata["dataset_path"] == self._dataset_path
            ):
                return

        thresholds = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55]

        assert self._post_sem_op_reference_df_path.exists()

        ref_df = self._read_pickle(self._post_sem_op_reference_df_path)
        ref_schema = ref_df.schema()
        ref_row_ids = {row.id(ref_schema) for row in ref_df.collect().rows}

        query_vectors: list[np.ndarray] = []
        for _ in range(trial_count):
            ctx = SessionContext()
            ctx.register_model_config(
                CortexModelConfig(
                    [EVALUATION_LANGUAGE_MODELS[0]], EMBEDDING_MODEL, CONNECTION_NAME
                )
            )
            query_vector = InsertSimilarityFilter.generate_query_vector(
                filter_prompt_str, ctx.session_state()
            )
            assert query_vector is not None
            query_vectors.append(np.array(query_vector))

        input_df = SessionContext().read_json(self._dataset_path, self.SCHEMA.key)
        input_rows = input_df.collect().rows
        input_schema = input_df.schema()

        sentence_embeddings_field_name = next(
            name
            for name in input_schema.field_names()
            if name.endswith(SENTENCE_EMBEDDINGS_FIELD_SUFFIX)
        )
        sentence_embeddings_index = input_schema.index_of(
            sentence_embeddings_field_name
        )

        results: list[dict[str, object]] = []

        for threshold in thresholds:
            trials: list[dict[str, object]] = []

            for query_vector in query_vectors:
                selected_row_ids: set[RowId] = set()

                for row in input_rows:
                    sentence_embeddings = row[sentence_embeddings_index]
                    max_similarity = np.max(
                        np.array(sentence_embeddings) @ query_vector
                    )

                    if max_similarity >= threshold:
                        selected_row_ids.add(row.id(input_schema))

                trials.append(
                    {
                        "tp": len(ref_row_ids & selected_row_ids),
                        "selected_count": len(selected_row_ids),
                    }
                )

            results.append(
                {
                    "threshold": threshold,
                    "trials": trials,
                    "ground_truth_selected_row_count": len(ref_row_ids),
                    "total_row_count": len(input_rows),
                }
            )

        output_path = self._results_dir / "sim_filter_sensitivity_analysis.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "name": self.NAME,
                        "dataset_path": self._dataset_path,
                        "filter_prompt": filter_prompt_str,
                    },
                    "results": results,
                },
                f,
                indent=JSON_INDENT,
            )
