import json
import shutil
from collections.abc import Callable
from enum import Enum, auto
from functools import cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiments.common import DEFAULT_LANGUAGE_MODEL, MODEL_PRICING

RESULTS_DIR = Path("experiments/results/claim_evaluator")
FIGURES_DIR = Path("experiments/figures")
PAPER_DIR = Path("paper")
PAPER_FIGURES_DIR = PAPER_DIR / "figures"

CLAIM_TO_CLAIM_ID = {
    "johns_roast_pork\nsummarize\ncardinal_claim_1": "C1",
    "johns_roast_pork\nsummarize\nexistential_claim_1": "C2",
    "johns_roast_pork\nsummarize\nproportional_claim_1": "C3",
    "johns_roast_pork\nsummarize\nuniversal_claim_1": "C4",
    "mcdonalds_mo\ncompare\nexistential_universal_claim_1": "C5",
    "mcdonalds_mo\ncompare\nproportional_cardinal_claim_1": "C6",
    "mcdonalds_mo\ncompare\nproportional_existential_claim_1": "C7",
    "mcdonalds_mo\ncompare\nuniversal_cardinal_claim_1": "C8",
    "mcdonalds_mo\nrank\nordinal_claim_1": "C9",
    "mcdonalds_mo\nrank\nordinal_claim_2": "C10",
    "mcdonalds_mo\nrank\nordinal_claim_3": "C11",
    "mcdonalds_mo\nrank\nordinal_claim_4": "C12",
    "village_whiskey\nsummarize\ncardinal_claim_1": "C13",
    "village_whiskey\nsummarize\nexistential_claim_1": "C14",
    "village_whiskey\nsummarize\nproportional_claim_1": "C15",
    "village_whiskey\nsummarize\nuniversal_claim_1": "C16",
    "play_station\nsummarize\ncardinal_claim_1": "C17",
    "play_station\nsummarize\ncardinal_claim_2": "C18",
    "play_station\nsummarize\nproportional_claim_1": "C19",
    "play_station\nsummarize\nproportional_claim_2": "C20",
    "airlines\ncompare\ncardinal_proportional_claim_1": "C21",
    "airlines\ncompare\ncardinal_universal_claim_1": "C22",
    "airlines\ncompare\nexistential_proportional_claim_1": "C23",
    "airlines\ncompare\nuniversal_existential_claim_1": "C24",
    "airlines\nrank\nordinal_claim_1": "C25",
    "airlines\nrank\nordinal_claim_2": "C26",
    "airlines\nrank\nordinal_claim_3": "C27",
    "airlines\nrank\nordinal_claim_4": "C28",
    "uber\nsummarize\ncardinal_claim_1": "C29",
    "uber\nsummarize\ncardinal_claim_2": "C30",
    "uber\nsummarize\nproportional_claim_1": "C31",
    "uber\nsummarize\nproportional_claim_2": "C32",
}

CLAIM_ID_TO_CLAIM = {v: k for k, v in CLAIM_TO_CLAIM_ID.items()}


@cache
def claim_operator_ids() -> dict[str, tuple[object | None, object | None]]:
    """Map each claim key (``'{dataset}\\n{task}\\n{NAME}'``) to (filter, map) ids.

    Operator ids are content-derived (kind + dataset + prompt [+ return_type])
    via ``agreement._dedup_key``, so claims that reuse the same semantic operator
    share an id and can be de-duplicated. Every benchmark claim has exactly one
    map and at most one filter, so a single (filter_id, map_id) pair per claim
    suffices. The key matches the ``"claim"`` column built in ``load_results``.
    """
    from experiments.claim_evaluator import SemanticOperator, SemanticOperatorKind
    from experiments.scripts.agreement import all_evaluator_classes

    def op_id(operator: SemanticOperator, dataset_path: str) -> tuple[object, ...]:
        # Mirror agreement._dedup_key: maps also key on return_type (same prompt
        # with a different label space is a different decision); filters are bool.
        if operator.kind is SemanticOperatorKind.MAP:
            return (
                operator.kind,
                dataset_path,
                operator.prompt_str,
                operator.return_type,
            )
        return (operator.kind, dataset_path, operator.prompt_str)

    mapping: dict[str, tuple[object | None, object | None]] = {}
    for cls in all_evaluator_classes():
        evaluator = cls()
        task = cls.AGG_RESULT_PATH.name.split("_")[0]
        key = f"{cls.AGG_RESULT_PATH.parent.name}\n{task}\n{cls.NAME}"

        filter_id: object | None = None
        map_id: object | None = None
        for operator in evaluator.semantic_operators():
            if operator.kind is SemanticOperatorKind.FILTER:
                filter_id = op_id(operator, evaluator.dataset_path)
            else:
                map_id = op_id(operator, evaluator.dataset_path)
        mapping[key] = (filter_id, map_id)
    return mapping


IMPL_MODEL_LABELS = {
    "base_rm_claude-opus-4-6": "base_rm Claude Opus 4.6",
    "base_rm_claude-sonnet-4-6": "base_rm Claude Sonnet 4.6",
    "base_rm_claude-haiku-4-5": "base_rm Claude Haiku 4.5",
    "rag_agent_claude-opus-4-6": "rag_agent Claude Opus 4.6",
    "rag_agent_claude-sonnet-4-6": "rag_agent Claude Sonnet 4.6",
    "rag_agent_claude-haiku-4-5": "rag_agent Claude Haiku 4.5",
    "rlm_claude-opus-4-6": "rlm Claude Opus 4.6",
    "rlm_claude-sonnet-4-6": "rlm Claude Sonnet 4.6",
    "rlm_claude-haiku-4-5": "rlm Claude Haiku 4.5",
    "evg_unopt_claude-opus-4-6": "evg_unopt Claude Opus 4.6",
    "evg_unopt_claude-sonnet-4-6": "evg_unopt Claude Sonnet 4.6",
    "evg_unopt_claude-haiku-4-5": "evg_unopt Claude Haiku 4.5",
    "evg_opt_claude-opus-4-6": "evg_opt Claude Opus 4.6",
    "evg_opt_claude-sonnet-4-6": "evg_opt Claude Sonnet 4.6",
    "evg_opt_claude-haiku-4-5": "evg_opt Claude Haiku 4.5",
    "evg_opt_qwen3-vl-235b-a22b": "evg_opt Qwen3-VL",
    "evg_opt_qwen3-next-80b-a3b": "evg_opt Qwen3-Next",
    # Reference-query variants emit the filter/map/provenance metric.
    "evg_unopt_ref_query_claude-opus-4-6": "evg_unopt_ref_query Claude Opus 4.6",
    "evg_unopt_ref_query_claude-sonnet-4-6": "evg_unopt_ref_query Claude Sonnet 4.6",
    "evg_unopt_ref_query_claude-haiku-4-5": "evg_unopt_ref_query Claude Haiku 4.5",
    "evg_opt_ref_query_claude-opus-4-6": "evg_opt_ref_query Claude Opus 4.6",
    "evg_opt_ref_query_claude-sonnet-4-6": "evg_opt_ref_query Claude Sonnet 4.6",
    "evg_opt_ref_query_claude-haiku-4-5": "evg_opt_ref_query Claude Haiku 4.5",
    "evg_opt_ref_query_qwen3-vl-235b-a22b": "evg_opt_ref_query Qwen3-VL",
    "evg_opt_ref_query_qwen3-next-80b-a3b": "evg_opt_ref_query Qwen3-Next",
}

IMPLEMENTATIONS = [
    "base_rm",
    "rag_agent",
    "rlm",
    "evg_unopt",
    "evg_opt",
]

# Implementations that run the reference query and therefore carry the
# filter/map/provenance operator-quality metrics.
REF_QUERY_IMPLEMENTATIONS = ["evg_unopt_ref_query", "evg_opt_ref_query"]

# Implementations that execute the compiled query and therefore incur the
# one-time claim-compilation cost/latency.
COMPILED_QUERY_IMPLEMENTATIONS = ["evg_unopt", "evg_opt"]

_MODEL_LABELS = {
    "claude-opus-4-6": "Opus 4.6",
    "claude-sonnet-4-6": "Sonnet 4.6",
    "claude-haiku-4-5": "Haiku 4.5",
    "qwen3-vl-235b-a22b": "Qwen3-VL",
    "qwen3-next-80b-a3b": "Qwen3-Next",
}

# LaTeX row macros for the results tables. The ``*_ref_query`` runs supply the
# operator-quality metrics but are displayed under the same implementation macro.
LATEX_IMPL_MACROS = {
    "base_rm": r"\baserm",
    "rag_agent": r"\ragagent",
    "rlm": r"\rlm",
    "evg_unopt": r"\evgunopt",
    "evg_opt": r"\evgopt",
    "evg_unopt_ref_query": r"\evgunopt",
    "evg_opt_ref_query": r"\evgopt",
}

LATEX_MODEL_NAMES = {
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "qwen3-vl-235b-a22b": "Qwen3-VL",
    "qwen3-next-80b-a3b": "Qwen3-Next",
}

ABLATION_MODEL = "claude-haiku-4-5"

_ABLATION_STEP_LABELS = {
    "evg_opt": "All Opt",
    "evg_abl_no_es": "ES",
    "evg_abl_no_rs": "RS",
    "evg_abl_no_ecs": "ECS",
    "evg_abl_no_of": "OF",
    "evg_abl_no_sf": "SF",
    "evg_abl_no_pc": "PC",
}

ABLATION_IMPLEMENTATIONS = list(_ABLATION_STEP_LABELS)

ABLATION_LABEL_MAP = {
    f"{impl}_{ABLATION_MODEL}": label for impl, label in _ABLATION_STEP_LABELS.items()
}

ABLATION_APPLICABLE_CLAIMS = {
    "evg_abl_no_es": (
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C13",
        "C14",
        "C15",
        "C16",
        "C17",
        "C18",
        "C19",
        "C20",
        "C21",
        "C22",
        "C23",
        "C24",
        "C29",
        "C30",
        "C31",
        "C32",
    ),
    "evg_abl_no_rs": ("C2", "C6", "C7", "C8", "C13", "C14", "C24"),
    "evg_abl_no_ecs": (
        "C1",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C15",
        "C16",
        "C17",
        "C18",
        "C19",
        "C20",
        "C21",
        "C22",
        "C23",
        "C24",
        "C29",
        "C30",
        "C31",
        "C32",
    ),
    "evg_abl_no_of": (
        "C3",
        "C9",
        "C10",
        "C11",
        "C12",
        "C16",
        "C21",
        "C22",
    ),
    "evg_abl_no_sf": (
        "C3",
        "C9",
        "C10",
        "C11",
        "C12",
        "C16",
        "C21",
        "C22",
    ),
    "evg_abl_no_pc": ("C9", "C10", "C11", "C12", "C25", "C26", "C27", "C28"),
}


FIGURE_SIZE = (40, 5)


class FormatType(Enum):
    COST = auto()
    LATENCY = auto()
    SCORE = auto()


def compute_token_costs(
    language_model: str, lm_metrics: list[dict[str, Any]]
) -> tuple[float, float]:
    pricing = MODEL_PRICING[language_model]

    input_token_count = sum(m["input_token_count"] for m in lm_metrics)
    output_token_count = sum(m["output_token_count"] for m in lm_metrics)
    cache_creation_input_token_count = sum(
        m.get("cache_creation_input_token_count", 0) for m in lm_metrics
    )
    cache_read_input_token_count = sum(
        m.get("cache_read_input_token_count", 0) for m in lm_metrics
    )

    input_token_cost = (
        (
            input_token_count
            + cache_creation_input_token_count
            + cache_read_input_token_count
        )
        * pricing["input"]
        / 1e6
    )
    output_token_cost = output_token_count * pricing["output"] / 1e6

    return input_token_cost, output_token_cost


def get_impl_models(
    df: pd.DataFrame,
    impls: list[str] | None = None,
    labels: dict[str, str] | None = None,
) -> list[str]:
    if impls is None:
        impls = IMPLEMENTATIONS
    label_map = labels if labels is not None else IMPL_MODEL_LABELS
    impl_models: list[str] = []
    for impl in impls:
        impl_df = df[df["implementation"] == impl]
        # Order models consistently based on MODEL_PRICING key order
        for model in MODEL_PRICING:
            key = f"{impl}_{model}"
            if key in impl_df["impl_model"].values and key in label_map:
                impl_models.append(key)
    return impl_models


def get_reference(df: pd.DataFrame) -> pd.Series:
    """Ground-truth verification result per claim (the evg_ref ensemble)."""
    return df[df["implementation"] == "evg_ref"].set_index("claim")[
        "verification_result"
    ]


def get_ablation_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rows used for ablations: the ablation model plus the reference labels."""
    return df[(df["model"] == ABLATION_MODEL) | (df["implementation"] == "evg_ref")]


def load_compilation_costs() -> dict[tuple[str, int], dict[str, float]]:
    """Map (claim, trial_id) -> compilation cost/latency/token counts.

    Compilation is a one-time planning step (run with the compiler model) that
    produces the query consumed by evg_opt / evg_unopt, so its cost and latency
    are attributed to those implementations.
    """
    costs: dict[tuple[str, int], dict[str, float]] = {}

    for file_path in RESULTS_DIR.rglob("compiled_query_*.json"):
        with open(file_path) as file:
            data = json.load(file)

        compilation = data["claim_compilation_result"]
        metadata = data["metadata"]
        lm_metrics = compilation["language_model_metrics"]

        input_cost, output_cost = compute_token_costs(
            DEFAULT_LANGUAGE_MODEL, lm_metrics
        )

        path_parts = file_path.relative_to(RESULTS_DIR).parts
        claim = f"{path_parts[1]}\n{path_parts[2]}\n{metadata['name']}"
        trial_id = int(file_path.stem.split("_")[-1])

        costs[(claim, trial_id)] = {
            "input_token_cost": input_cost,
            "output_token_cost": output_cost,
            "total_token_cost": input_cost + output_cost,
            "lm_call_count": sum(
                m["prompt_count"] - m["prompt_cache_hit_count"] for m in lm_metrics
            ),
            "input_token_count": sum(m["input_token_count"] for m in lm_metrics),
            "output_token_count": sum(m["output_token_count"] for m in lm_metrics),
            "latency": compilation["latency"],
        }

    return costs


def load_results() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    compilation_costs = load_compilation_costs()

    for file_path in RESULTS_DIR.rglob("*.json"):
        with open(file_path) as file:
            data = json.load(file)

        if "evaluation_result" not in data:
            continue

        metadata = data["metadata"]
        evaluation_result = data["evaluation_result"]
        query_metrics = evaluation_result["query_metrics"]

        exec_lm_metrics = query_metrics["language_model_metrics"]
        opt_lm_metrics = query_metrics["optimizer_language_model_metrics"]

        model_name = exec_lm_metrics[0]["model_name"]
        impl = metadata["implementation"]

        # evg_ref runs the ground-truth ensemble; its cost is never reported and
        # its models are not in MODEL_PRICING, so skip pricing it entirely.
        if impl == "evg_ref":
            input_token_cost = float("nan")
            output_token_cost = float("nan")
            total_token_cost = float("nan")
        else:
            exec_input_token_cost, exec_output_token_cost = compute_token_costs(
                model_name, exec_lm_metrics
            )
            opt_input_token_cost, opt_output_token_cost = compute_token_costs(
                DEFAULT_LANGUAGE_MODEL, opt_lm_metrics
            )

            input_token_cost = exec_input_token_cost + opt_input_token_cost
            output_token_cost = exec_output_token_cost + opt_output_token_cost
            total_token_cost = input_token_cost + output_token_cost

        lm_metrics = exec_lm_metrics + opt_lm_metrics

        lm_call_count = sum(
            m["prompt_count"] - m["prompt_cache_hit_count"] for m in lm_metrics
        )
        input_token_count = sum(m["input_token_count"] for m in lm_metrics)
        output_token_count = sum(m["output_token_count"] for m in lm_metrics)

        latency = query_metrics["planning_latency"] + query_metrics["execution_latency"]

        # Parse path: .../claim_evaluator/{source}/{dataset}/{task}/{claim}/
        path_parts = file_path.relative_to(RESULTS_DIR).parts
        claim_key = f"{path_parts[1]}\n{path_parts[2]}\n{metadata['name']}"
        filter_op_id, map_op_id = claim_operator_ids()[claim_key]

        row: dict[str, Any] = {
            "claim": claim_key,
            "filter_op_id": filter_op_id,
            "map_op_id": map_op_id,
            "implementation": impl,
            "model": model_name,
            "trial_id": metadata["trial_id"],
            "impl_model": f"{impl}_{model_name}",
            "verification_result": evaluation_result["verification_result"],
            "input_token_cost": input_token_cost,
            "output_token_cost": output_token_cost,
            "total_token_cost": total_token_cost,
            "lm_call_count": lm_call_count,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "latency": latency,
        }

        if impl in COMPILED_QUERY_IMPLEMENTATIONS:
            compilation = compilation_costs[(row["claim"], row["trial_id"])]
            for field in (
                "input_token_cost",
                "output_token_cost",
                "total_token_cost",
                "lm_call_count",
                "input_token_count",
                "output_token_count",
                "latency",
            ):
                row[field] += compilation[field]

        filter_metrics = evaluation_result["filter_metrics"]
        map_metrics = evaluation_result["map_metrics"]
        prov_tokens = evaluation_result["prov_tokens"]

        if prov_tokens and all(t["is_valid"] is not None for t in prov_tokens):
            row["prov_valid_count"] = sum(t["is_valid"] for t in prov_tokens)
            row["prov_total"] = len(prov_tokens)
            row["provenance_precision"] = row["prov_valid_count"] / row["prov_total"]

        if filter_metrics:
            # Raw counts for aggregate computation
            row["filter_tp"] = filter_metrics["tp"]
            row["filter_fp"] = filter_metrics["fp"]
            row["filter_fn"] = filter_metrics["fn"]
            # Computed metrics for per-claim view
            row["filter_precision"] = filter_metrics["precision"]
            row["filter_recall"] = filter_metrics["recall"]
            row["filter_f1"] = filter_metrics["f1_score"]

        if map_metrics:
            # Raw counts for aggregate computation
            row["map_correct"] = sum(map_metrics["correct_by_column"].values())
            row["map_total"] = map_metrics["total_rows"] * len(
                map_metrics["correct_by_column"]
            )
            # Computed metric for per-claim view
            row["map_accuracy"] = map_metrics["accuracy"]

        rows.append(row)

    return pd.DataFrame(rows)


def format_value(value: float, format_type: FormatType) -> str:
    match format_type:
        case FormatType.COST:
            return f"${value:.2f}" if value >= 1 else f"${value:.3f}"
        case FormatType.LATENCY:
            return f"{value / 1e3:.1f}k" if value >= 1e3 else f"{value:.0f}"
        case FormatType.SCORE:
            return f"{value:.2f}"


def save_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{filename}.pdf")
    plt.close()


def compute_precision_recall_f1(
    tp: int, fp: int, fn: int
) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1


def hallucination_counts(ref: pd.Series, pred: pd.Series) -> tuple[int, int, int]:
    """Confusion counts with the positive class being a detected hallucination.

    ``verification_result`` is ``True`` when a claim is grounded, so a hallucination
    (false claim) is a ``False`` value. Negating turns "not grounded" into the
    positive class for precision/recall/F1.
    """
    tp = int((~ref & ~pred).sum())  # both flag a hallucination
    fp = int((ref & ~pred).sum())  # predicted hallucination, actually grounded
    fn = int((~ref & pred).sum())  # missed an actual hallucination
    return tp, fp, fn


def _mean_range(values: list[float], format_type: FormatType) -> str:
    """Format per-trial values as ``mean [min, max]`` via the shared formatter.

    Cost cells live under a ``Cost (\\$)`` header, so the ``$`` that
    ``format_value`` prepends for currency is stripped.
    """
    arr = np.array(values, dtype=float)
    cell = (
        f"{format_value(float(arr.mean()), format_type)} "
        f"[{format_value(float(arr.min()), format_type)}, "
        f"{format_value(float(arr.max()), format_type)}]"
    )
    return cell.replace("$", "")


def _verification_cells(
    df: pd.DataFrame, impl_model: str, reference: pd.Series
) -> dict[str, str]:
    """Per-trial verification quality, cost, and latency, aggregated over trials."""
    impl_df = df[df["impl_model"] == impl_model]
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    accuracy: list[float] = []
    cost: list[float] = []
    latency: list[float] = []
    for trial_id in sorted(impl_df["trial_id"].unique()):
        trial_df = impl_df[impl_df["trial_id"] == int(trial_id)]
        pred = trial_df.set_index("claim")["verification_result"]
        common = reference.index.intersection(pred.index)
        ref_common, pred_common = reference.loc[common], pred.loc[common]

        tp, fp, fn = hallucination_counts(ref_common, pred_common)
        p, r, f = compute_precision_recall_f1(tp, fp, fn)
        precision.append(p)
        recall.append(r)
        f1.append(f)
        accuracy.append(
            (ref_common == pred_common).sum() / len(common) if len(common) else 0
        )
        cost.append(trial_df["total_token_cost"].mean())
        latency.append(trial_df["latency"].mean())

    return {
        "Precision": _mean_range(precision, FormatType.SCORE),
        "Recall": _mean_range(recall, FormatType.SCORE),
        "F1 Score": _mean_range(f1, FormatType.SCORE),
        "Accuracy": _mean_range(accuracy, FormatType.SCORE),
        "Cost (\\$)": _mean_range(cost, FormatType.COST),
        "Latency (s)": _mean_range(latency, FormatType.LATENCY),
    }


def _component_cells(df: pd.DataFrame, impl_model: str) -> dict[str, str]:
    """Per-trial provenance/filter/map quality, micro-averaged across claims.

    Filter and map metrics are de-duplicated by operator id: some claims reuse
    the same semantic operator over the same data (e.g. the four ranking claims
    share one filter and one map), so pooling raw per-claim counts would count
    those operators multiple times. Provenance is a query-level metric (it
    depends on the full plan, including the deterministic operators that differ
    across otherwise-shared claims), so it stays pooled across all claims.
    """
    impl_df = df[df["impl_model"] == impl_model]
    prov: list[float] = []
    filter_p: list[float] = []
    filter_r: list[float] = []
    filter_f1: list[float] = []
    map_acc: list[float] = []
    for trial_id in sorted(impl_df["trial_id"].unique()):
        trial_df = impl_df[impl_df["trial_id"] == int(trial_id)]

        prov_total = trial_df["prov_total"].sum()
        prov.append(
            trial_df["prov_valid_count"].sum() / prov_total if prov_total else 0
        )

        filter_rows = trial_df[trial_df["filter_tp"].notna()].drop_duplicates(
            "filter_op_id"
        )
        tp = int(filter_rows["filter_tp"].sum())
        fp = int(filter_rows["filter_fp"].sum())
        fn = int(filter_rows["filter_fn"].sum())
        p, r, f = compute_precision_recall_f1(tp, fp, fn)
        filter_p.append(p)
        filter_r.append(r)
        filter_f1.append(f)

        map_rows = trial_df[trial_df["map_total"].notna()].drop_duplicates("map_op_id")
        map_total = map_rows["map_total"].sum()
        map_acc.append(map_rows["map_correct"].sum() / map_total if map_total else 0)

    return {
        "Provenance Precision": _mean_range(prov, FormatType.SCORE),
        "Filter Precision": _mean_range(filter_p, FormatType.SCORE),
        "Filter Recall": _mean_range(filter_r, FormatType.SCORE),
        "Filter F1 Score": _mean_range(filter_f1, FormatType.SCORE),
        "Map Accuracy": _mean_range(map_acc, FormatType.SCORE),
    }


def _latex_results_tabular(
    df: pd.DataFrame,
    impls: list[str],
    columns: list[str],
    cell_fn: Callable[[pd.DataFrame, str], dict[str, str]],
    col_spec: str,
) -> str:
    """Render a ``tabular`` body grouping implementations by row with ``\\multirow``."""
    header = " & ".join(["Implementation", "LLM", *columns])
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "  \\toprule",
        f"  {header} \\\\",
        "  \\midrule",
    ]
    for impl_index, impl in enumerate(impls):
        impl_models = get_impl_models(df, impls=[impl])
        if not impl_models:
            continue
        macro = LATEX_IMPL_MACROS[impl]
        lines.append(f"    \\multirow{{{len(impl_models)}}}{{*}}{{{macro}}}")
        for impl_model in impl_models:
            model = impl_model[len(impl) + 1 :]
            cells = cell_fn(df, impl_model)
            values = " & ".join(cells[column] for column in columns)
            lines.append(f"      & {LATEX_MODEL_NAMES[model]} & {values} \\\\")
        if impl_index != len(impls) - 1:
            lines.append("  \\addlinespace")
    lines += ["  \\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def save_verification_results_table(df: pd.DataFrame) -> None:
    """Emit the verification quality/cost/latency tabular to figures/."""
    reference = get_reference(df)
    columns = [
        "Precision",
        "Recall",
        "F1 Score",
        "Accuracy",
        "Cost (\\$)",
        "Latency (s)",
    ]
    tabular = _latex_results_tabular(
        df,
        ["base_rm", "rag_agent", "rlm", "evg_unopt", "evg_opt"],
        columns,
        lambda data, impl_model: _verification_cells(data, impl_model, reference),
        "llcccccc",
    )
    (FIGURES_DIR / "verification_results.tex").write_text(tabular)


def save_component_quality_table(df: pd.DataFrame) -> None:
    """Emit the provenance/filter/map operator-quality tabular to figures/."""
    columns = [
        "Provenance Precision",
        "Filter Precision",
        "Filter Recall",
        "Filter F1 Score",
        "Map Accuracy",
    ]
    tabular = _latex_results_tabular(
        df,
        ["evg_unopt_ref_query", "evg_opt_ref_query"],
        columns,
        _component_cells,
        "llccccc",
    )
    (FIGURES_DIR / "component_quality.tex").write_text(tabular)


def plot_verification_heatmap(df: pd.DataFrame) -> None:
    reference = get_reference(df)

    impl_models = get_impl_models(df)
    rows: list[dict[str, object]] = []
    for impl_model in impl_models:
        impl_df = df[df["impl_model"] == impl_model]
        for _, row_data in impl_df.iterrows():
            claim = row_data["claim"]
            result = row_data["verification_result"]
            rows.append(
                {
                    "claim": claim,
                    "impl_model": impl_model,
                    "correct": float(result == reference.loc[str(claim)]),
                }
            )

    correctness_df = pd.DataFrame(rows)
    grouped = correctness_df.groupby(["claim", "impl_model"])["correct"]
    # Color encodes the fraction correct; annotation shows the integer count of
    # correct trials (out of the number of trials).
    pivot = grouped.mean().unstack()
    counts = grouped.sum().unstack()

    # Keep both matrices aligned through the same reordering/relabeling.
    order = [c for c in impl_models if c in pivot.columns]
    pivot, counts = pivot[order], counts[order]

    claim_ids = [CLAIM_TO_CLAIM_ID[c] for c in pivot.index]
    pivot.index = counts.index = claim_ids
    row_order = sorted(claim_ids, key=lambda c: int(c[1:]))
    pivot, counts = pivot.reindex(row_order), counts.reindex(row_order)

    labels = [IMPL_MODEL_LABELS[c] for c in order]
    pivot.columns = counts.columns = labels

    _, axis = plt.subplots(figsize=(7.0, 2))
    sns.heatmap(
        pivot.T,
        annot=counts.T.astype(int),
        fmt="d",
        ax=axis,
        vmin=0,
        vmax=1,
        cbar=False,
    )  # type: ignore
    axis.set(xlabel="Claim", ylabel="Implementation")  # type: ignore
    save_figure("verification_result")


def plot_metric(
    df: pd.DataFrame,
    column: str,
    ylabel: str,
    filename: str,
    log_scale: bool = False,
    impls: list[str] | None = None,
) -> None:
    """Plot a metric per claim grouped by implementation."""
    impl_models = get_impl_models(df, impls)
    filtered = df[df[column].notna() & df["impl_model"].isin(impl_models)].copy()
    filtered["impl_label"] = filtered["impl_model"].map(IMPL_MODEL_LABELS)
    filtered["claim_id"] = filtered["claim"].map(CLAIM_TO_CLAIM_ID)

    hue_order = [IMPL_MODEL_LABELS[m] for m in impl_models]
    claim_order = sorted(filtered["claim_id"].unique(), key=lambda c: int(c[1:]))

    # Split claims across two stacked rows so each bar has room to breathe.
    half = (len(claim_order) + 1) // 2
    claim_rows = [claim_order[:half], claim_order[half:]]

    _, axes = plt.subplots(
        2, 1, figsize=(FIGURE_SIZE[0] / 2, FIGURE_SIZE[1] * 2), sharey=True
    )
    for i, (axis, claims_subset) in enumerate(zip(axes, claim_rows, strict=True)):
        sns.barplot(
            data=filtered[filtered["claim_id"].isin(claims_subset)],
            x="claim_id",
            y=column,
            hue="impl_label",
            hue_order=hue_order,
            order=claims_subset,
            ax=axis,
            errorbar=("pi", 100),
            err_kws={"alpha": 0.5},
        )
        axis.set(  # type: ignore
            xlabel="claim", ylabel=ylabel, yscale="symlog" if log_scale else "linear"
        )
        if i == 0:
            axis.legend(title="implementation")  # type: ignore
        elif axis.get_legend() is not None:
            axis.get_legend().remove()
    save_figure(filename)


def plot_ablation_quality(df: pd.DataFrame) -> None:
    """Save quality metric changes for all ablation variants."""

    def _quality(impl_df: pd.DataFrame, ref: pd.Series) -> pd.DataFrame:
        rows_q: list[dict[str, float]] = []
        for tid in impl_df["trial_id"].unique():
            trial = impl_df[impl_df["trial_id"] == tid].set_index("claim")[
                "verification_result"
            ]
            common = ref.index.intersection(trial.index)
            if not len(common):
                rows_q.append({"precision": 0.0, "recall": 0.0, "f1": 0.0})
                continue
            r, p = ref.loc[common], trial.loc[common]
            tp, fp, fn = hallucination_counts(r, p)
            precision, recall, f1 = compute_precision_recall_f1(tp, fp, fn)
            rows_q.append({"precision": precision, "recall": recall, "f1": f1})
        return pd.DataFrame(rows_q)

    ablation_df = get_ablation_df(df)
    reference = get_reference(ablation_df)
    baseline_key = f"evg_opt_ref_query_{ABLATION_MODEL}"

    rows: list[dict[str, object]] = []

    for impl, label in _ABLATION_STEP_LABELS.items():
        if impl == "evg_opt":
            continue

        abl_key = f"{impl}_{ABLATION_MODEL}"
        claim_ids = ABLATION_APPLICABLE_CLAIMS[impl]
        claims = {CLAIM_ID_TO_CLAIM[c] for c in claim_ids}
        sub = ablation_df[ablation_df["claim"].isin(claims)]

        base = sub[sub["impl_model"] == baseline_key]
        abl = sub[sub["impl_model"] == abl_key]
        ref = reference[reference.index.isin(claims)]

        base_q = _quality(base, ref)
        abl_q = _quality(abl, ref)

        for metric_col, metric_label in [
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1"),
        ]:
            base_val = base_q[metric_col].mean()
            abl_val = abl_q[metric_col].mean()
            rows.append(
                {
                    "Implementation": label,
                    "Metric": metric_label,
                    "value": f"{base_val:.3f}",
                }
            )
            rows.append(
                {
                    "Implementation": label,
                    "Metric": f"Δ {metric_label}",
                    "value": f"{abl_val - base_val:+.3f}",
                }
            )

    result = pd.DataFrame(rows)
    label_order = [v for k, v in _ABLATION_STEP_LABELS.items() if k != "evg_opt"]
    result["Implementation"] = pd.Categorical(
        result["Implementation"], categories=label_order, ordered=True
    )
    table = result.pivot(
        index="Implementation",
        columns="Metric",
        values="value",
    ).reset_index()
    table.columns.name = None
    claims_map = {
        label: ", ".join(
            sorted(ABLATION_APPLICABLE_CLAIMS[impl], key=lambda c: int(c[1:]))
        )
        for impl, label in _ABLATION_STEP_LABELS.items()
        if impl != "evg_opt"
    }
    table["Claims"] = table["Implementation"].map(claims_map)
    table = table[
        [
            "Implementation",
            "Claims",
            "Precision",
            "Δ Precision",
            "Recall",
            "Δ Recall",
            "F1",
            "Δ F1",
        ]
    ]
    table.to_csv(FIGURES_DIR / "ablation_quality.csv", index=False)


def plot_ablation_cost(df: pd.DataFrame) -> None:
    """Plot per-claim cost multipliers (No X / All Opt) over applicable claims."""
    ablation_df = get_ablation_df(df)
    baseline_key = f"evg_opt_ref_query_{ABLATION_MODEL}"
    steps = [
        (impl, label)
        for impl, label in _ABLATION_STEP_LABELS.items()
        if impl not in ("evg_opt", "evg_abl_no_pc")
    ]

    rng = np.random.default_rng(0)
    _, axis = plt.subplots(figsize=(2.24, 1.2))
    col_name = "total_token_cost"

    for x_pos, (impl, _label) in enumerate(steps):
        abl_key = f"{impl}_{ABLATION_MODEL}"
        finite: list[float] = []

        for cid in ABLATION_APPLICABLE_CLAIMS[impl]:
            claim = CLAIM_ID_TO_CLAIM[cid]
            base = ablation_df[
                (ablation_df["impl_model"] == baseline_key)
                & (ablation_df["claim"] == claim)
            ]
            abl = ablation_df[
                (ablation_df["impl_model"] == abl_key) & (ablation_df["claim"] == claim)
            ]
            if base.empty or abl.empty:
                continue

            # Average across trials on each side, then take the ratio.
            base_mean = base[col_name].mean()
            abl_mean = abl[col_name].mean()
            if base_mean <= 0 or abl_mean <= 0:
                continue
            finite.append(abl_mean / base_mean)

        jitter = rng.uniform(-0.15, 0.15, size=len(finite))
        axis.scatter(
            np.full(len(finite), x_pos) + jitter,
            finite,
            s=4,
            alpha=0.5,
            zorder=3,
        )
        if finite:
            geo_mean = float(np.exp(np.mean(np.log(finite))))
            axis.plot(
                [x_pos - 0.2, x_pos + 0.2],
                [geo_mean, geo_mean],
                color="black",
                linewidth=1,
                zorder=4,
            )
            axis.text(
                x_pos,
                geo_mean * 1.05,
                f"{geo_mean:.1f}",
                ha="center",
                va="bottom",
                zorder=4,
            )

    axis.axhline(y=1.0, color="gray", linestyle="--", linewidth=1)
    axis.set_yscale("log")
    axis.set_xticks(range(len(steps)))
    axis.set_xticklabels([label for _, label in steps])
    axis.set(xlabel="Implementation", ylabel="Cost Multiplier")

    save_figure("ablation_cost")


def load_sim_filter_results() -> pd.DataFrame:
    """Load sim_filter_sensitivity_analysis.json files into a DataFrame."""
    rows: list[dict[str, Any]] = []
    for file_path in RESULTS_DIR.rglob("sim_filter_sensitivity_analysis.json"):
        with open(file_path) as f:
            data = json.load(f)
        metadata = data["metadata"]
        path_parts = file_path.relative_to(RESULTS_DIR).parts
        claim_key = f"{path_parts[1]}\n{path_parts[2]}\n{metadata['name']}"
        claim_id = CLAIM_TO_CLAIM_ID[claim_key]
        for result in data["results"]:
            gt_count = result["ground_truth_selected_row_count"]
            total_count = result["total_row_count"]
            recalls = [t["tp"] / gt_count for t in result["trials"]]
            filter_rates = [
                1 - t["selected_count"] / total_count for t in result["trials"]
            ]
            rows.append(
                {
                    "claim_id": claim_id,
                    "threshold": result["threshold"],
                    "recall_mean": np.mean(recalls),
                    "recall_min": np.min(recalls),
                    "recall_max": np.max(recalls),
                    "filter_rate_mean": np.mean(filter_rates),
                    "filter_rate_min": np.min(filter_rates),
                    "filter_rate_max": np.max(filter_rates),
                }
            )
    return pd.DataFrame(rows)


def plot_sim_filter_sensitivity() -> None:
    """Plot similarity filter recall and filter rate vs threshold."""
    sim_filter_df = load_sim_filter_results()

    claim_order = sorted(sim_filter_df["claim_id"].unique(), key=lambda c: int(c[1:]))

    fig_recall, ax_recall = plt.subplots(figsize=(2.24, 1.2))
    fig_rate, ax_filter_rate = plt.subplots(figsize=(2.24, 1.2))

    for claim_id in claim_order:
        claim_df = sim_filter_df[sim_filter_df["claim_id"] == claim_id].sort_values(
            "threshold"
        )
        thresholds = claim_df["threshold"]

        (line_recall,) = ax_recall.plot(
            thresholds,
            claim_df["recall_mean"],
            marker="o",
            markersize=2,
            label=claim_id,
        )
        ax_recall.fill_between(
            thresholds,
            claim_df["recall_min"],
            claim_df["recall_max"],
            alpha=0.25,
            color=line_recall.get_color(),
        )

        (line_filter_rate,) = ax_filter_rate.plot(
            thresholds,
            claim_df["filter_rate_mean"],
            marker="o",
            markersize=2,
            label=claim_id,
        )
        ax_filter_rate.fill_between(
            thresholds,
            claim_df["filter_rate_min"],
            claim_df["filter_rate_max"],
            alpha=0.25,
            color=line_filter_rate.get_color(),
        )

    for ax in (ax_recall, ax_filter_rate):
        ax.locator_params(axis="both", nbins=6)
        ax.axvline(x=0.15, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Similarity Threshold")

    ax_recall.set_ylabel("Recall")
    ax_recall.legend()
    ax_filter_rate.set_ylabel("Filter Rate")

    for fig, name in ((fig_recall, "sim_filter_recall"), (fig_rate, "sim_filter_rate")):
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"{name}.pdf")
        plt.close(fig)
    sim_filter_df.to_csv(FIGURES_DIR / "sim_filter_sensitivity.csv", index=False)


def plot_ablation_per_claim(df: pd.DataFrame) -> None:
    """
    Save per-claim cost/latency/correctness for each ablation vs evg_opt_ref_query.
    """
    ablation_df = get_ablation_df(df)
    baseline_key = f"evg_opt_ref_query_{ABLATION_MODEL}"
    reference = get_reference(ablation_df)

    rows: list[dict[str, object]] = []
    for impl, label in _ABLATION_STEP_LABELS.items():
        if impl == "evg_opt":
            continue
        abl_key = f"{impl}_{ABLATION_MODEL}"
        claim_ids = ABLATION_APPLICABLE_CLAIMS[impl]
        for cid in sorted(claim_ids, key=lambda c: int(c[1:])):
            claim = CLAIM_ID_TO_CLAIM[cid]
            base = ablation_df[
                (ablation_df["impl_model"] == baseline_key)
                & (ablation_df["claim"] == claim)
            ]
            abl = ablation_df[
                (ablation_df["impl_model"] == abl_key) & (ablation_df["claim"] == claim)
            ]
            ref_val = reference[claim]
            opt_correct = (base["verification_result"] == ref_val).sum()
            abl_correct = (abl["verification_result"] == ref_val).sum()
            n_trials = len(base)

            for metric, col, decimals in [
                ("Cost", "total_token_cost", 3),
                ("Latency", "latency", 1),
            ]:
                rows.append(
                    {
                        "Implementation": label,
                        "Claim": cid,
                        "Metric": metric,
                        "OPT Mean": round(base[col].mean(), decimals),
                        "OPT Min": round(base[col].min(), decimals),
                        "OPT Max": round(base[col].max(), decimals),
                        "ABL Mean": round(abl[col].mean(), decimals),
                        "ABL Min": round(abl[col].min(), decimals),
                        "ABL Max": round(abl[col].max(), decimals),
                    }
                )
            rows.append(
                {
                    "Implementation": label,
                    "Claim": cid,
                    "Metric": "Correctness",
                    "OPT Mean": f"{opt_correct}/{n_trials}",
                    "ABL Mean": f"{abl_correct}/{n_trials}",
                }
            )

    pd.DataFrame(rows).to_csv(FIGURES_DIR / "ablation_per_claim.csv", index=False)


def plot_cost_f1_pareto(df: pd.DataFrame) -> None:
    """Scatter plot of mean F1 vs mean cost per claim, with Pareto frontier."""
    reference = get_reference(df)

    # Aggregate F1 and cost per implementation–model pair across trials.
    records: list[dict[str, Any]] = []
    for impl_model in get_impl_models(df):
        impl_df = df[df["impl_model"] == impl_model]
        impl = impl_df["implementation"].iloc[0]
        model = impl_df["model"].iloc[0]

        trial_f1s: list[float] = []
        trial_costs: list[float] = []
        for trial_id in impl_df["trial_id"].unique():
            trial_df = impl_df[impl_df["trial_id"] == trial_id]
            trial_claims = trial_df.set_index("claim")["verification_result"]
            common = reference.index.intersection(trial_claims.index)
            ref, pred = reference.loc[common], trial_claims.loc[common]
            tp, fp, fn = hallucination_counts(ref, pred)
            _, _, f1 = compute_precision_recall_f1(tp, fp, fn)
            trial_f1s.append(f1)
            trial_costs.append(trial_df["total_token_cost"].mean())

        records.append(
            {
                "implementation": impl,
                "model": model,
                "f1": float(np.mean(trial_f1s)),
                "cost": float(np.mean(trial_costs)),
            }
        )

    points = pd.DataFrame(records)

    # Pareto frontier: point i is dominated if some j has f1 >= f1[i] and
    # cost <= cost[i] with at least one strict inequality.
    costs = points["cost"].values
    f1s = points["f1"].values
    is_pareto = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        for j in range(len(points)):
            if (
                i != j
                and f1s[j] >= f1s[i]
                and costs[j] <= costs[i]
                and (f1s[j] > f1s[i] or costs[j] < costs[i])
            ):
                is_pareto[i] = False
                break

    _, ax = plt.subplots(figsize=(3.33, 1.8))

    markers = ["o", "s", "p", "D", "^"]
    colors = ["C0", "C1", "C4", "C3", "C2"]  # make evg_opt green

    for impl, marker, color in zip(IMPLEMENTATIONS, markers, colors, strict=True):
        subset = points[points["implementation"] == impl]
        if subset.empty:
            continue
        ax.scatter(
            subset["cost"],
            subset["f1"],
            label=impl,
            marker=marker,
            s=10,
            color=color,
        )

    for _, row in points.iterrows():
        ax.annotate(
            _MODEL_LABELS[row["model"]],
            (row["cost"], row["f1"]),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
        )

    pareto_points = points[is_pareto].sort_values("cost")
    ax.plot(
        pareto_points["cost"],
        pareto_points["f1"],
        color="gray",
        linestyle="--",
        linewidth=1,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Cost ($)")
    ax.set_ylabel("F1 Score")
    ax.set_xlim(0.06, 25)
    ax.set_ylim(0.69, 1)
    ax.legend()

    save_figure("cost_f1_pareto")


def save_compilation_summary() -> None:
    comp = load_compilation_costs()
    comp_df = pd.DataFrame(
        [
            {"claim": claim, "trial_id": trial_id, **vals}
            for (claim, trial_id), vals in comp.items()
        ]
    )
    per_claim = comp_df.groupby("claim")[["total_token_cost", "latency"]].mean()

    summary = pd.DataFrame(
        [
            {
                "Metric": label,
                "Mean": per_claim[col].mean(),
                "Std": per_claim[col].std(),  # pandas std uses ddof=1
                "N": int(per_claim[col].count()),
            }
            for label, col in [
                ("Cost ($)", "total_token_cost"),
                ("Latency (s)", "latency"),
            ]
        ]
    )
    summary.to_csv(FIGURES_DIR / "compilation_summary.csv", index=False)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    sns.set_palette("colorblind")
    # Paper figures are authored at their exact on-page width (scale 1 in LaTeX),
    # so these absolute point sizes render identically across every figure.
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.size": 6,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
        }
    )

    df = load_results()
    print(f"Loaded {len(df)} results")
    print(df[["claim", "impl_model", "verification_result"]].to_string())

    print("\nGround Truth:")
    print(
        df[df["implementation"] == "evg_ref"][
            ["claim", "verification_result"]
        ].to_string(index=False)
    )

    # Compilation metrics
    save_compilation_summary()

    # Result tables (LaTeX tabular bodies)
    save_verification_results_table(df)
    save_component_quality_table(df)

    # Summary plots (aggregated across claims)
    plot_cost_f1_pareto(df)

    # Ablations
    plot_ablation_quality(df)
    plot_ablation_cost(df)
    plot_ablation_per_claim(df)

    # Sensitivity analysis
    plot_sim_filter_sensitivity()

    # Per-claim plots
    plot_verification_heatmap(df)
    plot_metric(
        df,
        column="total_token_cost",
        ylabel="total token cost ($)",
        filename="total_token_cost",
        log_scale=True,
    )
    plot_metric(
        df,
        column="latency",
        ylabel="latency (s)",
        filename="latency",
        log_scale=True,
    )

    print(f"\nFigures saved to {FIGURES_DIR}")

    if PAPER_DIR.is_dir():
        PAPER_FIGURES_DIR.mkdir(exist_ok=True)
        paper_assets = {
            "verification_result.pdf",
            "ablation_cost.pdf",
            "sim_filter_recall.pdf",
            "sim_filter_rate.pdf",
            "cost_f1_pareto.pdf",
            "verification_results.tex",
            "component_quality.tex",
        }
        for asset in FIGURES_DIR.iterdir():
            if asset.name in paper_assets:
                shutil.copy2(asset, PAPER_FIGURES_DIR / asset.name)
        print(f"Figures copied to {PAPER_FIGURES_DIR}")


if __name__ == "__main__":
    main()
