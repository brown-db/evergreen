from enum import Enum
from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import (
    Expr,
    bool_or,
    col,
    prompt,
    proportion,
)
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)


class Satisfaction(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"
    NEUTRAL = "neutral"


class ExistentialProportionalClaim1Evaluator(ClaimEvaluator):
    def query(
        self,
        df: DataFrame,
        impl: Implementation,
        language_models: tuple[str, ...],
        trial_id: int,
    ) -> DataFrame:
        return (
            df.map(
                prompt(
                    "Based on the customer support {dialog}, identify the customer's "
                    "satisfaction level: "
                    "positive (customer seems satisfied/happy with the resolution), "
                    "negative (customer seems dissatisfied/frustrated), "
                    "mixed (customer expresses both satisfaction and dissatisfaction), "
                    "or neutral (no clear satisfaction signal)",
                    Satisfaction,
                ).alias("satisfaction")
            )
            .log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.POST_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .aggregate(
                [
                    proportion(
                        col("satisfaction").eq(Satisfaction.POSITIVE)
                        | col("satisfaction").eq(Satisfaction.MIXED)
                    ).alias("positive_or_mixed_prop"),
                    proportion(
                        col("satisfaction").eq(Satisfaction.NEGATIVE)
                        | col("satisfaction").eq(Satisfaction.MIXED)
                    ).alias("negative_or_mixed_prop"),
                ],
                group_by=[col("company_id")],
            )
            .aggregate(
                [
                    bool_or(
                        (col("positive_or_mixed_prop") >= 0.2)
                        & (col("negative_or_mixed_prop") >= 0.2)
                    ).alias("some_mixed_bag")
                ]
            )
            .check(col("some_mixed_bag"))
        )

    def check_predicate(self) -> Expr:
        return col("some_mixed_bag")

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("satisfaction"),)

    def hints(self) -> str:
        return (
            "Classify each customer support dialog's satisfaction as "
            "positive (customer seems satisfied/happy with the resolution), "
            "negative (customer seems dissatisfied/frustrated), "
            "mixed (customer expresses both satisfaction and dissatisfaction), "
            "or neutral (no clear satisfaction signal). "
            "An airline's satisfaction is considered a 'mixed bag' if at least 20% of "
            "its dialogs are positive or mixed AND at least 20% are negative or mixed."
        )


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = ExistentialProportionalClaim1Evaluator(
        claim_compilation_result_path=Path(
            "experiments/results/claim_compiler/twitter_customer_support/airlines/compare/existential_proportional_claim_1_2026-06-03_12-02-11.json"
        ),
        dataset_key=("dialog_id",),
        text_field_name="dialog",
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
