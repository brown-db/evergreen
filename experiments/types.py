from dataclasses import dataclass

from evergreen.data_frame import FilterMetrics, MapMetrics, QueryMetrics
from evergreen.provenance import Token


@dataclass(frozen=True)
class EvaluationResult:
    verification_result: bool
    query_metrics: QueryMetrics
    filter_metrics: FilterMetrics | None
    map_metrics: MapMetrics | None
    prov_tokens: dict[Token, bool]
    reasoning: str | list[dict[str, object]] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "verification_result": self.verification_result,
            "query_metrics": self.query_metrics.to_dict(),
            "filter_metrics": self.filter_metrics.to_dict()
            if self.filter_metrics
            else None,
            "map_metrics": self.map_metrics.to_dict() if self.map_metrics else None,
            "prov_tokens": [
                {
                    "row_id": token.row_id,
                    "predicate": str(token.predicate),
                    "sign": token.sign,
                    "is_valid": is_valid,
                }
                for token, is_valid in self.prov_tokens.items()
            ],
            "reasoning": self.reasoning,
        }
