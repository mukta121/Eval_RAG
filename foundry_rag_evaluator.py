# -*- coding: utf-8 -*-
"""
Production-grade RAG evaluation runner for Microsoft Foundry cloud evaluations.

Design goals
------------
1. Evaluate precomputed RAG responses in Microsoft Foundry.
2. Keep deterministic retrieval metrics local (chunk-ID based).
3. Use Foundry built-in evaluators for RAG quality and safety.
4. Preserve raw cloud output for debugging/auditability.
5. Treat NOT_APPLICABLE / MISSING / ERROR separately from FAIL.
6. Use a stable case_id rather than matching cloud results by question text.
7. Keep evaluation resources by default for audit/reproducibility.
8. Support JSON, JSONL, and common wrapper-object inputs.

Expected ground-truth row (minimum)
-----------------------------------
{
  "query": "...",
  "ground_truth_answer": "...",
  "relevant_contexts": [
    {"chunk_id": "doc_chunk_001", "content": "..."}
  ]
}

Expected response row (minimum)
-------------------------------
{
  "query": "...",
  "generated_answer": "...",
  "retrieved_contexts": [
    {"chunk_id": "doc_chunk_001", "content": "...", "score": 0.91}
  ]
}

Environment variables
---------------------
FOUNDRY_PROJECT_ENDPOINT=https://<account>.services.ai.azure.com/api/projects/<project>
FOUNDRY_MODEL_DEPLOYMENT=<judge model deployment name>

Optional
--------
FOUNDRY_EVAL_POLL_SECONDS=5
FOUNDRY_EVAL_TIMEOUT_SECONDS=3600
FOUNDRY_DATASET_NAME=rag-eval-dataset
LOG_LEVEL=INFO

CLI example
-----------
python foundry_rag_evaluator.py \
    --ground-truth ground_truth.json \
    --responses response_generator.json \
    --output evaluation_foundry_result.json \
    --excel-output evaluation_foundry_result.xlsx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import TestingCriterionAzureAIEvaluator
from azure.identity import DefaultAzureCredential
from openai.types.eval_create_params import DataSourceConfigCustom
from openai.types.evals.create_eval_jsonl_run_data_source_param import (
    CreateEvalJSONLRunDataSourceParam,
    SourceFileID,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from evaluation_reporting import (
    MetricResult,
    QuestionResult,
    aggregate_results,
    write_excel,
    write_json,
)

LOGGER = logging.getLogger("foundry_rag_evaluator")
JsonObject = dict[str, Any]

TERMINAL_STATUSES = {"completed", "failed", "cancelled", "canceled"}
QUALITY_SCALE_METRICS = {
    "retrieval",
    "groundedness",
    "relevance",
    "response_completeness",
}
SAFETY_METRICS = {
    "hate_unfairness",
    "ungrounded_attributes",
    "indirect_attack",
}

# Only metrics selected here are sent to Foundry.
# Context recall/precision/hit@k/MRR are local deterministic metrics.
DEFAULT_FOUNDRY_METRICS = (
    "retrieval",
    "groundedness",
    "relevance",
    "response_completeness",
    "hate_unfairness",
    "ungrounded_attributes",
)

# XPIA is intentionally opt-in. It is meaningful for adversarial/injected-context
# cases, not as a universal hard gate for every benign RAG question.
OPTIONAL_FOUNDRY_METRICS = ("indirect_attack",)


class Settings(BaseSettings):
    """Runtime settings loaded from environment/.env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    foundry_project_endpoint: str = Field(alias="FOUNDRY_PROJECT_ENDPOINT")
    foundry_model_deployment: str = Field(alias="FOUNDRY_MODEL_DEPLOYMENT")

    foundry_eval_poll_seconds: int = Field(
        default=5, alias="FOUNDRY_EVAL_POLL_SECONDS", ge=1, le=60
    )
    foundry_eval_timeout_seconds: int = Field(
        default=3600, alias="FOUNDRY_EVAL_TIMEOUT_SECONDS", ge=60, le=21600
    )
    foundry_dataset_name: str = Field(
        default="rag-eval-dataset", alias="FOUNDRY_DATASET_NAME"
    )


class ContextItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    chunk_id: str | None = None
    content: str = ""
    score: float | None = None
    document_id: str | None = None
    source_file: str | None = None
    page_number: int | None = None


class GroundTruthRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    query: str
    ground_truth_answer: str = ""
    normalized_ground_truth_answer: str = ""
    relevant_contexts: list[ContextItem] = Field(default_factory=list)


class ResponseRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    query: str
    generated_answer: str = ""
    retrieved_contexts: list[ContextItem] = Field(default_factory=list)


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    question_number: int
    query: str
    response: str
    ground_truth: str
    context: str
    expected_chunk_ids: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]


def _normalise_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _stable_case_id(query: str) -> str:
    digest = hashlib.sha256(_normalise_text(query).encode("utf-8")).hexdigest()[:16]
    return f"q_{digest}"


def _read_json_or_jsonl(path: Path) -> list[JsonObject]:
    """Read JSON array/object wrapper or JSONL with useful validation."""
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        records: list[JsonObject] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid JSONL at line {line_no}: {exc}"
                ) from exc
            if not isinstance(item, dict):
                raise TypeError(f"{path}: JSONL line {line_no} must be an object.")
            records.append(item)
        return records

    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        for key in (
            "records",
            "items",
            "data",
            "questions",
            "results",
            "responses",
            "question_results",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        return [payload]

    raise TypeError(f"{path}: input must be JSON object, array, or JSONL.")


def _parse_ground_truth(raw: Mapping[str, Any]) -> GroundTruthRecord:
    canonical = dict(raw)
    canonical["query"] = _normalise_text(
        raw.get("query") or raw.get("question") or raw.get("prompt")
    )
    canonical["ground_truth_answer"] = _normalise_text(
        raw.get("ground_truth_answer")
        or raw.get("answer")
        or raw.get("expected_answer")
        or raw.get("ground_truth")
    )
    canonical["normalized_ground_truth_answer"] = _normalise_text(
        raw.get("normalized_ground_truth_answer")
    )
    canonical["relevant_contexts"] = (
        raw.get("relevant_contexts")
        or raw.get("ground_truth_contexts")
        or raw.get("contexts")
        or []
    )
    return GroundTruthRecord.model_validate(canonical)


def _parse_response(raw: Mapping[str, Any]) -> ResponseRecord:
    canonical = dict(raw)
    canonical["query"] = _normalise_text(
        raw.get("query") or raw.get("question") or raw.get("prompt")
    )
    canonical["generated_answer"] = _normalise_text(
        raw.get("generated_answer")
        or raw.get("response")
        or raw.get("answer")
        or raw.get("output")
    )
    canonical["retrieved_contexts"] = (
        raw.get("retrieved_contexts")
        or raw.get("contexts")
        or raw.get("retrieval_results")
        or []
    )
    return ResponseRecord.model_validate(canonical)


def _index_unique_by_query(
    records: Iterable[Mapping[str, Any]], parser: Any, source_name: str
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for idx, raw in enumerate(records, start=1):
        try:
            parsed = parser(raw)
        except ValidationError as exc:
            raise ValueError(f"{source_name} row {idx}: {exc}") from exc

        query = _normalise_text(parsed.query)
        if not query:
            raise ValueError(f"{source_name} row {idx}: query is empty.")
        if query in output:
            raise ValueError(
                f"{source_name}: duplicate query detected. "
                "Use unique queries or add an explicit case-id mapping."
            )
        output[query] = parsed
    return output


def _dedupe_nonempty(values: Iterable[str | None]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(v.strip() for v in values if v and v.strip()))


def _context_text(contexts: Sequence[ContextItem]) -> str:
    """
    Preserve chunk boundaries and IDs. This is superior to blind newline joining:
    the evaluator sees distinct retrieved units and debugging remains traceable.
    """
    blocks: list[str] = []
    for rank, item in enumerate(contexts, start=1):
        if not item.content.strip():
            continue
        metadata = [f"rank={rank}"]
        if item.chunk_id:
            metadata.append(f"chunk_id={item.chunk_id}")
        if item.document_id:
            metadata.append(f"document_id={item.document_id}")
        if item.page_number is not None:
            metadata.append(f"page={item.page_number}")
        blocks.append(
            f"[RETRIEVED_CONTEXT {' '.join(metadata)}]\n{item.content.strip()}"
        )
    return "\n\n".join(blocks)


def build_cases(
    ground_truth_records: list[JsonObject],
    response_records: list[JsonObject],
    limit: int | None = None,
) -> tuple[list[EvalCase], list[str]]:
    gt = _index_unique_by_query(
        ground_truth_records, _parse_ground_truth, "ground truth"
    )
    responses = _index_unique_by_query(
        response_records, _parse_response, "responses"
    )

    missing_responses = [query for query in gt if query not in responses]
    selected_queries = list(gt)
    if limit is not None:
        selected_queries = selected_queries[:limit]

    cases: list[EvalCase] = []
    for number, query in enumerate(selected_queries, start=1):
        response = responses.get(query)
        if response is None:
            continue
        truth = gt[query]
        ground_truth_answer = (
            truth.normalized_ground_truth_answer or truth.ground_truth_answer
        )
        cases.append(
            EvalCase(
                case_id=_stable_case_id(query),
                question_number=number,
                query=query,
                response=response.generated_answer,
                ground_truth=ground_truth_answer,
                context=_context_text(response.retrieved_contexts),
                expected_chunk_ids=_dedupe_nonempty(
                    ctx.chunk_id for ctx in truth.relevant_contexts
                ),
                retrieved_chunk_ids=_dedupe_nonempty(
                    ctx.chunk_id for ctx in response.retrieved_contexts
                ),
            )
        )
    return cases, missing_responses


def _local_retrieval_metrics(case: EvalCase) -> dict[str, MetricResult]:
    expected = list(case.expected_chunk_ids)
    retrieved = list(case.retrieved_chunk_ids)
    expected_set = set(expected)
    retrieved_set = set(retrieved)

    if not expected:
        na = MetricResult(
            name="context_recall",
            status="not_applicable",
            reason="No authoritative chunk IDs were supplied in ground truth.",
        )
        return {
            "context_recall": na,
            "context_precision": MetricResult(
                name="context_precision",
                status="not_applicable",
                reason="No authoritative chunk IDs were supplied in ground truth.",
            ),
            "hit_at_k": MetricResult(
                name="hit_at_k",
                status="not_applicable",
                reason="No authoritative chunk IDs were supplied in ground truth.",
            ),
            "mrr": MetricResult(
                name="mrr",
                status="not_applicable",
                reason="No authoritative chunk IDs were supplied in ground truth.",
            ),
        }

    hits = expected_set & retrieved_set
    recall = len(hits) / len(expected_set)
    precision = len(hits) / len(retrieved_set) if retrieved_set else 0.0
    hit_at_k = 1.0 if hits else 0.0

    reciprocal_rank = 0.0
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in expected_set:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "context_recall": MetricResult(
            name="context_recall",
            status="completed",
            score=recall,
            label="pass" if recall >= 0.5 else "fail",
            passed=recall >= 0.5,
            threshold=0.5,
            reason=f"Retrieved {len(hits)} of {len(expected_set)} authoritative chunks.",
            details={
                "expected": expected,
                "retrieved": retrieved,
                "hits": sorted(hits),
            },
        ),
        "context_precision": MetricResult(
            name="context_precision",
            status="completed",
            score=precision,
            label="pass" if precision >= 0.5 else "fail",
            passed=precision >= 0.5,
            threshold=0.5,
            reason=f"{len(hits)} of {len(retrieved_set)} retrieved chunks were authoritative.",
        ),
        "hit_at_k": MetricResult(
            name="hit_at_k",
            status="completed",
            score=hit_at_k,
            label="pass" if hit_at_k == 1 else "fail",
            passed=hit_at_k == 1,
            threshold=1.0,
            reason="At least one authoritative chunk was retrieved."
            if hit_at_k
            else "No authoritative chunk was retrieved.",
        ),
        "mrr": MetricResult(
            name="mrr",
            status="completed",
            score=reciprocal_rank,
            label="pass" if reciprocal_rank > 0 else "fail",
            passed=reciprocal_rank > 0,
            threshold=0.0,
            reason=(
                f"First authoritative chunk rank gives reciprocal rank {reciprocal_rank:.3f}."
                if reciprocal_rank
                else "No authoritative chunk occurred in retrieved results."
            ),
        ),
    }


def _criterion(
    *,
    name: str,
    evaluator_name: str,
    mapping: Mapping[str, str],
    deployment: str | None = None,
    threshold: float | int | None = None,
) -> TestingCriterionAzureAIEvaluator:
    init: dict[str, Any] = {}
    if deployment:
        # Current Foundry docs use deployment_name for LLM-judge evaluators.
        init["deployment_name"] = deployment
    if threshold is not None:
        init["threshold"] = threshold

    kwargs: dict[str, Any] = {
        "type": "azure_ai_evaluator",
        "name": name,
        "evaluator_name": evaluator_name,
        "data_mapping": dict(mapping),
    }
    if init:
        kwargs["initialization_parameters"] = init
    return TestingCriterionAzureAIEvaluator(**kwargs)


def build_testing_criteria(
    deployment: str,
    selected_metrics: Sequence[str],
) -> list[TestingCriterionAzureAIEvaluator]:
    """Create only evaluators that are meaningful for precomputed RAG responses."""
    available: dict[str, TestingCriterionAzureAIEvaluator] = {
        "retrieval": _criterion(
            name="retrieval",
            evaluator_name="builtin.retrieval",
            deployment=deployment,
            mapping={
                "query": "{{item.query}}",
                "context": "{{item.context}}",
            },
        ),
        "groundedness": _criterion(
            name="groundedness",
            evaluator_name="builtin.groundedness",
            deployment=deployment,
            mapping={
                "query": "{{item.query}}",
                "response": "{{item.response}}",
                "context": "{{item.context}}",
            },
        ),
        "relevance": _criterion(
            name="relevance",
            evaluator_name="builtin.relevance",
            deployment=deployment,
            mapping={
                "query": "{{item.query}}",
                "response": "{{item.response}}",
            },
        ),
        "response_completeness": _criterion(
            name="response_completeness",
            evaluator_name="builtin.response_completeness",
            deployment=deployment,
            mapping={
                "response": "{{item.response}}",
                "ground_truth": "{{item.ground_truth}}",
            },
        ),
        "hate_unfairness": _criterion(
            name="hate_unfairness",
            evaluator_name="builtin.hate_unfairness",
            mapping={
                "query": "{{item.query}}",
                "response": "{{item.response}}",
            },
        ),
        "ungrounded_attributes": _criterion(
            name="ungrounded_attributes",
            evaluator_name="builtin.ungrounded_attributes",
            mapping={
                "query": "{{item.query}}",
                "response": "{{item.response}}",
                "context": "{{item.context}}",
            },
        ),
        "indirect_attack": _criterion(
            name="indirect_attack",
            evaluator_name="builtin.indirect_attack",
            mapping={
                "query": "{{item.query}}",
                "response": "{{item.response}}",
            },
        ),
    }

    unknown = sorted(set(selected_metrics) - set(available))
    if unknown:
        raise ValueError(f"Unsupported Foundry metric(s): {', '.join(unknown)}")
    return [available[name] for name in selected_metrics]


def build_data_source_config() -> DataSourceConfigCustom:
    return DataSourceConfigCustom(
        type="custom",
        item_schema={
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "question_number": {"type": "integer"},
                "query": {"type": "string"},
                "response": {"type": "string"},
                "ground_truth": {"type": "string"},
                "context": {"type": "string"},
            },
            "required": [
                "case_id",
                "question_number",
                "query",
                "response",
                "ground_truth",
                "context",
            ],
        },
        # Precomputed response evaluation does not use a target-generated sample.
        include_sample_schema=False,
    )


def _case_to_cloud_item(case: EvalCase) -> JsonObject:
    return {
        "case_id": case.case_id,
        "question_number": case.question_number,
        "query": case.query,
        "response": case.response,
        "ground_truth": case.ground_truth,
        "context": case.context,
    }


def _write_cloud_jsonl(cases: Sequence[EvalCase], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(json.dumps(_case_to_cloud_item(case), ensure_ascii=False))
            handle.write("\n")


def _dump_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if isinstance(value, Mapping):
        return {str(k): _dump_model(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump_model(v) for v in value]
    return value


def _walk_dicts(node: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(node, Mapping):
        yield node
        for value in node.values():
            yield from _walk_dicts(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_dicts(value)


def _find_case_id(node: Any) -> str | None:
    for obj in _walk_dicts(node):
        value = obj.get("case_id")
        if isinstance(value, str) and value:
            return value
        item = obj.get("item")
        if isinstance(item, Mapping):
            value = item.get("case_id")
            if isinstance(value, str) and value:
                return value
    return None


def _normalise_metric_name(value: str) -> str:
    text = value.strip().lower()
    text = text.replace("builtin.", "")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    aliases = {
        "hate_and_unfairness": "hate_unfairness",
        "indirect_attack_xpia": "indirect_attack",
        "quality_grader": "quality_grader",
    }
    return aliases.get(text, text)


def _extract_metric_objects(
    cloud_item: Any,
    expected_metric_names: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    """
    Parse evaluator results by their semantic fields rather than relying on one
    SDK nesting layout. Foundry's documented evaluator output includes fields
    such as name, metric, score, label, reason, threshold, and passed.
    """
    expected = {_normalise_metric_name(x) for x in expected_metric_names}
    candidates: dict[str, Mapping[str, Any]] = {}

    for obj in _walk_dicts(cloud_item):
        # A metric payload normally has at least an identity and one result field.
        identities = [
            obj.get("metric"),
            obj.get("name"),
            obj.get("evaluator_name"),
        ]
        resultish = any(
            key in obj for key in ("score", "label", "passed", "reason", "threshold")
        )
        if not resultish:
            continue

        for identity in identities:
            if not isinstance(identity, str):
                continue
            normal = _normalise_metric_name(identity)
            if normal in expected:
                # Prefer the smallest payload already containing the actual score.
                previous = candidates.get(normal)
                if previous is None or (
                    "score" not in previous and "score" in obj
                ):
                    candidates[normal] = obj
                break

    return candidates


def _metric_from_cloud(
    name: str,
    payload: Mapping[str, Any] | None,
) -> MetricResult:
    if payload is None:
        return MetricResult(
            name=name,
            status="missing",
            reason="Metric was not found in the Foundry output item.",
        )

    score = payload.get("score")
    threshold = payload.get("threshold")
    passed = payload.get("passed")
    label = payload.get("label")
    reason = payload.get("reason")

    # Some SDK shapes may nest evaluator output one level deeper.
    if score is None:
        for key in (f"{name}_score", "value"):
            if key in payload:
                score = payload.get(key)
                break

    if isinstance(passed, str):
        passed = passed.strip().lower() == "true"
    if passed is None and isinstance(label, str):
        if label.lower() == "pass":
            passed = True
        elif label.lower() == "fail":
            passed = False

    return MetricResult(
        name=name,
        status="completed",
        score=float(score) if isinstance(score, (int, float)) else None,
        label=str(label) if label is not None else None,
        passed=bool(passed) if isinstance(passed, bool) else None,
        threshold=float(threshold)
        if isinstance(threshold, (int, float))
        else None,
        reason=str(reason) if reason is not None else "",
        details=dict(payload),
    )


def normalize_cloud_outputs(
    output_items: Sequence[Any],
    cases: Sequence[EvalCase],
    foundry_metrics: Sequence[str],
) -> tuple[dict[str, dict[str, MetricResult]], list[Any]]:
    """
    Return metrics keyed by case_id and raw serializable output.

    Missing cloud rows are retained explicitly; they are never silently converted
    into zeroes.
    """
    by_case = {case.case_id: case for case in cases}
    metrics_by_case: dict[str, dict[str, MetricResult]] = {}
    raw_dump: list[Any] = []

    for raw in output_items:
        data = _dump_model(raw)
        raw_dump.append(data)
        case_id = _find_case_id(data)

        if not case_id or case_id not in by_case:
            LOGGER.warning(
                "Unable to correlate one Foundry output item to a known case_id."
            )
            continue

        payloads = _extract_metric_objects(data, foundry_metrics)
        metrics_by_case[case_id] = {
            metric: _metric_from_cloud(metric, payloads.get(metric))
            for metric in foundry_metrics
        }

    for case in cases:
        if case.case_id not in metrics_by_case:
            metrics_by_case[case.case_id] = {
                metric: MetricResult(
                    name=metric,
                    status="missing",
                    reason="No Foundry output item was returned/correlated for this case.",
                )
                for metric in foundry_metrics
            }

    return metrics_by_case, raw_dump


def _poll_run(
    client: Any,
    eval_id: str,
    run_id: str,
    settings: Settings,
) -> Any:
    started = time.monotonic()
    last_status: str | None = None

    while True:
        run = client.evals.runs.retrieve(run_id=run_id, eval_id=eval_id)
        status = str(run.status).lower()

        if status != last_status:
            LOGGER.info("Foundry evaluation status: %s", status)
            last_status = status

        if status in TERMINAL_STATUSES:
            return run

        if time.monotonic() - started > settings.foundry_eval_timeout_seconds:
            raise TimeoutError(
                f"Foundry evaluation timed out after "
                f"{settings.foundry_eval_timeout_seconds}s."
            )
        time.sleep(settings.foundry_eval_poll_seconds)


def _question_results(
    cases: Sequence[EvalCase],
    cloud_metrics: Mapping[str, Mapping[str, MetricResult]],
    foundry_metrics: Sequence[str],
) -> list[QuestionResult]:
    results: list[QuestionResult] = []

    for case in cases:
        metrics = {}
        metrics.update(_local_retrieval_metrics(case))
        metrics.update(cloud_metrics.get(case.case_id, {}))

        results.append(
            QuestionResult.finalize(
                case_id=case.case_id,
                question_number=case.question_number,
                query=case.query,
                metrics=metrics,
                # Hard gates: system quality + local context recall.
                # Safety metrics remain visible but should not be forced into a
                # universal "x/10" denominator when not applicable.
                hard_gate_metrics={
                    "context_recall",
                    "groundedness",
                    "relevance",
                    "response_completeness",
                },
            )
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production RAG evaluation using Microsoft Foundry cloud evaluators."
    )
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_foundry_result.json"),
    )
    parser.add_argument("--excel-output", type=Path)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_FOUNDRY_METRICS),
        choices=list(DEFAULT_FOUNDRY_METRICS + OPTIONAL_FOUNDRY_METRICS),
        help="Foundry metrics to run. indirect_attack is opt-in.",
    )
    parser.add_argument(
        "--delete-evaluation",
        action="store_true",
        help="Delete the Foundry evaluation after the run. Default is to KEEP it.",
    )
    parser.add_argument(
        "--fail-on-missing-cloud-metric",
        action="store_true",
        help="Return non-zero if any requested cloud metric is missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    try:
        settings = Settings()
    except ValidationError as exc:
        LOGGER.error("Invalid environment configuration: %s", exc)
        return 2

    ground_truth_records = _read_json_or_jsonl(args.ground_truth)
    response_records = _read_json_or_jsonl(args.responses)
    cases, missing_queries = build_cases(
        ground_truth_records,
        response_records,
        limit=args.limit,
    )

    if not cases:
        LOGGER.error("No matched evaluation cases were found.")
        return 2
    if missing_queries:
        LOGGER.warning("%d ground-truth queries have no response.", len(missing_queries))

    eval_id: str | None = None
    run_id: str | None = None
    raw_output_items: list[Any] = []
    credential = DefaultAzureCredential()

    try:
        with AIProjectClient(
            endpoint=settings.foundry_project_endpoint,
            credential=credential,
        ) as project:
            with project.get_openai_client() as client:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

                # Production-friendly: upload a reusable JSONL dataset instead of
                # embedding all rows directly into one eval run request.
                with tempfile.TemporaryDirectory(prefix="foundry-rag-eval-") as tmp:
                    dataset_path = Path(tmp) / "evaluation_items.jsonl"
                    _write_cloud_jsonl(cases, dataset_path)

                    dataset = project.datasets.upload_file(
                        name=f"{settings.foundry_dataset_name}-{timestamp}",
                        version="1",
                        file_path=str(dataset_path),
                    )

                evaluation = client.evals.create(
                    name=f"RAG evaluation {timestamp}",
                    data_source_config=build_data_source_config(),
                    testing_criteria=build_testing_criteria(
                        settings.foundry_model_deployment,
                        args.metrics,
                    ),
                )
                eval_id = evaluation.id
                LOGGER.info("Created evaluation %s", eval_id)

                run = client.evals.runs.create(
                    eval_id=eval_id,
                    name=f"RAG evaluation run {timestamp}",
                    metadata={
                        "scenario": "rag-precomputed-responses",
                        "schema_version": "2",
                    },
                    data_source=CreateEvalJSONLRunDataSourceParam(
                        type="jsonl",
                        source=SourceFileID(
                            type="file_id",
                            id=dataset.id,
                        ),
                    ),
                )
                run_id = run.id
                LOGGER.info("Created evaluation run %s", run_id)

                final_run = _poll_run(client, eval_id, run_id, settings)

                if str(final_run.status).lower() != "completed":
                    LOGGER.error(
                        "Foundry evaluation ended with status %s",
                        final_run.status,
                    )

                output_items = list(
                    client.evals.runs.output_items.list(
                        run_id=run_id,
                        eval_id=eval_id,
                    )
                )
                cloud_metrics, raw_output_items = normalize_cloud_outputs(
                    output_items,
                    cases,
                    args.metrics,
                )

                question_results = _question_results(
                    cases,
                    cloud_metrics,
                    args.metrics,
                )
                overall = aggregate_results(question_results)

                report = {
                    "schema_version": "2.0",
                    "evaluation_run_id": str(uuid.uuid4()),
                    "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "evaluator_mode": "microsoft_foundry_cloud",
                    "foundry": {
                        "project_endpoint": settings.foundry_project_endpoint,
                        "model_deployment": settings.foundry_model_deployment,
                        "evaluation_id": eval_id,
                        "run_id": run_id,
                        "run_status": str(final_run.status),
                        "report_url": getattr(final_run, "report_url", None),
                        "dataset_id": dataset.id,
                    },
                    "question_count": len(cases),
                    "missing_queries": missing_queries,
                    "foundry_metrics_requested": list(args.metrics),
                    "local_metrics": [
                        "context_recall",
                        "context_precision",
                        "hit_at_k",
                        "mrr",
                    ],
                    "question_results": [q.model_dump(mode="json") for q in question_results],
                    "overall": overall,
                }

                write_json(args.output, report)
                if args.raw_output:
                    write_json(
                        args.raw_output,
                        {
                            "eval_id": eval_id,
                            "run_id": run_id,
                            "output_items": raw_output_items,
                        },
                    )
                if args.excel_output:
                    write_excel(args.excel_output, question_results, overall)

                missing_metric_count = sum(
                    1
                    for question in question_results
                    for metric in question.metrics.values()
                    if metric.status == "missing"
                )

                LOGGER.info(
                    "Completed %d cases. Hard-gate pass rate: %.1f%%",
                    len(question_results),
                    100.0 * overall["hard_gate_pass_rate"],
                )
                LOGGER.info("JSON report: %s", args.output)
                if args.excel_output:
                    LOGGER.info("Excel report: %s", args.excel_output)

                if args.fail_on_missing_cloud_metric and missing_metric_count:
                    return 3
                if str(final_run.status).lower() != "completed":
                    return 4
                return 0

    except Exception:
        LOGGER.exception("Evaluation failed.")
        return 1
    finally:
        # Keep cloud evaluations by default. Production evaluation artifacts are
        # valuable for traceability, comparisons, and root-cause analysis.
        if args.delete_evaluation and eval_id:
            try:
                with AIProjectClient(
                    endpoint=settings.foundry_project_endpoint,
                    credential=credential,
                ) as project:
                    with project.get_openai_client() as client:
                        client.evals.delete(eval_id=eval_id)
                        LOGGER.info("Deleted Foundry evaluation %s", eval_id)
            except Exception:
                LOGGER.exception("Failed to delete evaluation %s", eval_id)
        credential.close()


if __name__ == "__main__":
    sys.exit(main())
