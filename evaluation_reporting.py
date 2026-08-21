# -*- coding: utf-8 -*-
"""Reporting models and writers for the Foundry RAG evaluation pipeline."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from pydantic import BaseModel, Field


class MetricResult(BaseModel):
    name: str
    status: str
    score: float | None = None
    label: str | None = None
    passed: bool | None = None
    threshold: float | None = None
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class QuestionResult(BaseModel):
    case_id: str
    question_number: int
    query: str
    metrics: dict[str, MetricResult]
    hard_gate_passed: bool
    applicable_metrics: int
    passed_metrics: int
    failed_metrics: int
    missing_metrics: int
    not_applicable_metrics: int

    @classmethod
    def finalize(
        cls,
        *,
        case_id: str,
        question_number: int,
        query: str,
        metrics: Mapping[str, MetricResult],
        hard_gate_metrics: set[str],
    ) -> "QuestionResult":
        completed = [m for m in metrics.values() if m.status == "completed"]
        passed = [m for m in completed if m.passed is True]
        failed = [m for m in completed if m.passed is False]
        missing = [m for m in metrics.values() if m.status == "missing"]
        na = [m for m in metrics.values() if m.status == "not_applicable"]

        hard_gate_passed = True
        for name in hard_gate_metrics:
            metric = metrics.get(name)
            # Production hard gate: missing/error is not silently treated as a pass.
            if metric is None or metric.status != "completed" or metric.passed is not True:
                hard_gate_passed = False
                break

        return cls(
            case_id=case_id,
            question_number=question_number,
            query=query,
            metrics=dict(metrics),
            hard_gate_passed=hard_gate_passed,
            applicable_metrics=len(completed),
            passed_metrics=len(passed),
            failed_metrics=len(failed),
            missing_metrics=len(missing),
            not_applicable_metrics=len(na),
        )


def aggregate_results(results: list[QuestionResult]) -> dict[str, Any]:
    if not results:
        return {
            "question_count": 0,
            "hard_gate_pass_rate": 0.0,
            "metrics": {},
        }

    metric_bucket: dict[str, list[MetricResult]] = defaultdict(list)
    for question in results:
        for name, metric in question.metrics.items():
            metric_bucket[name].append(metric)

    summary: dict[str, Any] = {}
    for name, values in sorted(metric_bucket.items()):
        completed = [v for v in values if v.status == "completed"]
        numeric_scores = [v.score for v in completed if v.score is not None]
        pass_count = sum(v.passed is True for v in completed)
        fail_count = sum(v.passed is False for v in completed)
        summary[name] = {
            "completed": len(completed),
            "missing": sum(v.status == "missing" for v in values),
            "not_applicable": sum(v.status == "not_applicable" for v in values),
            "passed": pass_count,
            "failed": fail_count,
            "pass_rate": (pass_count / len(completed)) if completed else None,
            "mean_score": (
                sum(numeric_scores) / len(numeric_scores)
                if numeric_scores
                else None
            ),
        }

    hard_passed = sum(q.hard_gate_passed for q in results)
    return {
        "question_count": len(results),
        "hard_gate_passed": hard_passed,
        "hard_gate_failed": len(results) - hard_passed,
        "hard_gate_pass_rate": hard_passed / len(results),
        "metrics": summary,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _autosize(ws: Any, max_width: int = 60) -> None:
    for column_cells in ws.columns:
        letter = get_column_letter(column_cells[0].column)
        width = min(
            max(
                len(str(cell.value)) if cell.value is not None else 0
                for cell in column_cells
            )
            + 2,
            max_width,
        )
        ws.column_dimensions[letter].width = max(width, 10)


def write_excel(
    path: Path,
    results: list[QuestionResult],
    overall: Mapping[str, Any],
) -> None:
    """
    Create an auditable workbook.
    Missing/NA stay textual; they are never rendered as misleading zeroes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()

    # ---- Question Results ----
    ws = wb.active
    ws.title = "Question Results"

    metric_names = sorted(
        {name for result in results for name in result.metrics}
    )
    headers = [
        "Question #",
        "Case ID",
        "Question",
        *metric_names,
        "Passed / Applicable",
        "Missing",
        "N/A",
        "Hard Gate",
    ]
    ws.append(headers)

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for result in results:
        row: list[Any] = [
            result.question_number,
            result.case_id,
            result.query,
        ]
        for metric_name in metric_names:
            metric = result.metrics.get(metric_name)
            if metric is None:
                row.append("")
            elif metric.status == "not_applicable":
                row.append("N/A")
            elif metric.status == "missing":
                row.append("MISSING")
            elif metric.score is not None:
                row.append(metric.score)
            elif metric.label:
                row.append(metric.label.upper())
            else:
                row.append("COMPLETED")

        row.extend(
            [
                f"{result.passed_metrics}/{result.applicable_metrics}",
                result.missing_metrics,
                result.not_applicable_metrics,
                "PASS" if result.hard_gate_passed else "FAIL",
            ]
        )
        ws.append(row)

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for row in ws.iter_rows(min_row=2):
        row[2].alignment = Alignment(wrap_text=True, vertical="top")

    # Highlight hard-gate status using neutral fills.
    hard_gate_col = len(headers)
    pass_fill = PatternFill("solid", fgColor="C6EFCE")
    fail_fill = PatternFill("solid", fgColor="FFC7CE")
    ws.conditional_formatting.add(
        f"{get_column_letter(hard_gate_col)}2:{get_column_letter(hard_gate_col)}{ws.max_row}",
        CellIsRule(operator="equal", formula=['"PASS"'], fill=pass_fill),
    )
    ws.conditional_formatting.add(
        f"{get_column_letter(hard_gate_col)}2:{get_column_letter(hard_gate_col)}{ws.max_row}",
        CellIsRule(operator="equal", formula=['"FAIL"'], fill=fail_fill),
    )
    _autosize(ws)

    # ---- Metric Summary ----
    summary_ws = wb.create_sheet("Metric Summary")
    summary_headers = [
        "Metric",
        "Completed",
        "Passed",
        "Failed",
        "Pass Rate",
        "Mean Score",
        "Missing",
        "Not Applicable",
    ]
    summary_ws.append(summary_headers)
    for cell in summary_ws[1]:
        cell.fill = header_fill
        cell.font = header_font

    for name, metric in overall.get("metrics", {}).items():
        summary_ws.append(
            [
                name,
                metric.get("completed"),
                metric.get("passed"),
                metric.get("failed"),
                metric.get("pass_rate"),
                metric.get("mean_score"),
                metric.get("missing"),
                metric.get("not_applicable"),
            ]
        )

    summary_ws.append([])
    summary_ws.append(["Overall Hard Gate Pass Rate", overall.get("hard_gate_pass_rate")])
    summary_ws.append(["Questions", overall.get("question_count")])
    summary_ws.append(["Hard Gate Passed", overall.get("hard_gate_passed")])
    summary_ws.append(["Hard Gate Failed", overall.get("hard_gate_failed")])

    for row in summary_ws.iter_rows(min_row=2):
        if isinstance(row[4].value, float):
            row[4].number_format = "0.0%"
    _autosize(summary_ws)

    # ---- Metric Details ----
    detail_ws = wb.create_sheet("Metric Details")
    detail_ws.append(
        [
            "Question #",
            "Case ID",
            "Metric",
            "Status",
            "Score",
            "Threshold",
            "Label",
            "Passed",
            "Reason",
        ]
    )
    for cell in detail_ws[1]:
        cell.fill = header_fill
        cell.font = header_font

    for result in results:
        for name, metric in sorted(result.metrics.items()):
            detail_ws.append(
                [
                    result.question_number,
                    result.case_id,
                    name,
                    metric.status,
                    metric.score,
                    metric.threshold,
                    metric.label,
                    metric.passed,
                    metric.reason,
                ]
            )
    for row in detail_ws.iter_rows(min_row=2):
        row[8].alignment = Alignment(wrap_text=True, vertical="top")
    detail_ws.freeze_panes = "A2"
    detail_ws.auto_filter.ref = detail_ws.dimensions
    _autosize(detail_ws)

    wb.save(path)
