from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
    / "scripts"
    / "eval"
    / "summarize_navqa_official_style_results.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "summarize_navqa_official_style_results", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summary_uses_only_navqa_official_style_metric() -> None:
    module = _load_module()

    summary = module.summarize_payload(
        {
            "run_name": "run",
            "run_metadata": {
                "route_id": "CG6-clean+IR-QA-F-GER-v3-binding",
                "cg_id": "CG6-clean",
                "ir_qa_id": "IR-QA-F-GER-v3-binding",
            },
            "rows": [
                {
                    "new": {
                        "correct": 1,
                        "total": 1,
                        "accuracy_percent": 100.0,
                        "metric": "navqa_official_style_accuracy",
                        "by_type": {
                            "text": {
                                "correct": 1,
                                "total": 1,
                                "accuracy_percent": 100.0,
                            }
                        },
                    }
                },
                {
                    "new": {
                        "correct": 1,
                        "total": 1,
                        "accuracy_percent": 100.0,
                        "metric": "navqa_official_style_accuracy",
                        "by_type": {
                            "binary": {
                                "correct": 1,
                                "total": 1,
                                "accuracy_percent": 100.0,
                            }
                        },
                    }
                },
                {
                    "new": {
                        "correct": 0,
                        "total": 1,
                        "accuracy_percent": 0.0,
                        "metric": "legacy_metric",
                        "by_type": {
                            "text": {
                                "correct": 0,
                                "total": 1,
                                "accuracy_percent": 0.0,
                            }
                        },
                    }
                },
            ],
        }
    )

    assert summary["metric"] == "navqa_official_style_accuracy"
    assert summary["route_id"] == "CG6-clean+IR-QA-F-GER-v3-binding"
    assert summary["overall"] == {
        "correct": 2,
        "total": 2,
        "accuracy_percent": 100.0,
    }
    assert summary["by_type"]["text"] == {
        "correct": 1,
        "total": 1,
        "accuracy_percent": 100.0,
    }
    assert summary["by_type"]["binary"] == {
        "correct": 1,
        "total": 1,
        "accuracy_percent": 100.0,
    }
