import json
import subprocess
import sys
from src.evaluation.datasets import load_fixture
from src.evaluation.run import evaluate
from src.retrieval.hybrid import SearchIndex


def test_fixture_references_exist():
    docs, cases, _ = load_fixture()
    ids = {d.id for d in docs}
    assert len(ids) == len(docs)
    assert len({c["id"] for c in cases}) == len(cases)
    assert all(set(c["relevance"]) <= ids for c in cases)


def test_actual_retrieval_clears_smoke_gate():
    docs, cases, _ = load_fixture()
    result = evaluate(SearchIndex(docs), [c for c in cases if c["relevance"]], ["tfidf", "bm25"])
    assert all(r["summary"]["recall@5"] >= 0.8 for r in result.values())


def test_cli_quality_gate_returns_nonzero_for_failure(tmp_path):
    # k=1 forces imperfect rankings on the development fixture.
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluation.run",
            "--k",
            "1",
            "--min-recall",
            "1",
            "--out",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    report = json.loads((tmp_path / "retrieval.json").read_text())
    failed = any(r["summary"]["recall@1"] < 1 for r in report["results"].values())
    assert failed
    assert run.returncode != 0


def test_fixture_fingerprint_ignores_line_endings_but_detects_data_changes(tmp_path):
    from pathlib import Path

    data = json.loads(Path("data/demo.json").read_text(encoding="utf-8"))
    fixture = tmp_path / "demo.json"
    fixture.write_bytes(json.dumps(data, indent=2).replace("\n", "\r\n").encode())
    first = load_fixture(fixture)[2]
    fixture.write_bytes(json.dumps(data, separators=(",", ":")).encode())
    assert load_fixture(fixture)[2] == first
    data["cases"][0]["question"] = "Changed question"
    fixture.write_text(json.dumps(data), encoding="utf-8")
    assert load_fixture(fixture)[2] != first
