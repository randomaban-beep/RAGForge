import csv
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path
from src.models import Document

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download_scifact(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "scifact.zip"
    if not archive.exists():
        with urllib.request.urlopen(SCIFACT_URL, timeout=120) as response:
            archive.write_bytes(response.read())
    # Only extract the three expected data files; never arbitrary archive paths.
    with zipfile.ZipFile(archive) as zipped:
        for name in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
            dest = root / "scifact" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zipped.read("scifact/" + name))
    return root / "scifact"


def load_beir(root):
    root = Path(root)
    paths = [root / "corpus.jsonl", root / "queries.jsonl", root / "qrels/test.tsv"]
    docs = []
    for line in paths[0].read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        docs.append(
            Document(
                str(row["_id"]),
                (row.get("title", "") + "\n" + row["text"]).strip(),
                {"source": "BEIR/SciFact", "title": row.get("title", "")},
            )
        )
    queries = {
        str(row["_id"]): row["text"]
        for row in (json.loads(line) for line in paths[1].read_text(encoding="utf-8").splitlines())
    }
    qrels = {}
    with paths[2].open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    doc_ids = {d.id for d in docs}
    if len(doc_ids) != len(docs):
        raise ValueError("Duplicate corpus IDs.")
    cases = []
    for qid, relevance in sorted(qrels.items()):
        if not set(relevance) <= doc_ids or qid not in queries:
            raise ValueError("Judgment refers to a missing document or query.")
        if any(score > 0 for score in relevance.values()):
            cases.append({"id": qid, "question": queries[qid], "relevance": relevance})
    return docs, cases, {str(p.relative_to(root)): sha256(p) for p in paths}


def load_fixture(path="data/demo.json"):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    docs = [Document(**row) for row in data["documents"]]
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return docs, data["cases"], {path.name: fingerprint}
