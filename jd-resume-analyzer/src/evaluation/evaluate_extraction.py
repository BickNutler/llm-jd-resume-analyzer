from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from rich import print

from ..modeling.llm_extract import extract_with_llm
from ..utils.text import normalize_token

FIELDS = ["skills","tools","requirements"]

def _set(x: List[str]) -> set[str]:
    return set(normalize_token(i) for i in (x or []) if normalize_token(i))

def prf(gold: set[str], pred: set[str]) -> Tuple[float,float,float]:
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall) else 0.0
    return precision, recall, f1

def evaluate_file(path: Path) -> Dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    text = obj.get("job_text") or ""
    gold_obj = obj.get("gold") or {}
    gold = {f: _set(gold_obj.get(f, [])) for f in FIELDS}

    pred_struct = extract_with_llm(text)
    if pred_struct is None:
        raise RuntimeError("LLM extraction disabled. Set LLM_PROVIDER in .env to run evaluation.")

    pred = {
        "skills": _set(pred_struct.skills),
        "tools": _set(pred_struct.tools),
        "requirements": _set(pred_struct.requirements),
    }

    per={}
    for f in FIELDS:
        p,r,f1 = prf(gold[f], pred[f])
        per[f]={"precision":p,"recall":r,"f1":f1,"gold_n":len(gold[f]),"pred_n":len(pred[f])}
    return {"id": obj.get("id", path.stem), "per_field": per}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled_dir", type=str, required=True)
    args = ap.parse_args()

    labeled_dir = Path(args.labeled_dir)
    files = sorted(labeled_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No .json files found in {labeled_dir}")

    results = [evaluate_file(f) for f in files]

    # micro-average across all samples
    totals = {f: {"tp":0,"fp":0,"fn":0} for f in FIELDS}
    for fpath in files:
        obj = json.loads(fpath.read_text(encoding="utf-8"))
        text = obj.get("job_text") or ""
        gold_obj = obj.get("gold") or {}
        gold = {f: _set(gold_obj.get(f, [])) for f in FIELDS}
        pred_struct = extract_with_llm(text)
        pred = {
            "skills": _set(pred_struct.skills),
            "tools": _set(pred_struct.tools),
            "requirements": _set(pred_struct.requirements),
        }
        for fld in FIELDS:
            g, pset = gold[fld], pred[fld]
            totals[fld]["tp"] += len(g & pset)
            totals[fld]["fp"] += len(pset - g)
            totals[fld]["fn"] += len(g - pset)

    print("[bold]\nPer-sample results[/bold]")
    for r in results:
        print(f"- {r['id']}: {r['per_field']}")

    print("[bold]\nMicro-average[/bold]")
    for fld in FIELDS:
        tp, fp, fn = totals[fld]["tp"], totals[fld]["fp"], totals[fld]["fn"]
        precision = tp/(tp+fp) if (tp+fp) else 0.0
        recall = tp/(tp+fn) if (tp+fn) else 0.0
        f1 = (2*precision*recall)/(precision+recall) if (precision+recall) else 0.0
        print(f"{fld}: precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} (tp={tp} fp={fp} fn={fn})")

if __name__ == "__main__":
    main()
