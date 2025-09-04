from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict

from ..config import OUTPUTS_DIR


LATEST_FILE = OUTPUTS_DIR / "latest_run.txt"


def _load_out_dir() -> Path:
    return Path(LATEST_FILE.read_text(encoding="utf-8").strip())


def _extract_answer(text: str) -> str:
    # Find first A/B/C/D (uppercase)
    m = re.search(r"\b([ABCD])\b", text.strip().upper())
    if m:
        return m.group(1)
    # Fallback: first char
    ch = text.strip().upper()[:1]
    if ch in {"A", "B", "C", "D"}:
        return ch
    return ""


def main():
    out_dir = _load_out_dir()
    meta_path = out_dir / "dataset_meta.json"
    out_path = out_dir / "batch_output.jsonl"
    if not meta_path.exists():
        raise SystemExit("dataset_meta.json 미존재")
    if not out_path.exists():
        raise SystemExit("batch_output.jsonl 미존재 (배치 완료 여부 확인)")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    gold: Dict[str, str] = {f"kmmlu-{m['idx']}": m["answer"] for m in meta}

    preds: Dict[str, str] = {}
    ok = 0
    total = 0

    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id")
            body = (
                (obj.get("response") or {}).get("body")
                if isinstance(obj.get("response"), dict)
                else None
            )
            if not cid or not body:
                continue
            # chat.completions format
            try:
                content = body["choices"][0]["message"]["content"]
            except Exception:
                content = ""
            ans = _extract_answer(str(content))
            if ans:
                preds[cid] = ans

    for k, g in gold.items():
        total += 1
        p = preds.get(k, "")
        if p == g:
            ok += 1

    acc = ok / total if total else 0.0
    print(f"KMMLU Criminal-Law Accuracy: {acc:.4f} ({ok}/{total})")

    metrics = {"accuracy": acc, "correct": ok, "total": total}
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

