from __future__ import annotations

from typing import Dict, Any, List
from pathlib import Path
import json

from ..config import CHAT_MODEL, OUTPUTS_DIR, timestamp_dir
from ..agent.prompt import SYSTEM_PROMPT, build_user_prompt
from ..rag.prepare import retrieve, load_index
from .kmmlu_loader import load_kmmlu_criminal_law


LATEST_FILE = OUTPUTS_DIR / "latest_run.txt"


def main():
    # Ensure index exists
    load_index()

    items = load_kmmlu_criminal_law()
    out_dir = timestamp_dir("kmmlu_criminal_law")
    (out_dir / "_ok").touch()

    # Save dataset meta (idx, answer)
    dataset_meta = [
        {
            "idx": it.idx,
            "answer": it.answer,
            "question": it.question,
            "choices": it.choices,
            "meta": it.meta,
        }
        for it in items
    ]
    (out_dir / "dataset_meta.json").write_text(
        json.dumps(dataset_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows: List[Dict[str, Any]] = []
    for it in items:
        contexts = retrieve(it.question, k=3)
        user_prompt = build_user_prompt(it.question, it.choices, contexts)
        custom_id = f"kmmlu-{it.idx}"
        rows.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": CHAT_MODEL,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            }
        )

    input_path = out_dir / "batch_input.jsonl"
    with input_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save latest
    LATEST_FILE.write_text(str(out_dir), encoding="utf-8")
    print("Saved batch input:", input_path)


if __name__ == "__main__":
    main()

