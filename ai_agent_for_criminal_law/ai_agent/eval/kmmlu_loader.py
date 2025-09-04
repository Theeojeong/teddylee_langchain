from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datasets import load_dataset
import pandas as pd

from ..config import KMMLU_DATASET_NAME, KMMLU_CATEGORY, KMMLU_SPLIT


@dataclass
class KMMLUItem:
    idx: int
    question: str
    choices: List[str]
    answer: str  # One of A/B/C/D
    meta: Dict[str, Any]


def _standardize_row(row: Dict[str, Any], idx: int) -> Optional[KMMLUItem]:
    # Try to extract subject/category
    subject = row.get("subject") or row.get("category") or row.get("topic")

    # Extract question
    question = row.get("question") or row.get("prompt") or row.get("q")
    if not question:
        return None

    # Extract choices in various possible formats
    choices: List[str] = []
    if isinstance(row.get("choices"), list):
        choices = row["choices"]
    elif isinstance(row.get("options"), list):
        choices = row["options"]
    else:
        # Sometimes flattened: A/B/C/D fields
        tmp = []
        for k in ["A", "B", "C", "D", "a", "b", "c", "d"]:
            if k in row:
                tmp.append(str(row[k]))
        if tmp:
            choices = tmp

    # Normalize choices
    if not choices or len(choices) < 2:
        return None

    # Extract answer (letter or index)
    ans = row.get("answer") or row.get("label") or row.get("correct")
    if ans is None:
        return None

    # Convert to letter A-D
    if isinstance(ans, (int, float)):
        ans_idx = int(ans)
        if ans_idx < 0 or ans_idx > 3:
            return None
        answer_letter = ["A", "B", "C", "D"][ans_idx]
    else:
        s = str(ans).strip().upper()
        if s in {"A", "B", "C", "D"}:
            answer_letter = s
        else:
            # Some datasets store the text of the correct choice
            # Try to map by exact match
            try:
                idx_match = [c.strip() for c in choices].index(s)
                answer_letter = ["A", "B", "C", "D"][idx_match]
            except Exception:
                # Could not resolve
                return None

    return KMMLUItem(
        idx=idx,
        question=str(question),
        choices=[str(c) for c in choices[:4]],
        answer=answer_letter,
        meta={"subject": subject},
    )


def load_kmmlu_criminal_law() -> List[KMMLUItem]:
    ds = load_dataset(KMMLU_DATASET_NAME, split=KMMLU_SPLIT)

    # Filter to Criminal-Law category via best-effort matching on known fields
    def _match_cat(x: Dict[str, Any]) -> bool:
        target = KMMLU_CATEGORY.lower()
        for key in ["subject", "category", "topic", "subset"]:
            v = x.get(key)
            if isinstance(v, str) and target in v.lower().replace(" ", "-"):
                return True
        return False

    try:
        ds = ds.filter(_match_cat)
    except Exception:
        # If filter fails (no such column), continue and hope the dataset is a single-subject config
        pass

    items: List[KMMLUItem] = []
    for i, row in enumerate(ds):
        it = _standardize_row(row, i)
        if it is not None:
            items.append(it)

    if not items:
        # As a fallback, try pandas-based normalization to inspect columns
        df = ds.to_pandas()
        for i, row in df.iterrows():
            it = _standardize_row(row.to_dict(), int(i))
            if it is not None:
                items.append(it)

    if not items:
        raise RuntimeError(
            "KMMLU Criminal-Law 항목을 파싱하지 못했습니다. 데이터셋 스키마를 확인해주세요."
        )

    return items

