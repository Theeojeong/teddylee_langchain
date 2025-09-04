from __future__ import annotations

from .rag.prepare import build_index
from .eval.build_batch import main as build_batch
from .eval.submit_batch import submit, wait_and_fetch, load_out_dir
from .config import OUTPUTS_DIR


def main():
    print("[1/5] RAG 준비 (정제/인덱스 생성)")
    build_index()

    print("[2/5] KMMLU 로드/배치 입력 생성")
    build_batch()

    print("[3/5] OpenAI Batch 제출")
    out_dir = load_out_dir(None)
    bid = submit(out_dir)

    print("[4/5] Batch 완료 대기 및 결과 수집")
    wait_and_fetch(out_dir, batch_id=bid)

    print("[5/5] 결과 채점")
    from .eval.score import main as score_main

    score_main()


if __name__ == "__main__":
    main()

