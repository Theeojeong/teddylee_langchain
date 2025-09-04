from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

from ..config import (
    OUTPUTS_DIR,
    BATCH_COMPLETION_WINDOW,
    BATCH_ENDPOINT,
    CHAT_MODEL,
)


LATEST_FILE = OUTPUTS_DIR / "latest_run.txt"


def load_out_dir(arg_dir: Optional[str]) -> Path:
    if arg_dir:
        return Path(arg_dir)
    if not LATEST_FILE.exists():
        raise SystemExit("latest_run.txt가 없어 출력 디렉토리를 찾을 수 없습니다.")
    return Path(LATEST_FILE.read_text(encoding="utf-8").strip())


def submit(out_dir: Path) -> str:
    client = OpenAI()
    input_path = out_dir / "batch_input.jsonl"
    if not input_path.exists():
        raise SystemExit(f"입력 파일을 찾을 수 없습니다: {input_path}")

    up = client.files.create(file=input_path.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=up.id,
        endpoint=BATCH_ENDPOINT,
        completion_window=BATCH_COMPLETION_WINDOW,
        metadata={"chat_model": CHAT_MODEL},
    )

    run_meta_path = out_dir / "run_meta.json"
    meta = {
        "batch_id": batch.id,
        "input_file_id": up.id,
        "endpoint": BATCH_ENDPOINT,
        "completion_window": BATCH_COMPLETION_WINDOW,
        "chat_model": CHAT_MODEL,
        "status": batch.status,
    }
    run_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Submitted batch:", batch.id)
    return batch.id


def wait_and_fetch(out_dir: Path, batch_id: Optional[str] = None):
    client = OpenAI()
    run_meta_path = out_dir / "run_meta.json"
    if batch_id is None:
        if not run_meta_path.exists():
            raise SystemExit("run_meta.json이 없어 batch_id를 확인할 수 없습니다.")
        meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        batch_id = meta["batch_id"]

    print("Waiting for batch to complete:", batch_id)
    while True:
        b = client.batches.retrieve(batch_id)
        print("status:", b.status)
        if b.status in {"completed", "failed", "cancelled", "expired"}:
            break
        time.sleep(15)

    # Update meta
    meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
    meta["status"] = b.status
    if getattr(b, "output_file_id", None):
        meta["output_file_id"] = b.output_file_id
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    if b.status != "completed":
        print("Batch not completed. Status:", b.status)
        return

    # Download output JSONL
    file_id = b.output_file_id
    print("Downloading output file:", file_id)
    content = client.files.content(file_id)
    out_path = out_dir / "batch_output.jsonl"
    # content is a stream in 1.x; write bytes
    with out_path.open("wb") as f:
        f.write(content.read())
    print("Saved:", out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--submit", action="store_true", help="배치 제출")
    p.add_argument("--wait", action="store_true", help="완료 대기 및 결과 다운로드")
    p.add_argument("--fetch", action="store_true", help="특정 배치 ID로 결과 조회/다운로드")
    p.add_argument("--batch-id", type=str, default=None, help="--fetch 시 배치 ID")
    p.add_argument("--out-dir", type=str, default=None, help="출력 디렉토리 경로")
    args = p.parse_args()

    out_dir = load_out_dir(args.out_dir)

    if args.submit:
        submit(out_dir)
    elif args.wait:
        wait_and_fetch(out_dir)
    elif args.fetch:
        if not args.batch_id:
            raise SystemExit("--fetch 사용 시 --batch-id 필요")
        wait_and_fetch(out_dir, batch_id=args.batch_id)
    else:
        p.print_help()


if __name__ == "__main__":
    main()

