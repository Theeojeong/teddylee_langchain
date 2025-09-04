#!/usr/bin/env bash
set -euo pipefail

echo "[1/5] RAG 준비 (정제/인덱스 생성)"
python -m ai_agent.rag.prepare

echo "[2/5] KMMLU 로드/배치 입력 생성"
python -m ai_agent.eval.build_batch

echo "[3/5] OpenAI Batch 제출"
python -m ai_agent.eval.submit_batch --submit

echo "[4/5] Batch 완료 대기 및 결과 수집"
python -m ai_agent.eval.submit_batch --wait

echo "[5/5] 결과 채점"
python -m ai_agent.eval.score

echo "완료: outputs 디렉토리를 확인하세요."
