"""검색 단계만 따로 평가한다: Recall@k 와 MRR.

    python evaluate.py            # 기본 설정
    python evaluate.py --k 3      # k를 줄이면 어디서 무너지는지 보인다
    python evaluate.py --no-links # 링크 확장의 기여도 확인

생성 품질을 손대기 전에 검색부터 재는 이유는 vault/10-개념/RAG 품질 평가.md 참고.
평가셋(eval_questions.json)은 실제 사용자 질문 로그로 계속 채워 넣는 것이 가장 좋다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retriever import Retriever
from vault_index import load_index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", type=Path, default=Path(__file__).parent.parent / "vault")
    ap.add_argument("--questions", type=Path, default=Path(__file__).parent / "eval_questions.json")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--no-links", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="실패 사례를 출력하지 않음")
    args = ap.parse_args()

    cases = json.loads(args.questions.read_text(encoding="utf-8"))
    retriever = Retriever(load_index(args.vault))

    hit_count = 0
    top1 = 0
    rr_sum = 0.0
    failures = []

    for case in cases:
        hits = retriever.search(case["q"], k=args.k, expand_links=not args.no_links)
        paths = [h.chunk.note_path for h in hits]
        if case["expect"] in paths:
            rank = paths.index(case["expect"]) + 1
            hit_count += 1
            rr_sum += 1 / rank
            top1 += rank == 1
        else:
            failures.append((case["q"], case["expect"], paths[:3]))

    n = len(cases)
    print(f"질문 {n}개 · k={args.k} · 링크확장={'off' if args.no_links else 'on'}")
    print(f"  Recall@{args.k} : {hit_count / n:.2f}  ({hit_count}/{n})")
    print(f"  Top-1 정확도  : {top1 / n:.2f}")
    print(f"  MRR           : {rr_sum / n:.3f}")

    if failures and not args.quiet:
        print(f"\n실패 {len(failures)}건:")
        for q, expect, got in failures:
            print(f"  - {q}\n      기대: {expect}\n      실제: {got}")


if __name__ == "__main__":
    main()
