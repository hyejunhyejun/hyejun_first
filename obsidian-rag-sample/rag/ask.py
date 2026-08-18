"""볼트에 질문하고 근거와 함께 답을 받는다 (검색 → 프롬프트 조립 → Claude 생성).

    python ask.py "청크는 얼마나 크게 잘라야 하나요"
    python ask.py "약어 검색이 왜 실패했나" --show-context
    python ask.py "..." --dry-run       # API 호출 없이 조립된 프롬프트만 출력

--dry-run 은 API 키 없이도 동작한다. RAG를 디버깅할 때는 답변이 아니라
'프롬프트에 무엇이 들어갔는가'를 먼저 봐야 한다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from retriever import Hit, Retriever
from vault_index import load_index

MODEL = "claude-opus-5"
STALE_DAYS = 365

# 시스템 프롬프트는 요청마다 바뀌지 않는다 → 프롬프트 캐시 접두부에 둔다.
# (실제 캐시 적중은 접두부가 약 1024토큰을 넘을 때부터 발생한다.
#  vault/10-개념/프롬프트 캐싱.md 참고)
SYSTEM_PROMPT = """당신은 팀의 사내 위키 어시스턴트입니다. 아래 규칙을 지킵니다.

1. 답변은 제공된 <노트 발췌> 안의 내용만 근거로 삼습니다. 발췌에 없는 사실은 쓰지 않습니다.
2. 모든 주장 문장 끝에 근거 번호를 [1] 형식으로 붙입니다. 여러 개면 [1][3] 처럼 씁니다.
3. 발췌만으로 답할 수 없으면 추측하지 말고 "볼트에서 근거를 찾지 못했습니다"라고 말한 뒤,
   어떤 노트를 새로 쓰면 되는지 한 줄로 제안합니다.
4. 발췌에 (오래됨) 표시가 있는 노트를 근거로 쓸 때는 정보가 오래되었을 수 있다고 덧붙입니다.
5. 답변은 한국어로, 결론을 먼저 쓰고 필요한 만큼만 설명합니다. 발췌를 그대로 길게 옮기지 않습니다."""


def is_stale(updated: str) -> bool:
    try:
        age = dt.date.today() - dt.date.fromisoformat(updated)
    except ValueError:
        return False
    return age.days > STALE_DAYS


def build_context(hits: list[Hit]) -> str:
    """검색 결과를 번호가 붙은 근거 블록으로 만든다. 번호가 곧 인용 키가 된다."""
    blocks = []
    for n, hit in enumerate(hits, 1):
        c = hit.chunk
        meta = [c.note_path]
        if c.heading_path:
            meta.append(f"§ {c.heading_path}")
        if c.updated:
            meta.append(f"updated {c.updated}" + (" (오래됨)" if is_stale(c.updated) else ""))
        blocks.append(f"[{n}] {' · '.join(meta)}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def build_user_message(question: str, hits: list[Hit]) -> str:
    return (
        "<노트 발췌>\n"
        f"{build_context(hits)}\n"
        "</노트 발췌>\n\n"
        f"질문: {question}"
    )


def print_sources(hits: list[Hit]) -> None:
    print("\n\n출처")
    for n, hit in enumerate(hits, 1):
        mark = " · 링크 확장" if hit.via_link else ""
        stale = " · 오래됨" if is_stale(hit.chunk.updated) else ""
        print(f"  [{n}] {hit.chunk.citation()}{stale}{mark}")


def answer(question: str, hits: list[Hit], model: str, use_fallbacks: bool) -> None:
    try:
        import anthropic
    except ImportError:
        sys.exit("anthropic SDK가 없습니다. `pip install -r requirements.txt` 후 다시 실행하세요.\n"
                 "API 키 없이 프롬프트만 보려면 --dry-run 을 쓰세요.")

    client = anthropic.Anthropic()   # ANTHROPIC_API_KEY 등 환경에서 자격증명을 읽는다
    kwargs = dict(
        model=model,
        max_tokens=8000,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},   # 고정 접두부만 캐시
        }],
        messages=[{"role": "user", "content": build_user_message(question, hits)}],
        thinking={"type": "adaptive"},
    )

    try:
        if use_fallbacks:
            # 안전 분류기가 요청을 거절(stop_reason="refusal")했을 때 서버가 대체 모델로
            # 자동 재라우팅한다. 베타 기능이라 기본값은 꺼둔다.
            stream_ctx = client.beta.messages.stream(
                **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        else:
            stream_ctx = client.messages.stream(**kwargs)

        with stream_ctx as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            final = stream.get_final_message()

        if final.stop_reason == "refusal":
            print("\n[모델이 응답을 거절했습니다]", file=sys.stderr)
        usage = final.usage
        print(f"\n\n토큰: 입력 {usage.input_tokens} / 출력 {usage.output_tokens}"
              f" / 캐시읽기 {getattr(usage, 'cache_read_input_tokens', 0)}")
    except anthropic.NotFoundError:
        sys.exit(f"모델 `{model}` 을 찾을 수 없습니다. --model 로 다른 모델을 지정하세요.")
    except anthropic.RateLimitError:
        sys.exit("요청이 한도를 초과했습니다. 잠시 후 다시 시도하세요.")
    except anthropic.APIStatusError as e:
        sys.exit(f"API 오류 {e.status_code}: {e.message}")
    except anthropic.APIConnectionError as e:
        sys.exit(f"연결 실패: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="옵시디언 볼트 RAG Q&A")
    parser.add_argument("question", nargs="+", help="질문")
    parser.add_argument("--vault", type=Path, default=Path(__file__).parent.parent / "vault")
    parser.add_argument("-k", type=int, default=6, help="프롬프트에 넣을 근거 청크 수 (기본 6)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--show-context", action="store_true", help="조립된 근거를 함께 출력")
    parser.add_argument("--dry-run", action="store_true", help="API 호출 없이 프롬프트만 출력")
    parser.add_argument("--reindex", action="store_true", help="캐시를 무시하고 다시 색인")
    parser.add_argument("--no-links", action="store_true", help="링크 그래프 확장 끄기")
    parser.add_argument("--fallbacks", action="store_true",
                        help="거절 시 서버측 모델 폴백 사용(베타)")
    args = parser.parse_args()

    question = " ".join(args.question)
    index = load_index(args.vault, force=args.reindex)
    hits = Retriever(index).search(question, k=args.k, expand_links=not args.no_links)

    if not hits:
        sys.exit("검색 결과가 없습니다. 볼트 경로를 확인하세요.")

    if args.dry_run or args.show_context:
        print("=" * 70)
        print("SYSTEM\n" + SYSTEM_PROMPT)
        print("=" * 70)
        print("USER\n" + build_user_message(question, hits))
        print("=" * 70)
        approx = (len(SYSTEM_PROMPT) + len(build_user_message(question, hits))) // 2
        print(f"(대략 {approx} 토큰 — 한글은 글자당 0.5~1.5 토큰, 정확한 값은 count_tokens API로)")
        if args.dry_run:
            print_sources(hits)
            return
        print()

    answer(question, hits, args.model, args.fallbacks)
    print_sources(hits)


if __name__ == "__main__":
    main()
