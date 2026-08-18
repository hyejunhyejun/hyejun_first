"""하이브리드 검색: BM25(키워드) + 벡터(의미) + 위키링크 그래프 확장.

의존성 없음. 벡터는 학습이 필요 없는 해싱 임베딩을 기본으로 쓴다 —
오프라인에서 파이프라인 전체를 검증할 수 있고, 나중에 진짜 임베딩 API로
`embed()` 함수 하나만 갈아끼우면 된다(아래 `EMBEDDING 교체` 주석 참고).

배경 설명은 vault/10-개념/하이브리드 검색.md 참고.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass

from vault_index import Chunk, VaultIndex

# --------------------------------------------------------------------------- #
# 토큰화 — 한국어를 형태소 분석기 없이 다루는 절충안
# --------------------------------------------------------------------------- #

HANGUL_RE = re.compile(r"[가-힣]+")
LATIN_RE = re.compile(r"[a-zA-Z0-9_]+")

# 조사·의문사 같은 기능어는 2-gram으로 자르기 전에 지운다.
# 안 지우면 "얼마나" → '얼마','마나' 같은 조각이 드물다는 이유로 높은 IDF를 받아
# 엉뚱한 문서를 1위로 끌어올린다(실제로 이 샘플에서 겪은 문제다).
STOPWORDS = [
    "무엇인가", "어떻게", "얼마나", "어떤", "무엇", "언제", "어디", "누구", "왜",
    "하나요", "인가요", "있나요", "되나요", "할까요", "합니까", "입니까", "나요", "까요",
    "합니다", "습니다", "입니다", "니다", "하는", "해서", "하고", "이다",
    "그리고", "하지만", "때문에", "위해서", "위해", "대해서", "대해", "대한",
    "그것", "이것", "저것", "것을", "것이", "우리",
]
STOPWORD_RE = re.compile("|".join(sorted(STOPWORDS, key=len, reverse=True)))


def tokenize(text: str) -> list[str]:
    """한글 구간은 문자 2-gram, 영문/숫자는 단어 단위로 자른다.

    `RAG를` / `RAG는` 처럼 조사가 붙어도 겹치는 2-gram이 생겨 매칭된다.
    형태소 분석기(kiwi, mecab)를 쓸 수 있다면 이 함수만 바꾸면 된다.
    """
    text = STOPWORD_RE.sub(" ", text.lower())
    tokens: list[str] = [m.group() for m in LATIN_RE.finditer(text)]
    for m in HANGUL_RE.finditer(text):
        word = m.group()
        if len(word) <= 2:
            tokens.append(word)
        else:
            tokens.extend(word[i:i + 2] for i in range(len(word) - 1))
    return tokens


def chunk_text_for_search(chunk: Chunk) -> str:
    """검색용 텍스트에는 제목/헤딩/태그를 함께 넣어 가중치를 준다."""
    # 제목과 헤딩을 두 번씩 넣어 가중치를 준다. "비공개 노트는 어떻게 막나요" 처럼
    # 질문이 헤딩과 거의 같은 경우가 실제 위키 검색에서 아주 흔하다.
    return " ".join([chunk.note_title, chunk.note_title,
                     chunk.heading_path, chunk.heading_path,
                     " ".join(chunk.tags), chunk.text])


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #

class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avgdl = (sum(self.lengths) / len(docs)) if docs else 0.0
        self.df: Counter = Counter()
        for d in self.docs:
            self.df.update(d.keys())
        self.N = len(docs)

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        if n == 0:
            return 0.0
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def score(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.N
        for term in set(query_tokens):
            idf = self.idf(term)
            if idf == 0.0:
                continue
            for i, doc in enumerate(self.docs):
                tf = doc.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / denom
        return scores


# --------------------------------------------------------------------------- #
# 벡터 — 해싱 임베딩(기본값)
# --------------------------------------------------------------------------- #

DIM = 512


def embed(text: str, idf=None) -> list[float]:
    """텍스트를 L2 정규화된 고정 차원 벡터로 만든다.

    토큰을 해시해 차원에 배정하고 부호를 섞는 'hashing trick'. 학습도 네트워크도
    필요 없지만 의미 유사도는 약하다(동의어를 못 잡는다).

    === EMBEDDING 교체 ===
    실제 서비스에서는 이 함수를 임베딩 API/로컬 모델 호출로 바꾼다. 인터페이스는 그대로다.
    한국어 품질은 반드시 자기 데이터로 비교해서 고를 것 — vault/10-개념/임베딩과 벡터 검색.md 참고.

        def embed(text: str) -> list[float]:
            vec = my_embedding_client.embed(text, model="...")   # 예: Voyage AI, bge-m3
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            return [v / norm for v in vec]

    Anthropic은 임베딩 엔드포인트를 제공하지 않는다. 검색은 임베딩 제공자, 생성은 Claude로 나눈다.
    """
    vec = [0.0] * DIM
    for token, count in Counter(tokenize(text)).items():
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        idx = h % DIM
        sign = 1.0 if (h >> 16) & 1 else -1.0
        weight = (1 + math.log(count)) * (idf(token) if idf else 1.0)   # 서브리니어 TF × IDF
        vec[idx] += sign * weight
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))               # 둘 다 정규화되어 있으므로 내적 = 코사인


# --------------------------------------------------------------------------- #
# 하이브리드 검색
# --------------------------------------------------------------------------- #

@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None
    via_link: bool = False        # 링크 그래프 확장으로 들어온 결과인가

    def why(self) -> str:
        bits = []
        if self.bm25_rank is not None:
            bits.append(f"키워드 {self.bm25_rank + 1}위")
        if self.vector_rank is not None:
            bits.append(f"벡터 {self.vector_rank + 1}위")
        if self.via_link:
            bits.append("링크 확장")
        return ", ".join(bits) or "-"


RRF_K = 60            # RRF 상수. 보통 60을 그대로 쓴다
CANDIDATES = 30       # 각 검색기에서 뽑아 융합할 후보 수. 전체를 융합하면 점수가 뭉개진다
LINK_SLOTS = 2        # 결과 k개 중 링크 확장에 내어줄 자리 수(0이면 확장 끔)
HUB_TAGS = {"moc", "index"}   # 목차 노트는 거의 모든 노트를 링크하므로 확장 기준에서 뺀다


class Retriever:
    def __init__(self, index: VaultIndex):
        self.index = index
        self.chunks = index.chunks
        texts = [chunk_text_for_search(c) for c in self.chunks]
        self.bm25 = BM25([tokenize(t) for t in texts])
        # 해싱 임베딩에도 IDF를 걸어준다. 흔한 조사 2-gram이 벡터를 지배하는 것을 막는다.
        self.vectors = [embed(t, self.bm25.idf) for t in texts]

    def search(self, query: str, k: int = 6, expand_links: bool = True) -> list[Hit]:
        q_tokens = tokenize(query)
        q_vec = embed(query, self.bm25.idf)

        bm25_scores = self.bm25.score(q_tokens)
        vec_scores = [cosine(q_vec, v) for v in self.vectors]

        # 각 검색기의 상위 후보만 융합한다. 전체 순위를 융합하면 점수 차이가 사라져
        # 뒤에 붙는 가산점 하나에 순위가 통째로 뒤집힌다.
        bm25_top = [i for i in sorted(range(len(self.chunks)), key=lambda i: -bm25_scores[i])
                    if bm25_scores[i] > 0][:CANDIDATES]
        vec_top = sorted(range(len(self.chunks)), key=lambda i: -vec_scores[i])[:CANDIDATES]
        bm25_rank = {i: r for r, i in enumerate(bm25_top)}
        vec_rank = {i: r for r, i in enumerate(vec_top)}

        # RRF: 점수 스케일이 다른 두 랭킹을 순위만으로 합친다
        fused: dict[int, float] = {}
        for i in set(bm25_top) | set(vec_top):
            score = 0.0
            if i in bm25_rank:
                score += 1.0 / (RRF_K + bm25_rank[i] + 1)
            if i in vec_rank:
                score += 1.0 / (RRF_K + vec_rank[i] + 1)
            fused[i] = score

        ordered = sorted(fused, key=lambda i: -fused[i])

        # 링크 그래프 확장: 상위 결과를 밀어내지 않고, 마지막 몇 자리만 이웃 노트에 배정한다.
        # 점수에 가산점을 주는 방식은 쓰지 않는다 — RRF 점수는 간격이 촘촘해서
        # 작은 가산점 하나로 1위가 통째로 바뀐다(정확도가 무너진다).
        # 노리는 효과는 정밀도가 아니라 재현율이다: "개념 노트는 찾았는데 결론이 적힌
        # 회의록을 놓치는" 상황을 막는다. vault/30-기록/2026-08-17 파일럿 회고.md 참고.
        slots = LINK_SLOTS if expand_links else 0
        primary = ordered[:max(k - slots, 1)]
        picks: list[int] = []
        if slots and primary:
            hub = {n.path for n in self.index.notes if HUB_TAGS & set(n.tags)}
            seed_notes: list[str] = []
            for i in primary[:2]:
                if self.chunks[i].note_path not in seed_notes:
                    seed_notes.append(self.chunks[i].note_path)
            neighbors: set[str] = set()
            for path in seed_notes:
                neighbors |= self.index.neighbors(path)
            neighbors -= hub | {self.chunks[i].note_path for i in primary}
            picks = [i for i in ordered
                     if i not in primary and self.chunks[i].note_path in neighbors][:slots]

        selected = primary + picks + [i for i in ordered if i not in primary and i not in picks]
        selected = selected[:k]
        return [Hit(
            chunk=self.chunks[i],
            score=fused[i],
            bm25_rank=bm25_rank.get(i),
            vector_rank=vec_rank.get(i),
            via_link=i in picks,
        ) for i in selected]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from vault_index import load_index

    vault = Path(__file__).parent.parent / "vault"
    query = " ".join(sys.argv[1:]) or "청크는 얼마나 크게 잘라야 하나요"
    retriever = Retriever(load_index(vault))
    print(f"질문: {query}\n")
    for rank, hit in enumerate(retriever.search(query), 1):
        print(f"[{rank}] {hit.chunk.citation()}   (점수 {hit.score:.4f} · {hit.why()})")
        print(f"    {hit.chunk.text[:90].replace(chr(10), ' ')}...")
