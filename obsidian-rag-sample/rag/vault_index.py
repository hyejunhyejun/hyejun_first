"""옵시디언 볼트(마크다운 폴더)를 RAG 검색 단위로 색인한다.

의존성 없음(표준 라이브러리만). 하는 일:
  1. .md 파일을 모두 읽어 YAML 프론트매터를 분리한다
  2. 헤딩(#, ##, ###) 기준으로 청크를 만든다 — 짧은 절은 합치고 긴 절은 문단 경계에서 자른다
  3. [[위키링크]] 를 파싱해 노트 간 링크 그래프를 만든다
  4. 결과를 JSON으로 캐시하고, 파일 수정 시각이 바뀐 경우에만 다시 만든다

왜 헤딩 기준인가는 vault/10-개념/청킹 전략.md 참고.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

# 청킹 파라미터. 자기 볼트에 맞춰 조정하는 것이 첫 번째 튜닝 포인트다.
MIN_CHARS = 200    # 이보다 짧은 절은 앞 청크에 붙인다
MAX_CHARS = 1200   # 이보다 긴 절은 문단 경계에서 자른다
OVERLAP_CHARS = 300  # 자를 때 앞 청크의 끝 문단을 이만큼 겹쳐 남긴다

INDEX_VERSION = 3   # 색인 로직을 바꾸면 올린다 — 캐시가 자동으로 무효화된다

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
WIKILINK_RE = re.compile(r"\[\[([^\]\[|#]+)(?:[#|][^\]\[]*)?\]\]")


@dataclass
class Note:
    path: str                     # 볼트 기준 상대 경로
    title: str
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    updated: str = ""
    owner: str = ""
    links: list[str] = field(default_factory=list)   # 이 노트가 가리키는 노트 경로
    mtime: float = 0.0


@dataclass
class Chunk:
    id: str
    note_path: str
    note_title: str
    heading_path: str             # "RAG란 무엇인가 > 파이프라인 5단계"
    text: str
    tags: list[str] = field(default_factory=list)
    updated: str = ""

    def citation(self) -> str:
        return f"{self.note_path} § {self.heading_path}" if self.heading_path else self.note_path


# --------------------------------------------------------------------------- #
# 프론트매터
# --------------------------------------------------------------------------- #

def parse_frontmatter(raw: str) -> tuple[dict, str]:
    """`---` 로 감싼 YAML 프론트매터를 (dict, 본문) 으로 분리한다.

    실무에서 쓰는 부분집합만 지원한다(문자열, [a, b] 인라인 리스트, `- item` 블록 리스트).
    복잡한 YAML을 쓴다면 PyYAML로 교체하면 된다.
    """
    if not raw.startswith("---"):
        return {}, raw

    lines = raw.splitlines()
    end = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end = i
            break
    if end is None:
        return {}, raw

    meta: dict = {}
    key = None
    for line in lines[1:end]:
        if not line.strip():
            continue
        if line.lstrip().startswith("- ") and key:            # 블록 리스트 항목
            meta.setdefault(key, [])
            if isinstance(meta[key], list):
                meta[key].append(line.lstrip()[2:].strip())
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if value.startswith("[") and value.endswith("]"):     # 인라인 리스트
            items = [v.strip().strip("'\"") for v in value[1:-1].split(",")]
            meta[key] = [v for v in items if v]
        elif value:
            meta[key] = value.strip("'\"")
        else:
            meta[key] = []                                    # 뒤에 블록 리스트가 올 자리
    return meta, "\n".join(lines[end + 1:])


def _as_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value:
        return [value]
    return []


# --------------------------------------------------------------------------- #
# 청킹
# --------------------------------------------------------------------------- #

def split_sections(body: str) -> list[tuple[str, str]]:
    """본문을 (헤딩 경로, 텍스트) 목록으로 나눈다. 코드 펜스 안의 `#` 는 헤딩이 아니다.

    헤딩 줄은 본문에 그대로 남긴다. 짧은 절이 합쳐질 때 소제목이 사라지면
    "비공개 노트는 어떻게 막나요" 같은 FAQ 제목이 검색되지 않는다.
    """
    sections: list[tuple[str, str]] = []
    stack: list[str] = []
    buf: list[str] = []
    current = ""
    in_fence = False

    def flush():
        text = "\n".join(buf).strip()
        if text:
            sections.append((current, text))
        buf.clear()

    for line in body.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
        if not in_fence:
            m = HEADING_RE.match(line)
            if m:
                flush()
                level, title = len(m.group(1)), m.group(2).strip()
                stack = stack[: level - 1] + [title]
                current = " > ".join(stack)
                buf.append(line)          # 헤딩 줄을 본문에 유지
                continue
        buf.append(line)
    flush()
    return sections


def _split_long(text: str) -> list[str]:
    """긴 텍스트를 문단 경계에서 자르되 끝 문단을 다음 조각에 겹쳐 남긴다."""
    paragraphs = [p for p in re.split(r"\n{2,}", text) if p.strip()]
    parts: list[str] = []
    buf: list[str] = []
    size = 0
    for para in paragraphs:
        if size and size + len(para) > MAX_CHARS:
            parts.append("\n\n".join(buf))
            tail = buf[-1] if len(buf[-1]) <= OVERLAP_CHARS else ""
            buf = [tail] if tail else []
            size = len(tail)
        buf.append(para)
        size += len(para) + 2
    if buf:
        parts.append("\n\n".join(buf))
    return parts or [text]


def common_heading(headings: list[str]) -> str:
    """여러 절을 합칠 때 쓸 헤딩 경로 = 공통 상위 경로."""
    parts = [h.split(" > ") for h in headings if h]
    if not parts:
        return ""
    prefix = parts[0]
    for other in parts[1:]:
        keep = 0
        for a, b in zip(prefix, other):
            if a != b:
                break
            keep += 1
        prefix = prefix[:keep]
    return " > ".join(prefix) or parts[0][0]


def chunk_note(note: Note, body: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    pending: list[tuple[str, str]] = []   # 너무 짧아서 다음 절과 합칠 절들

    def emit(heading: str, text: str):
        for part in _split_long(text):
            cid = hashlib.sha1(f"{note.path}|{heading}|{part[:80]}".encode()).hexdigest()[:12]
            chunks.append(Chunk(
                id=cid,
                note_path=note.path,
                note_title=note.title,
                heading_path=heading,
                text=part.strip(),
                tags=note.tags,
                updated=note.updated,
            ))

    for heading, text in split_sections(body):
        if pending:
            heading = common_heading([h for h, _ in pending] + [heading])
            text = "\n\n".join([t for _, t in pending] + [text])
            pending = []
        if len(text) < MIN_CHARS:
            pending.append((heading, text))
            continue
        emit(heading, text)

    if pending:   # 마지막에 남은 짧은 절
        head0 = common_heading([h for h, _ in pending])
        tail = "\n\n".join(t for _, t in pending)
        if len(tail) < MIN_CHARS and chunks:
            chunks[-1].text += "\n\n" + tail       # 외톨이 청크를 만들지 않는다
        else:
            emit(head0, tail)
    return chunks


# --------------------------------------------------------------------------- #
# 볼트 전체 색인
# --------------------------------------------------------------------------- #

class VaultIndex:
    def __init__(self, notes: list[Note], chunks: list[Chunk]):
        self.notes = notes
        self.chunks = chunks
        self.by_path = {n.path: n for n in notes}
        self.backlinks: dict[str, set[str]] = {n.path: set() for n in notes}
        for n in notes:
            for target in n.links:
                if target in self.backlinks:
                    self.backlinks[target].add(n.path)

    def neighbors(self, path: str) -> set[str]:
        """이 노트가 가리키는 노트 + 이 노트를 가리키는 노트."""
        out = set(self.by_path[path].links) if path in self.by_path else set()
        return out | self.backlinks.get(path, set())

    def to_json(self) -> str:
        return json.dumps(
            {"version": INDEX_VERSION,
             "notes": [asdict(n) for n in self.notes],
             "chunks": [asdict(c) for c in self.chunks]},
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, raw: str) -> "VaultIndex":
        data = json.loads(raw)
        if data.get("version") != INDEX_VERSION:
            raise ValueError("index version mismatch")
        return cls([Note(**n) for n in data["notes"]], [Chunk(**c) for c in data["chunks"]])


def _resolve_links(notes: list[Note], raw_links: dict[str, list[str]]) -> None:
    """[[표시 이름]] 을 실제 노트 경로로 해석한다(제목 / 별칭 / 파일명 순)."""
    lookup: dict[str, str] = {}
    for n in notes:
        for key in [n.title, Path(n.path).stem, *n.aliases]:
            lookup.setdefault(key.strip().lower(), n.path)
    for n in notes:
        seen: list[str] = []
        for name in raw_links.get(n.path, []):
            target = lookup.get(name.strip().lower())
            if target and target != n.path and target not in seen:
                seen.append(target)
        n.links = seen


# moc/index 노트는 본문이 링크 목록뿐이라 검색 결과 자리만 차지한다. 색인에서 빼는 편이 낫다.
# (링크 그래프는 어차피 다른 노트들의 위키링크로 충분히 만들어진다.)
def build_index(vault: Path, skip_tags=("deprecated", "moc"), skip_private=True) -> VaultIndex:
    notes: list[Note] = []
    chunks: list[Chunk] = []
    raw_links: dict[str, list[str]] = {}

    for md in sorted(vault.rglob("*.md")):
        if any(part.startswith(".") for part in md.relative_to(vault).parts):
            continue                                  # .obsidian, .trash 등
        raw = md.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(raw)
        tags = _as_list(meta.get("tags"))

        # 색인 제외는 프롬프트가 아니라 여기서 한다 — vault/20-실무/자주 묻는 질문.md 참고
        if any(t in skip_tags for t in tags):
            continue
        if skip_private and str(meta.get("visibility", "")).lower() == "private":
            continue

        rel = str(md.relative_to(vault))
        note = Note(
            path=rel,
            title=str(meta.get("title") or md.stem),
            aliases=_as_list(meta.get("aliases")),
            tags=tags,
            updated=str(meta.get("updated", "")),
            owner=str(meta.get("owner", "")),
            mtime=md.stat().st_mtime,
        )
        raw_links[rel] = WIKILINK_RE.findall(body)
        notes.append(note)
        chunks.extend(chunk_note(note, body))

    _resolve_links(notes, raw_links)
    return VaultIndex(notes, chunks)


def load_index(vault: Path, cache: Path | None = None, force: bool = False) -> VaultIndex:
    """캐시가 최신이면 재사용하고, 노트가 바뀌었으면 다시 색인한다."""
    cache = cache or vault.parent / ".rag_index.json"
    newest = max((p.stat().st_mtime for p in vault.rglob("*.md")), default=0.0)
    if not force and cache.exists() and cache.stat().st_mtime >= newest:
        try:
            return VaultIndex.from_json(cache.read_text(encoding="utf-8"))
        except Exception:
            pass                                        # 캐시가 깨졌으면 그냥 다시 만든다
    index = build_index(vault)
    cache.write_text(index.to_json(), encoding="utf-8")
    return index


if __name__ == "__main__":
    import sys

    vault_path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent.parent / "vault")
    idx = build_index(vault_path)
    print(f"노트 {len(idx.notes)}개, 청크 {len(idx.chunks)}개")
    sizes = sorted(len(c.text) for c in idx.chunks)
    print(f"청크 길이: 최소 {sizes[0]} / 중앙값 {sizes[len(sizes)//2]} / 최대 {sizes[-1]}")
    for c in idx.chunks[:5]:
        print(f"  - [{len(c.text):>4}자] {c.citation()}")
