# 저장소에 포함된 스킬

`superpowers` 스킬을 저장소에 직접 복사해 두었습니다.

## 왜 플러그인이 아니라 복사인가

플러그인 자동 설치(`.claude/settings.json`의 `enabledPlugins`)는 폴더 신뢰(trust) 절차를 거쳐야 동작합니다.
Claude Code 웹/모바일 세션에는 그 절차가 없어서 플러그인이 설치되지 않았습니다.

스킬 파일을 저장소에 두면 설치·네트워크·신뢰 절차 없이 세션 시작 시 그대로 읽힙니다.

## 갱신

원본이 업데이트되면 수동으로 다시 복사해야 합니다.

```bash
claude plugin marketplace add obra/superpowers-marketplace
claude plugin install superpowers@superpowers-marketplace
cp -r ~/.claude/plugins/cache/superpowers-marketplace/superpowers/*/skills/* .claude/skills/
```

## 출처

[obra/superpowers](https://github.com/obra/superpowers) — MIT License, Copyright (c) 2025 Jesse Vincent.
전문은 `LICENSE-superpowers` 참조.
