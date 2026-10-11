# AGENTS.md

## Skills
- Library skills are wired via `.opencode/opencode.jsonc` → `/home/mahesh/code/skills/skills`. Scan advertised skills and load by description match before starting.
- `clean-code` is primary for any code change. `adr-skill` for one-way-door decisions. `debug` for bugs/failing tests. UI/CSS skills are low priority here.

## Quality bars (decided — treat as constraints, not preferences)
- Types: `mypy strict = true`, `ignore_missing_imports = false`. No new `Any`
  without a recorded reason; narrow at boundaries (`cast` only for untyped
  framework attrs like FastAPI `app.state`).
- Lint: ruff `E,F,I,UP,B,C4,SIM,S,RUF,A,G,PTH,RET,C90,PT,FBT`, complexity ≤10.
  `per-file-ignores` for tests only (`S101`). Never silently narrow the set —
  dropping a rule needs a stated reason.
- Tests: `pytest` + `pytest-cov`, floor 80 (`--cov-fail-under=80`). Hermetic by
  default; live-service tests (postgres, Azurite) behind `RAGLAB_LIVE_DB=1`
  and skipped otherwise. Migrations prove themselves with an
  upgrade→assert→downgrade round-trip, not just `upgrade head`.
- Freshness: at dependency-adoption time, check upstream latest (PyPI / GitHub
  / Docker Hub) and record versions + bump process in an ADR. Lock early, but
  verify latest first — `uv.lock` + digest pins freeze deliberate choices.
- Bar-height rule: strictest checker settings that stay green. If a gate
  passes trivially (lenient config, no coverage floor), the gate is the bug.

## Decisions (read before designing)
- Intent: `docs/intent/rag-lab.md` — local RAG lab, PDF+text v1, modular loaders later.
- ADRs: `docs/decisions/` (`0001`–`0004`). Don't rewrite closed ADRs; add a new numbered one that supersedes.
- Glossary: `docs/glossary.md` — use these terms.

## Intended direction (no code yet — don't assume it exists)
- Podman-first: postgres+pgvector + S3-compatible store (see ADR-002).
- Python + FastAPI + Pydantic + PydanticAI + LangChain + LangGraph, with seams: LangGraph = pipeline flow, PydanticAI = typed LLM calls, LangChain = loaders/splitters (see ADR-003).
- NVIDIA blueprint is concept reference only, not a deploy target (see ADR-001).

## Ops gotchas
- No build/test/lint commands, README, or CI exist yet. Don't invent commands; ask before adding toolchain.
- Commits are SSH-signed via global gitconfig (`commit.gpgsign=true`). If a commit fails with `ssh_askpass: No such file`, the key agent is locked — ask the user to run `ssh-add ~/.ssh/id_ed25519` in their terminal, then retry. Don't bypass with `--no-gpg-sign` without explicit approval.
- `.opencode/` is untracked scratch; don't commit it.
