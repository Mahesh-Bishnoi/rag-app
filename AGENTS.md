# AGENTS.md

## Skills
- Library skills are wired via `.opencode/opencode.jsonc` → `/home/mahesh/code/skills/skills`. Scan advertised skills and load by description match before starting.
- `clean-code` is primary for any code change. `adr-skill` for one-way-door decisions. `debug` for bugs/failing tests. UI/CSS skills are low priority here.

## Quality bars (decided — treat as constraints, not preferences)
- Types: `mypy strict = true`, `ignore_missing_imports = false`. No new `Any`
  without a recorded reason; narrow at boundaries (`cast` only for untyped
  framework attrs like FastAPI `app.state`).
- Lint: ruff `E,F,I,UP,B,C4,SIM,S,RUF,A,G,PTH,RET,C90,PT,FBT,FAST`, complexity
  ≤10. `per-file-ignores` for tests only (`S101`). Never silently narrow the
  set — dropping a rule needs a stated reason.
- FastAPI: vendor skill (`.agents/skills/fastapi`, version-matched) overrides
  generic habits — `Annotated` params, Pydantic returns, `fastapi dev`/`run`.
  Project ADRs override the skill where recorded.
- Persistence: SQLModel table rows (`adapters/tables.py`) mirror migrations
  exactly; Pydantic DTOs are never persisted and rows never leave endpoints.
- Tests: `pytest` + `pytest-cov`, floor 80 (`--cov-fail-under=80`). Hermetic by
  default; live-service tests (postgres, Azurite) behind `RAGLAB_LIVE_DB=1`
  and skipped otherwise. Migrations prove themselves with an
  upgrade→assert→downgrade round-trip, not just `upgrade head`.
- Freshness: at dependency-adoption time, check upstream latest (PyPI / GitHub
  / Docker Hub) and record versions + bump process in an ADR. Lock early, but
  verify latest first — `uv.lock` + digest pins freeze deliberate choices.
- Bar-height rule: strictest checker settings that stay green. If a gate
  passes trivially (lenient config, no coverage floor), the gate is the bug.
- Architecture: `lint-imports` (`.importlinter` layers contract) stays kept.
  Reorder layers rather than loosening when a real need arises.
- Supply chain: `uv audit` clean before release and on dependency changes;
  `deptry src alembic` clean (`per_rule_ignores` in pyproject only for
  used-without-import deps, with reason comments).
- Secrets: `gitleaks detect --source . --config .gitleaks.toml` clean;
  allowlist entries only for well-known non-secrets, each with a reason.

## Decisions (read before designing)
- Intent: `docs/intent/rag-lab.md` — local RAG lab, PDF+text v1, modular loaders later.
- ADRs: `docs/decisions/` (`0001`–`0004`). Don't rewrite closed ADRs; add a new numbered one that supersedes.
- Glossary: `docs/glossary.md` — use these terms.

## Intended direction (no code yet — don't assume it exists)
- Podman-first: postgres+pgvector + S3-compatible store (see ADR-002).
- Python + FastAPI + Pydantic + PydanticAI + LangChain + LangGraph, with seams: LangGraph = pipeline flow, PydanticAI = typed LLM calls, LangChain = loaders/splitters (see ADR-003).
- NVIDIA blueprint is concept reference only, not a deploy target (see ADR-001).

## Skills in use (`/setup` 2026-10-11, scope: project-local)

- Library wired via `.opencode/opencode.jsonc` `skills` → `/home/mahesh/code/skills/skills` (machine-local; teammates rerun `/setup`).
- Guardrail: `skillspector` v2.12.0 (uv) + `[mcp]` extra, `SKILLSPECTOR_PROVIDER=opencode_cli`, Cisco `skill-scanner` v2.2.2 pre-installed; `/skillspector` command copied to `.opencode/tools/`+`commands/`. OpenShell declined (no CLI, no policy).
- `@archify` (`.agents/skills/archify`, doctor ready) — Cisco FAIL 17 CRITICAL reviewed as signature overmatches (literal-arg child_process, `compile(` name matches in one test file); SkillSpector timed out on 11MB skill — user kept.
- `@agent-browser` (CLI 0.39.0, Chrome 155, doctor 8/8; stub `.agents/skills/agent-browser`; MCP `agent-browser` core profile in `.opencode/opencode.jsonc`) — SkillSpector LOW/SAFE, Cisco SAFE — user kept.
- `@ui-ux-pro-max` (`.opencode/skills/ui-ux-pro-max`, uipro v2.15.0, target opencode) — Cisco 2 HIGH are YARA overmatches on Uno Platform CSV doc rows — user kept.
- `@design-taste-frontend` (`.agents/skills/design-taste-frontend`) — SkillSpector CAUTION 46 (HIGHs are documented `<!-- -->` examples, no invisible chars), Cisco SAFE — user kept.
- Declined: drawio, browser-harness, graphify, openwiki.

## Ops gotchas
- No build/test/lint commands, README, or CI exist yet. Don't invent commands; ask before adding toolchain.
- Commits are SSH-signed via global gitconfig (`commit.gpgsign=true`). If a commit fails with `ssh_askpass: No such file`, the key agent is locked — ask the user to run `ssh-add ~/.ssh/id_ed25519` in their terminal, then retry. Don't bypass with `--no-gpg-sign` without explicit approval.
- `.opencode/` is untracked scratch; don't commit it.
