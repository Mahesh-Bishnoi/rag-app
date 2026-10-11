# ADR-0019: Generation prompts as plain .prompt files

## Status
Accepted

## Date
2026-10-11

## Context
- The grounded-answer prompt text lived inline in
  `src/rag_lab/adapters/generation.py`: two `grounded_prompt` branches plus
  the system envelope in `prompt_to_messages`.
- Editing prompt wording meant editing Python; prompt diffs were code diffs.

## Decision
1. Prompt text moves to `src/rag_lab/prompts/` as three plain-text files:
   `grounded_with_context.prompt` (`{passages}`, `{query}` slots),
   `grounded_no_context.prompt` (`{query}` slot), `system_cited.prompt`
   (static). No template logic or conditionals — one file per branch.
2. `rag_lab.prompts.load_prompt` (cached `importlib.resources` read) owns
   file access; `adapters/generation.py` keeps its single-owner rendering
   API with identical signatures and byte-identical output.
3. Slots are filled with `str.replace`, not `str.format`, so braces in user
   queries or passages can never break rendering.
4. `tests/test_prompts.py` pins exact rendered output plus the unknown-name
   `FileNotFoundError`; existing `test_graph.py` prompt assertions are
   unchanged.

## Alternatives Considered

### YAML frontmatter + body per file
- Pros: Version/model metadata alongside text
- Cons: New parser + header-stripper for metadata no caller consumes yet
- Rejected: speculative; revisit when eval needs prompt versioning

### Jinja2 templates
- Pros: Conditionals/loops could merge the two branches into one file
- Cons: New dependency + logic in templates for a two-branch render
- Rejected: one file per branch keeps templates logic-free

## Consequences
- Prompt wording changes touch only `.prompt` files; any drift is caught by
  the exact-output tests.
- Wheel builds must include the `.prompt` data files (verified by
  inspecting the built wheel); if hatchling ever drops them, packaging
  config gets an explicit include rather than moving the text back inline.
