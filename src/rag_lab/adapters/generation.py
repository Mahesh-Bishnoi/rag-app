"""Typed generation: PydanticAI owns LLM calls, LlmPort stays the transport.

Prompt text lives in rag_lab/prompts/*.prompt (plain text, {query}/{passages}
slots); this module only renders them and stays the single owner
(api/worker must not fork prompt text). The agent wraps the injected LlmPort
via FunctionModel, so the StubLlm fake keeps hermetic tests offline while
prod runs the same validated path over OpenAICompatibleLlm. Output is parsed
into GroundedAnswer — an empty answer fails validation and surfaces as a
generation error (never a silent 200).
"""

import json
from collections.abc import Sequence

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from rag_lab.adapters.llm_openai import LlmPort
from rag_lab.prompts import load_prompt


class GroundedAnswer(BaseModel):
    answer: str = Field(min_length=1)


def grounded_prompt(query: str, contexts: list[str]) -> str:
    """Numbered-context prompt; the single owner (api/worker must not fork it)."""
    if not contexts:
        return load_prompt("grounded_no_context.prompt").replace("{query}", query)
    numbered = "\n\n".join(
        f"[{index}] {content}" for index, content in enumerate(contexts, start=1)
    )
    return (
        load_prompt("grounded_with_context.prompt")
        .replace("{passages}", numbered)
        .replace("{query}", query)
    )


def prompt_to_messages(prompt: str) -> list[dict[str, str]]:
    """System + user envelope shared by every generation call."""
    return [
        {"role": "system", "content": load_prompt("system_cited.prompt")},
        {"role": "user", "content": prompt},
    ]


def complete(llm: LlmPort, prompt: str) -> GroundedAnswer:
    """Run a prebuilt prompt through a PydanticAI agent. Raises on failure."""

    def _model(_messages: Sequence[ModelMessage], _info: AgentInfo) -> ModelResponse:
        text = llm.chat(prompt_to_messages(prompt))
        return ModelResponse(parts=[TextPart(content=json.dumps({"answer": text}))])

    agent: Agent[None, GroundedAnswer] = Agent(
        model=FunctionModel(_model), output_type=GroundedAnswer
    )
    return agent.run_sync(prompt).output


def generate_answer(llm: LlmPort, query: str, contexts: list[str]) -> GroundedAnswer:
    """Run the grounded prompt through a PydanticAI agent. Raises on failure."""
    return complete(llm, grounded_prompt(query, contexts))
