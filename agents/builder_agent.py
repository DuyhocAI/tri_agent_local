"""
Hermes BuilderAgent — agents/builder_agent.py

Specialist for code generation, debugging, refactoring, and project building.
"""
from __future__ import annotations

from datetime import datetime

from agents.base_agent import BaseAgent, _TOOLS_DESCRIPTION


class BuilderAgent(BaseAgent):
    NAME = "Builder"

    def get_system_prompt(
        self,
        token_budget: int,
        long_mem: list | None = None,
        user_facts: list | None = None,
    ) -> str:
        now = datetime.now().strftime("%A, %B %d, %Y  %H:%M:%S")
        all_facts = list(long_mem or []) + list(user_facts or [])
        facts = ""
        if all_facts:
            facts = "\nKnown about user: " + "; ".join(
                f"{m.get('topic', '?')}: {m.get('value', '')}"
                for m in all_facts[-8:]
            )
        return (
            f"You are Hermes-Builder, a senior software engineer AI running on the user's own PC.\n"
            f"Current date and time: {now}\n"
            f"{facts}\n\n"
            f"SPECIALISATION: Code generation, debugging, refactoring, and project building.\n"
            f"Target response length: ~{token_budget} tokens.\n\n"
            f"RULES:\n"
            f"- Write COMPLETE, working code — never truncate or use placeholders/TODOs.\n"
            f"- Always save code to disk with write_file or edit_file — printing without saving is not enough.\n"
            f"- For modifications: read_file first → edit_file for surgical changes → run_command to verify.\n"
            f"- When building from scratch: write every file fully, then run tests.\n"
            f"{_TOOLS_DESCRIPTION}"
        )

    def run(self, ctx: list, progress_start: int, progress_end: int):
        """Stream primary LLM with tool execution loop."""
        yield from self._run_tool_loop(ctx, progress_start, progress_end)
