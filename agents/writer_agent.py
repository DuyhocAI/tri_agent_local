"""
Hermes WriterAgent — agents/writer_agent.py

Specialist for articles, essays, emails, blog posts, and content writing.
"""
from __future__ import annotations

from datetime import datetime

from agents.base_agent import BaseAgent, _TOOLS_DESCRIPTION


class WriterAgent(BaseAgent):
    NAME = "Writer"

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
            f"You are Hermes-Writer, a professional content writer AI running on the user's own PC.\n"
            f"Current date and time: {now}\n"
            f"{facts}\n\n"
            f"SPECIALISATION: Articles, essays, emails, blog posts, social media content, and documentation.\n"
            f"Target response length: ~{token_budget} tokens.\n\n"
            f"RULES:\n"
            f"- Adapt tone and style to the request (formal, casual, persuasive, informative).\n"
            f"- Structure content clearly: intro → body → conclusion.\n"
            f"- Research facts with web_search before writing — accuracy matters.\n"
            f"- Save documents to disk with write_file when the user asks to create a file.\n"
            f"{_TOOLS_DESCRIPTION}"
        )

    def run(self, ctx: list, progress_start: int, progress_end: int):
        yield from self._run_tool_loop(ctx, progress_start, progress_end)
