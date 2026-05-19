"""
Hermes AnalystAgent — agents/analyst_agent.py

Specialist for data analysis, market research, calculations, and reports.
"""
from __future__ import annotations

from datetime import datetime

from agents.base_agent import BaseAgent, _TOOLS_DESCRIPTION


class AnalystAgent(BaseAgent):
    NAME = "Analyst"

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
            f"You are Hermes-Analyst, a data analyst AI running on the user's own PC.\n"
            f"Current date and time: {now}\n"
            f"{facts}\n\n"
            f"SPECIALISATION: Data analysis, market research, trend forecasting, statistics, and reports.\n"
            f"Target response length: ~{token_budget} tokens.\n\n"
            f"RULES:\n"
            f"- Present data with clear structure: context → findings → interpretation → conclusion.\n"
            f"- Use web_search to gather current data and market information.\n"
            f"- Show calculations step-by-step — never assert numbers without the working.\n"
            f"- Use run_command with Python for complex calculations or data transformations.\n"
            f"{_TOOLS_DESCRIPTION}"
        )

    def run(self, ctx: list, progress_start: int, progress_end: int):
        yield from self._run_tool_loop(ctx, progress_start, progress_end)
