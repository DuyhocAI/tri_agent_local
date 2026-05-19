"""
Hermes SkillRegistry — loads YAML skill definitions and dispatches tool calls.

Design for P1 transition:
  - YAML files provide metadata: name, description, parameters, destructive flag.
  - Execution delegates to an injected tools_dict (the existing _TOOLS from
    orchestrator.py). This avoids circular imports while keeping zero behaviour
    change during the migration.
  - BM25 search enables intent-based skill lookup for agent routing (P2).
  - Hot-reload via reload() is used by the HCI Skills tab (P5).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger("hermes.skills")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SkillParam:
    type: str = "string"
    required: bool = False
    description: str = ""


@dataclass
class SkillDef:
    name: str
    description: str
    destructive: bool
    parameters: dict[str, SkillParam]
    domain: str = ""
    handler_path: str = ""  # dotted path for future direct dispatch


# ── Registry ─────────────────────────────────────────────────────────────────

class SkillRegistry:
    """
    Metadata store + dispatcher for Hermes skills.

    Args:
        skill_dirs:  Paths to scan for *.yaml skill definitions.
        tools_dict:  Callable dict in the form {name: lambda args: ToolResult}.
                     Injected at construction to avoid circular imports.
                     Falls back to empty dict (all calls return error) if omitted.
    """

    def __init__(
        self,
        skill_dirs: list[Path | str] | None = None,
        tools_dict: dict | None = None,
        destructive_set: set | None = None,
    ):
        if skill_dirs is None:
            skill_dirs = [Path(__file__).parent]
        self._skill_dirs: list[Path] = [Path(d) for d in skill_dirs]
        self._tools: dict = tools_dict or {}
        self._destructive: set[str] = destructive_set or set()
        self._skills: dict[str, SkillDef] = {}
        self.load_all()

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_all(self) -> None:
        """Walk all skill dirs and load every *.yaml file."""
        count_files = 0
        for d in self._skill_dirs:
            if not d.is_dir():
                continue
            for yaml_file in sorted(d.glob("*.yaml")):
                try:
                    self._load_file(yaml_file)
                    count_files += 1
                except Exception as e:
                    logger.warning(f"Skill load error [{yaml_file.name}]: {e}")
        logger.info(
            f"SkillRegistry: {len(self._skills)} skills from {count_files} files"
        )

    def _load_file(self, path: Path) -> None:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data:
            return
        domain = data.get("domain", path.stem)
        for raw in data.get("skills", []):
            name = raw.get("name", "")
            if not name:
                continue
            params: dict[str, SkillParam] = {}
            for pname, pdata in raw.get("parameters", {}).items():
                if isinstance(pdata, dict):
                    params[pname] = SkillParam(
                        type=pdata.get("type", "string"),
                        required=pdata.get("required", False),
                        description=pdata.get("description", ""),
                    )
                else:
                    params[pname] = SkillParam()
            self._skills[name] = SkillDef(
                name=name,
                description=raw.get("description", ""),
                destructive=raw.get("destructive", False),
                parameters=params,
                domain=domain,
                handler_path=raw.get("handler", ""),
            )

    def reload(self) -> None:
        """Hot-reload all YAML files (used by HCI Skills tab)."""
        self._skills.clear()
        self.load_all()
        logger.info("SkillRegistry reloaded")

    # ── Execution ─────────────────────────────────────────────────────────────

    def execute(self, name: str, args: dict):
        """Execute a skill by delegating to the injected tools_dict."""
        if name not in self._tools:
            from agents.base_agent import _make_tool_error
            return _make_tool_error(f"Unknown skill: {name}")
        try:
            return self._tools[name](args)
        except KeyError as e:
            from agents.base_agent import _make_tool_error
            return _make_tool_error(f"Missing argument for {name}: {e}")
        except Exception as e:
            from agents.base_agent import _make_tool_error
            return _make_tool_error(f"Skill {name} error: {e}")

    def is_destructive(self, name: str) -> bool:
        """Check if a skill requires supervisor approval before execution."""
        # Check YAML metadata first, then fall back to injected destructive_set
        skill = self._skills.get(name)
        if skill is not None:
            return skill.destructive
        return name in self._destructive

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 8) -> list[SkillDef]:
        """BM25 search over skill descriptions. Falls back to substring match."""
        if not self._skills:
            return []
        try:
            from rank_bm25 import BM25Okapi
            skills_list = list(self._skills.values())
            corpus = [s.description.lower().split() for s in skills_list]
            bm25 = BM25Okapi(corpus)
            scores = bm25.get_scores(query.lower().split())
            ranked = sorted(zip(scores, skills_list), key=lambda x: -x[0])
            results = [s for score, s in ranked[:top_k] if score > 0]
            return results or self._substring_search(query, top_k)
        except ImportError:
            return self._substring_search(query, top_k)

    def _substring_search(self, query: str, top_k: int) -> list[SkillDef]:
        q = query.lower()
        return [
            s for s in self._skills.values()
            if q in s.name.lower() or q in s.description.lower()
        ][:top_k]

    # ── Introspection ─────────────────────────────────────────────────────────

    def get(self, name: str) -> SkillDef | None:
        return self._skills.get(name)

    def all_skills(self) -> list[SkillDef]:
        return list(self._skills.values())

    def names(self) -> list[str]:
        return list(self._skills.keys())

    def by_domain(self) -> dict[str, list[SkillDef]]:
        result: dict[str, list[SkillDef]] = {}
        for s in self._skills.values():
            result.setdefault(s.domain, []).append(s)
        return result

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._tools  # executable = in tools_dict
