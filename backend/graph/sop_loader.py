"""
sop_loader.py

Loads Standard Operating Procedures from sops.yaml. This is the ONLY
place SOPs are read from disk -- nodes never hardcode policy content.
Editing sops.yaml (adding, removing, changing a rule) requires zero
changes here or anywhere else in the graph/LLM code.
"""
from __future__ import annotations

import os
import yaml
from dataclasses import dataclass
from typing import Optional


SOPS_PATH = os.path.join(os.path.dirname(__file__), "..", "sops", "sops.yaml")


@dataclass
class SOP:
    id: str
    category: str
    severity: str
    condition: str
    thresholds: Optional[dict]
    activities: list
    advice: str


def load_sops(path: str = SOPS_PATH) -> list[SOP]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    sops = []
    for raw in data.get("sops", []):
        sops.append(
            SOP(
                id=raw["id"],
                category=raw["category"],
                severity=raw["severity"],
                condition=raw["condition"].strip(),
                thresholds=raw.get("thresholds"),
                activities=raw.get("activities", []),
                advice=raw["advice"].strip(),
            )
        )
    return sops


def sops_as_prompt_block(sops: list[SOP]) -> str:
    """
    Render SOPs as a compact, LLM-readable text block for the matching prompt.
    Includes id/category/severity/condition/activities but NOT the advice text --
    the matcher only needs to decide WHICH SOP applies, not what it says.
    Keeping advice out of the matching prompt also avoids the LLM being tempted
    to paraphrase advice during matching instead of during composition.
    """
    lines = []
    for s in sops:
        lines.append(
            f"- id: {s.id}\n"
            f"  category: {s.category}\n"
            f"  severity: {s.severity}\n"
            f"  relevant_activities: {', '.join(s.activities)}\n"
            f"  condition: {s.condition}"
        )
    return "\n".join(lines)


def get_sop_by_id(sops: list[SOP], sop_id: str) -> Optional[SOP]:
    for s in sops:
        if s.id == sop_id:
            return s
    return None


if __name__ == "__main__":
    sops = load_sops()
    print(f"Loaded {len(sops)} SOPs.")
    for s in sops:
        print(f"  {s.id} [{s.category}/{s.severity}]")
