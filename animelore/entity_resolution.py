"""
Entity resolution utilities for AnimeloreGraph.

Maintains an alias map to resolve alternate titles to canonical node IDs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .models import BaseNode, Universe, build_node_id


def _alias_key(universe: Universe | str, name: str) -> str:
    """Normalise an alias string to the canonical ``universe::snake_case`` key."""
    return build_node_id(universe, name)


@dataclass
class AliasResolver:
    """Track aliases and resolve them to canonical node IDs."""

    alias_map: Dict[str, str] = field(default_factory=dict)

    def resolve(self, universe: Universe | str, name: str) -> Optional[str]:
        """Return a canonical node ID for *name* if it is a known alias."""
        return self.alias_map.get(_alias_key(universe, name))

    def register(self, universe: Universe | str, alias: str, node_id: str) -> None:
        """Register *alias* for the given *node_id* within *universe*."""
        self.alias_map[_alias_key(universe, alias)] = node_id

    def register_node(self, node: BaseNode) -> None:
        """Register the node's name (and aliases if present) to its ID."""
        self.register(node.universe, node.name, node.id)
        aliases = getattr(node, "aliases", None)
        if aliases:
            for alias in aliases:
                self.register(node.universe, alias, node.id)

    def snapshot(self) -> Dict[str, str]:
        """Return a shallow copy of the current alias map."""
        return dict(self.alias_map)
