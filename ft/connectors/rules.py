"""Behavior rules: tiny boolean expressions over connector features (spec 14.4).

    important: replied_after OR starred
    read: opened
    skip: NOT opened

Grammar (precedence NOT > AND > OR, parentheses allowed). Parsed into a tree; never eval()'d.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

TOKEN = re.compile(r"\s*(\(|\)|[A-Za-z_][A-Za-z0-9_]*)")


class RuleError(ValueError):
    pass


@dataclass
class Node:
    op: str  # "var" | "not" | "and" | "or"
    args: tuple[Any, ...]

    def eval(self, features: dict[str, bool]) -> bool:
        if self.op == "var":
            return bool(features[self.args[0]])
        if self.op == "not":
            return not self.args[0].eval(features)
        if self.op == "and":
            return all(a.eval(features) for a in self.args)
        return any(a.eval(features) for a in self.args)

    def variables(self) -> set[str]:
        if self.op == "var":
            return {self.args[0]}
        return set().union(*(a.variables() for a in self.args))


def _tokens(expr: str) -> list[str]:
    out, pos = [], 0
    expr = expr.strip()
    while pos < len(expr):
        m = TOKEN.match(expr, pos)
        if not m:
            raise RuleError(f"can't parse {expr!r} at {expr[pos:]!r}")
        out.append(m.group(1))
        pos = m.end()
    return out


def parse(expr: str) -> Node:
    toks = _tokens(expr)
    i = 0

    def peek() -> str | None:
        return toks[i] if i < len(toks) else None

    def take() -> str:
        nonlocal i
        if i >= len(toks):
            raise RuleError(f"unexpected end of rule {expr!r}")
        i += 1
        return toks[i - 1]

    def or_expr() -> Node:
        parts = [and_expr()]
        while peek() and peek().upper() == "OR":
            take()
            parts.append(and_expr())
        return parts[0] if len(parts) == 1 else Node("or", tuple(parts))

    def and_expr() -> Node:
        parts = [not_expr()]
        while peek() and peek().upper() == "AND":
            take()
            parts.append(not_expr())
        return parts[0] if len(parts) == 1 else Node("and", tuple(parts))

    def not_expr() -> Node:
        if peek() and peek().upper() == "NOT":
            take()
            return Node("not", (not_expr(),))
        return atom()

    def atom() -> Node:
        t = take()
        if t == "(":
            node = or_expr()
            if take() != ")":
                raise RuleError(f"missing ')' in {expr!r}")
            return node
        if t == ")" or t.upper() in ("AND", "OR", "NOT"):
            raise RuleError(f"unexpected {t!r} in {expr!r}")
        return Node("var", (t,))

    node = or_expr()
    if i != len(toks):
        raise RuleError(f"unexpected {toks[i]!r} in {expr!r}")
    return node


class Rules:
    """Ordered rules; the first match wins. Unknown features fail at construction, not mid-run."""

    def __init__(self, rules: list[dict[str, str]], features: set[str]):
        self.rules: list[tuple[str, Node]] = []
        for r in rules:
            if len(r) != 1:
                raise RuleError(f"each rule is one 'label: expression' pair, got {r}")
            ((label, expr),) = r.items()
            node = parse(expr)
            unknown = node.variables() - features
            if unknown:
                raise RuleError(f"rule {label!r} uses unknown features {sorted(unknown)}; known: {sorted(features)}")
            self.rules.append((label, node))

    @property
    def labels(self) -> list[str]:
        return list(dict.fromkeys(label for label, _ in self.rules))

    def label(self, features: dict[str, bool]) -> str | None:
        for label, node in self.rules:
            if node.eval(features):
                return label
        return None
