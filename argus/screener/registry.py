"""Auto-discovery registry for screener strategies.

Mirrors DRUVA's scanner registry idiom: strategy classes declare themselves
with ``@register_strategy`` (reading ``slug`` off the class itself, rather
than taking it as a decorator argument, since ``Strategy.slug`` is already a
required class attribute). ``all_strategies()``/``get_strategy()`` walk
``argus.screener.strategies`` on first use so the decorators fire — callers
never need to import strategy modules by hand.
"""

from __future__ import annotations

import importlib
import pkgutil

from argus.screener.base import Strategy

_REGISTRY: dict[str, type[Strategy]] = {}
_discovered = False


def register_strategy[T: type[Strategy]](cls: T) -> T:
    slug = cls.slug
    if not slug:
        raise ValueError(f"Strategy {cls.__name__} must define a non-empty 'slug'")
    if slug in _REGISTRY:
        raise ValueError(f"Strategy slug {slug!r} is already registered")
    _REGISTRY[slug] = cls
    return cls


def _ensure_discovered() -> None:
    global _discovered
    if _discovered:
        return
    _discovered = True
    pkg = importlib.import_module("argus.screener.strategies")
    for _, module_name, _ in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg.__name__}."):
        importlib.import_module(module_name)


def get_strategy(slug: str) -> type[Strategy]:
    _ensure_discovered()
    try:
        return _REGISTRY[slug]
    except KeyError:
        raise KeyError(f"No strategy registered under slug {slug!r}") from None


def all_strategies() -> dict[str, type[Strategy]]:
    _ensure_discovered()
    return dict(_REGISTRY)
