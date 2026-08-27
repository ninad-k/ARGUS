"""Built-in screener strategies.

Modules in this package must not be imported directly by application code —
``argus.screener.registry`` discovers and imports them via ``pkgutil`` on
first use of ``get_strategy``/``all_strategies``, which is what triggers each
module's ``@register_strategy`` decorator.
"""
