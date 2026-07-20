# -*- coding: utf-8 -*-
"""Class-definition-time validator for JF user-hook method names.

Catches the silent-dispatch trap where a user subclass defines a method
with a name JF doesn't actually call (e.g. ``postprocess`` instead of
``post_process``), which would otherwise just be silently skipped at
runtime.

类定义时校验 user hook 方法名，避免拼写错被 framework 静默 skip。
"""
from __future__ import annotations

import difflib
import warnings
from typing import FrozenSet, Mapping


def validate_hook_names(
    cls: type,
    optional_hooks: FrozenSet[str],
    typo_map: Mapping[str, str],
    template_name: str,
    fuzzy_cutoff: float = 0.85,
) -> None:
    """Scan a subclass's own attributes for likely hook-name typos.

    扫描子类自身定义的方法名，发现疑似 hook 拼写错就报错或警告。

    Args:
        cls: The subclass being validated (passed by ``__init_subclass__``).
        optional_hooks: Names of optional JF hooks the subclass MAY
            override. Anything in this set is whitelisted.
        typo_map: Hard blacklist mapping wrong-name → correct-name.
            Names in this dict trigger ``TypeError`` immediately.
        template_name: Human-readable name of the parent template
            (e.g. ``"ModelHandlerTemplate"``) used in the error message.
        fuzzy_cutoff: ``difflib`` similarity threshold for the soft warn
            pass (0.85 = quite strict; rare false positives).

    Raises:
        TypeError: if a name in ``typo_map`` appears in ``vars(cls)``.

    Note:
        Only inspects attributes defined directly on ``cls``
        (``vars(cls)``), not inherited ones. Private names (starting
        with underscore) are skipped. Non-callable attributes are
        skipped.
    """
    own_attrs = vars(cls)
    for name in own_attrs:
        if name.startswith('_'):
            continue
        attr = own_attrs[name]
        if not callable(attr):
            continue
        # Hard blacklist: known-bad name → fail loudly with the fix.
        # 硬黑名单：踩过的拼写错直接报错。
        if name in typo_map:
            raise TypeError(
                f"{cls.__name__} defines method `{name}`, but "
                f"{template_name} dispatches `{typo_map[name]}`. "
                f"Rename to fix silent skip."
            )
        if name in optional_hooks:
            continue
        # Soft fuzzy: warn if this name is suspiciously close to a hook.
        # 软模糊：跟某个 hook 名很像就 warn。
        match = difflib.get_close_matches(
            name, optional_hooks, n=1, cutoff=fuzzy_cutoff)
        if match:
            warnings.warn(
                f"{cls.__name__}.{name} looks similar to {template_name} "
                f"hook `{match[0]}`. If unrelated, ignore; if you meant "
                f"to override the hook, rename to match.",
                stacklevel=3,
            )
