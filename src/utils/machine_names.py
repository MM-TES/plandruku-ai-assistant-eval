"""Довідник коротких назв друкарського обладнання.

Три форми імені для кожної активної машини:
  * ``db`` — повна назва з ``aps.documents."11_equip_name"``
    (наприклад, ``Участок гл.печати ROTOMEC 2``).
  * ``medium`` — без префіксу ділянки
    (``ROTOMEC 2``, ``FISCHER 13 (S-2)``). Для веб-таблиць і Excel-колонок
    де місця достатньо.
  * ``short`` — 3-символьний код (``R-2``, ``S-2``). Для pivot-заголовків,
    Gantt-lanes та інших щільних таблиць.

Source of truth — ``config/ml_active_equipment.json`` (``active_equipment_names``).
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import NamedTuple

from src.utils.config_loader import load_config

_DEPT_PREFIXES = (
    "Участок гл.печати ",
    "Уч.печати ",
)
_FISCHER_SHORT_RE = re.compile(r"\((S-\d+)\)")
_ROTOMEC_SHORT_RE = re.compile(r"ROTOMEC\s+(\d+)", re.IGNORECASE)


class MachineNames(NamedTuple):
    db: str
    medium: str
    short: str


def _derive(db_name: str) -> MachineNames:
    medium = db_name
    for prefix in _DEPT_PREFIXES:
        if medium.startswith(prefix):
            medium = medium[len(prefix):]
            break

    m = _FISCHER_SHORT_RE.search(medium)
    if m:
        return MachineNames(db=db_name, medium=medium, short=m.group(1))
    m = _ROTOMEC_SHORT_RE.search(medium)
    if m:
        return MachineNames(db=db_name, medium=medium, short=f"R-{m.group(1)}")
    return MachineNames(db=db_name, medium=medium, short=medium)


@lru_cache(maxsize=1)
def _registry() -> dict[str, MachineNames]:
    cfg = load_config("ml_active_equipment")
    db_names: list[str] = list(cfg.get("active_equipment_names", []) or [])
    out: dict[str, MachineNames] = {}
    for db in db_names:
        info = _derive(db)
        out[info.db] = info
        out[info.medium] = info
        out[info.short] = info
    return out


def medium_name(name: str | None) -> str:
    """Будь-яка форма → medium. Невідома назва → derived fallback."""
    if not name:
        return ""
    s = str(name).strip()
    if not s:
        return ""
    info = _registry().get(s)
    if info is not None:
        return info.medium
    return _derive(s).medium


def short_name(name: str | None) -> str:
    """Будь-яка форма → short. Невідома назва → derived fallback."""
    if not name:
        return ""
    s = str(name).strip()
    if not s:
        return ""
    info = _registry().get(s)
    if info is not None:
        return info.short
    return _derive(s).short


def all_machines() -> list[MachineNames]:
    """Всі активні машини в порядку з ml_active_equipment.json."""
    cfg = load_config("ml_active_equipment")
    return [_derive(n) for n in cfg.get("active_equipment_names", []) or []]


def reset_cache() -> None:
    """Скинути кеш реєстру (для тестів, що переналаштовують конфіг)."""
    _registry.cache_clear()
