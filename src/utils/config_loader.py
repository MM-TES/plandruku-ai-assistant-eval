"""Loads JSON configs and resolves environment variables."""

import json
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv

# Project root is 3 levels up from this file: src/utils/config_loader.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"

load_dotenv(_PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR} placeholders with values from environment.

    Args:
        value: String that may contain ${VAR} placeholders.

    Returns:
        String with placeholders replaced by environment variable values.
    """
    def _replacer(match: re.Match) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return re.sub(r"\$\{(\w+)\}", _replacer, value)


def load_config(name: str, custom_path: Path | None = None) -> dict:
    """Read JSON config by name from config/ dir, or from custom_path.

    Args:
        name: Config filename, with or without .json extension. E.g.
            'material_predictor' or 'material_predictor.json'.
        custom_path: Optional override path. When provided, this file is read
            and `name` is retained only for log traceability. Custom paths are
            read as raw JSON without ${VAR} substitution.

    Returns:
        Parsed config dict (with ${VAR} resolved when reading the default
        config dir; raw JSON when reading from custom_path).

    Raises:
        FileNotFoundError: If neither the default path nor custom_path exists.
        json.JSONDecodeError: If the JSON file is malformed.
    """
    if custom_path is not None:
        path = Path(custom_path)
        if not path.is_file():
            raise FileNotFoundError(f"Custom config not found: {path}")
        logger.info("Loading config '%s' from custom path: %s", name, path)
        return json.loads(path.read_text(encoding="utf-8"))

    if name.endswith(".json"):
        name = name[: -len(".json")]

    path = _CONFIG_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    def _resolve(obj: object) -> object:
        if isinstance(obj, str):
            return _resolve_env_vars(obj)
        if isinstance(obj, dict):
            return {k: _resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve(item) for item in obj]
        return obj

    return _resolve(raw)


def get_db_url() -> str:
    """Build PostgreSQL connection URL from db.json and environment variables.

    Returns:
        SQLAlchemy-compatible URL: postgresql://user:password@host:port/database
    """
    cfg = load_config("db")
    return (
        f"postgresql://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )


def get_assistant_ro_db_url() -> str:
    """Build the PostgreSQL URL for the assistant's read-only role.

    Uses the ``assistant_ro_user`` / ``assistant_ro_password`` keys from db.json
    (resolved from DB_ASSISTANT_RO_USER / DB_ASSISTANT_RO_PASSWORD). The role is
    provisioned by ``sql/068_assistant_readonly_role.sql``; if it is absent the
    assistant degrades gracefully (text2SQL feature flag off).

    Returns:
        SQLAlchemy-compatible URL for the aps_assistant_ro role.
    """
    cfg = load_config("db")
    return (
        f"postgresql://{cfg['assistant_ro_user']}:{cfg['assistant_ro_password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )


def get_column_mapping(source: str) -> dict:
    """Return xlsx column index -> PG column name mapping.

    Args:
        source: Either "documents" or "orders".

    Returns:
        Dict mapping string column indices to column names.

    Raises:
        KeyError: If source is not "documents" or "orders".
    """
    return load_config("column_mapping")[source]["mapping"]


def get_preprocessing() -> dict:
    """Return preprocessing configuration.

    Returns:
        Full preprocessing.json as dict.
    """
    return load_config("preprocessing")


def get_model_params() -> dict:
    """Return print-time training config from training/configs/print_time.json.

    Source of truth since the training/-refactor cleanup pass. The legacy
    ``config/model_params.json`` fallback has been removed — the file is gone.
    """
    new_path = _PROJECT_ROOT / "training" / "configs" / "print_time.json"
    if not new_path.exists():
        raise FileNotFoundError(
            f"Print-time training config not found: {new_path}. "
            "Restore it from git or re-create from the training/README.md template."
        )
    return json.loads(new_path.read_text(encoding="utf-8"))


def load_training_config(name: str) -> dict:
    """Load a per-trainer config from training/configs/<name>.json.

    Merges _training_defaults.json into the result; per-trainer keys override
    defaults.
    """
    cfg_dir = _PROJECT_ROOT / "training" / "configs"
    defaults_path = cfg_dir / "_training_defaults.json"
    family_path = cfg_dir / f"{name}.json"
    defaults = (
        json.loads(defaults_path.read_text(encoding="utf-8"))
        if defaults_path.exists()
        else {}
    )
    family_cfg = (
        json.loads(family_path.read_text(encoding="utf-8"))
        if family_path.exists()
        else {}
    )
    merged: dict = {**defaults, **family_cfg}
    if "date_range" not in family_cfg and "date_range_default" in defaults:
        merged["date_range"] = defaults["date_range_default"]
    return merged


def get_paths() -> dict:
    """Return file/directory paths configuration.

    Returns:
        Full paths.json as dict.
    """
    return load_config("paths")
