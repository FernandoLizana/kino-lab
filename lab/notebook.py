from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

SCHEMA = "kino-experiment-v1"


def git_meta(root: Path) -> dict:
    meta = {"commit": None, "dirty": None}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
        )
        meta["commit"] = commit
        meta["dirty"] = bool(dirty.strip())
    except Exception:
        pass
    return meta


def new_experiment(payload: dict, root: Path) -> dict:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "schema": SCHEMA,
        "experiment_id": payload.get("experiment_id") or str(uuid4()),
        "title": payload.get("title") or "Sin título",
        "question": payload.get("question") or "",
        "hypothesis": payload.get("hypothesis") or "",
        "notes": payload.get("notes") or "",
        "conclusion": payload.get("conclusion") or "",
        "status": payload.get("status") or "draft",
        "created_at": now,
        "updated_at": now,
        "data_fingerprint": payload.get("data_fingerprint"),
        "profile": payload.get("profile"),
        "git": git_meta(root),
        "engine": "lab/1.0.0",
        "runs": [],
    }


def new_run(experiment: dict, config: dict, result: dict) -> dict:
    run = {
        "run_id": str(uuid4()),
        "experiment_id": experiment["experiment_id"],
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": result.get("status", "done"),
        "config": config,
        "result": result,
        "kind": result.get("kind"),
    }
    experiment.setdefault("runs", []).append(run)
    experiment["updated_at"] = run["started_at"]
    experiment["status"] = "has_runs"
    return run


def validate_import(raw: str | bytes) -> dict:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("El JSON debe ser un objeto.")
    if data.get("schema") != SCHEMA:
        raise ValueError(f"Schema no soportado: {data.get('schema')}")
    if "experiment_id" not in data or "title" not in data:
        raise ValueError("Faltan experiment_id o title.")
    # Nunca ejecutar campos como código.
    return data
