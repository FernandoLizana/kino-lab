from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BACKUP_VERSION = 1


def make_backup(data_dir: Path, extra: dict, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": BACKUP_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": [],
        "extra": extra,
    }
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        db = data_dir / "kino.db"
        if db.exists():
            zf.write(db, arcname="kino.db")
            manifest["files"].append(
                {"name": "kino.db", "sha256": hashlib.sha256(db.read_bytes()).hexdigest()}
            )
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
    return dest


def restore_backup(archive: Path, data_dir: Path) -> dict:
    with zipfile.ZipFile(archive, "r") as zf:
        names = zf.namelist()
        if "manifest.json" not in names:
            raise ValueError("Respaldo sin manifest.json")
        manifest = json.loads(zf.read("manifest.json"))
        if manifest.get("version") != BACKUP_VERSION:
            raise ValueError(f"Versión de respaldo no soportada: {manifest.get('version')}")
        data_dir.mkdir(parents=True, exist_ok=True)
        if "kino.db" in names:
            target = data_dir / "kino.db"
            backup_existing = data_dir / "kino.db.bak"
            if target.exists():
                target.replace(backup_existing)
            target.write_bytes(zf.read("kino.db"))
        return manifest
