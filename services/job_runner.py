from __future__ import annotations

import json
import threading
from typing import Any, Callable

from models import Job, db
from services.logging_setup import setup_logging

logger = setup_logging()


class JobRunner:
    """Ejecuta tareas largas en hilos daemon para no bloquear Flask."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._threads: dict[int, threading.Thread] = {}

    def create_job(self, kind: str) -> Job:
        job = Job(kind=kind, status="pending", progress=0.0, message="En cola")
        db.session.add(job)
        db.session.commit()
        return job

    def request_cancel(self, job_id: int) -> None:
        job = Job.query.get(job_id)
        if job:
            job.cancel_requested = True
            job.message = "Cancelación solicitada"
            db.session.commit()

    def request_pause(self, job_id: int) -> None:
        job = Job.query.get(job_id)
        if job:
            job.pause_requested = True
            job.message = "Pausa solicitada"
            db.session.commit()

    def clear_pause(self, job_id: int) -> None:
        job = Job.query.get(job_id)
        if job:
            job.pause_requested = False
            db.session.commit()

    def _update(self, job_id: int, **fields: Any) -> None:
        job = Job.query.get(job_id)
        if not job:
            return
        for key, value in fields.items():
            setattr(job, key, value)
        db.session.commit()

    def run(self, app, job_id: int, fn: Callable[[Job, Callable, Callable], dict]) -> None:
        def worker():
            with app.app_context():
                self._update(job_id, status="running", progress=0.0, message="Iniciando")

                def progress_cb(pct: float, message: str = "") -> None:
                    self._update(
                        job_id,
                        progress=float(pct),
                        message=message or "Procesando",
                    )

                def should_cancel() -> bool:
                    job = Job.query.get(job_id)
                    return bool(job and job.cancel_requested)

                def should_pause() -> bool:
                    job = Job.query.get(job_id)
                    return bool(job and getattr(job, "pause_requested", False))

                try:
                    job = Job.query.get(job_id)
                    try:
                        result = fn(job, progress_cb, should_cancel, should_pause)
                    except TypeError:
                        result = fn(job, progress_cb, should_cancel)
                    if isinstance(result, dict) and result.get("status") == "paused":
                        self._update(
                            job_id,
                            status="paused",
                            message="Pausado; se conservó el avance",
                            checkpoint_json=json.dumps(result.get("checkpoint", {}), default=str),
                            result_json=json.dumps(result, ensure_ascii=False, default=str),
                        )
                        return
                    status = "cancelled" if should_cancel() else "done"
                    self._update(
                        job_id,
                        status=status,
                        progress=100.0 if status == "done" else float(Job.query.get(job_id).progress),
                        message="Completado" if status == "done" else "Cancelado",
                        result_json=json.dumps(result, ensure_ascii=False, default=str),
                    )
                except Exception as exc:
                    logger.exception("Job %s failed", job_id)
                    self._update(
                        job_id,
                        status="error",
                        message=str(exc),
                        result_json=json.dumps({"error": str(exc)}),
                    )

        thread = threading.Thread(target=worker, daemon=True, name=f"job-{job_id}")
        with self._lock:
            self._threads[job_id] = thread
        thread.start()


job_runner = JobRunner()
