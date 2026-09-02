"""
🏋️  Background training queue.

Training runs a real grid search (up to ~1,440 XGBoost parameter
combinations x 5 CV folds x multiple lag/rolling configs) — that can take
anywhere from seconds (QUICK_TRAIN) to several minutes for real production
data. Running that synchronously inside a request handler means:
  - the HTTP client has to hold a connection open the whole time (many
    proxies/load balancers time that out well before training finishes)
  - N concurrent /train calls (from N users, or one bad actor) spin up N
    full grid searches at once and can take the whole server down

Fix: POST /train now just validates the upload, enqueues a job, and
returns immediately with a job_id. A small bounded worker pool
(MAX_CONCURRENT_TRAINING_JOBS, default 1) pulls jobs off the queue and
runs them one at a time (or a few at a time) in a background thread, so
the server always stays responsive no matter how many training requests
come in. The frontend polls GET /train/status/{job_id} until it's done.

Job records live in memory (self-hosted, single-process) AND get mirrored
to Firestore (users/{uid}/jobs/{job_id}) so a job's outcome isn't lost if
you restart the server mid-poll — though a job that was actually
*running* when the server restarted is marked "interrupted" and must be
resubmitted, since the in-memory queue itself doesn't survive a restart.

Scaling note: this in-process queue is correct for a single server
instance. If you outgrow one process, swap this for a real task queue
(Celery/RQ/Dramatiq with Redis) — the job_id/status contract for the
frontend stays identical.
"""
import asyncio
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum

import pandas as pd

logger = logging.getLogger("forecastiq")


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class TrainingJobQueue:
    def __init__(self, max_concurrent: int, db=None, on_complete=None):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._jobs: dict[str, dict] = {}
        self._db = db
        self._workers_started = False
        # Optional callable(uid, bundle) invoked right after a job finishes
        # successfully — main.py wires this up to save the trained model.
        self.on_complete = on_complete

    def start_workers(self, num_workers: int):
        if self._workers_started:
            return
        for _ in range(num_workers):
            asyncio.create_task(self._worker_loop())
        self._workers_started = True
        logger.info(f"Training job queue started with {num_workers} worker(s)")

    async def _worker_loop(self):
        while True:
            job_id = await self._queue.get()
            async with self._semaphore:
                await self._run_job(job_id)
            self._queue.task_done()

    def submit(self, uid: str, owner_email: str, df: pd.DataFrame) -> str:
        job_id = uuid.uuid4().hex
        self._jobs[job_id] = {
            "job_id": job_id,
            "uid": uid,
            "owner_email": owner_email,
            "status": JobStatus.QUEUED,
            "submitted_at": datetime.now(timezone.utc),
            "df": df,
            "result": None,
            "error": None,
        }
        # NOTE: deliberately NOT mirroring to Firestore here. This method
        # runs directly inside the POST /train request/response cycle, so
        # it must return instantly — any network call here (including
        # Firestore) would defeat the entire point of queuing (the request
        # would block exactly like the old synchronous /train did). The
        # worker loop mirrors status to Firestore once it actually starts
        # processing the job, off the request path.
        self._queue.put_nowait(job_id)
        return job_id

    async def _run_job(self, job_id: str):
        from Smart_Za3bola import train_on_df  # local import avoids a circular import with main.py

        job = self._jobs[job_id]
        job["status"] = JobStatus.RUNNING
        job["started_at"] = datetime.now(timezone.utc)
        self._mirror_async(job_id)  # fire-and-forget — never delays training itself
        logger.info(f"Training job {job_id} started for uid={job['uid']}")

        try:
            bundle = await asyncio.to_thread(train_on_df, job["df"])
            bundle["owner_uid"] = job["uid"]
            bundle["owner_email"] = job.get("owner_email", "unknown")
            job["status"] = JobStatus.DONE
            job["result"] = bundle
            job["finished_at"] = datetime.now(timezone.utc)
            logger.info(f"Training job {job_id} completed successfully")
            if self.on_complete:
                try:
                    await asyncio.to_thread(self.on_complete, job["uid"], bundle)
                except Exception as save_exc:
                    job["status"] = JobStatus.FAILED
                    job["error"] = f"Training succeeded but saving the model failed: {save_exc}"
                    logger.exception(f"Training job {job_id}: on_complete save failed: {save_exc}")
        except Exception as exc:
            job["status"] = JobStatus.FAILED
            job["error"] = str(exc)
            job["finished_at"] = datetime.now(timezone.utc)
            logger.exception(f"Training job {job_id} failed: {exc}")
        finally:
            # Don't keep the raw training DataFrame around once we're done with it.
            job.pop("df", None)
            self._mirror_async(job_id, include_result_metrics_only=True)

    def _mirror_async(self, job_id: str, include_result_metrics_only: bool = False):
        """
        Fire-and-forget Firestore status mirror. Deliberately NOT awaited by
        the caller: mirroring is a nice-to-have (lets you see job progress
        from another device/process) and must never be allowed to slow down
        or block the actual training work if Firestore is slow/unreachable.
        Errors are caught and logged inside _mirror_to_firestore itself.
        """
        asyncio.create_task(asyncio.to_thread(self._mirror_to_firestore, job_id, include_result_metrics_only))

    def get_status(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        if not job:
            return None
        public = {
            "job_id": job["job_id"],
            "status": job["status"],
            "submitted_at": job["submitted_at"].isoformat(),
        }
        if job.get("started_at"):
            public["started_at"] = job["started_at"].isoformat()
        if job.get("finished_at"):
            public["finished_at"] = job["finished_at"].isoformat()
        if job["status"] == JobStatus.DONE and job["result"]:
            metrics = job["result"]["metrics"]
            public["metrics"] = {
                "model_name":    metrics["model_name"],
                "train_rmse":    round(metrics["train_rmse"], 4),
                "val_rmse":      round(metrics["val_rmse"], 4),
                "baseline_rmse": round(metrics["baseline_rmse"], 4),
                "best_lags":     job["result"]["lags"],
                "best_roll":     job["result"]["roll"],
            }
        if job["status"] == JobStatus.FAILED:
            public["error"] = job["error"]
        return public

    def get_result_bundle(self, job_id: str) -> dict | None:
        """Internal use only — the raw trained bundle, for saving to model storage."""
        job = self._jobs.get(job_id)
        return job["result"] if job else None

    def owns(self, job_id: str, uid: str) -> bool:
        job = self._jobs.get(job_id)
        return bool(job) and job["uid"] == uid

    def _mirror_to_firestore(self, job_id: str, include_result_metrics_only: bool = False):
        if self._db is None:
            return
        job = self._jobs.get(job_id)
        if not job:
            return
        try:
            doc = {
                "status": job["status"],
                "submitted_at": job["submitted_at"],
            }
            if job.get("started_at"):
                doc["started_at"] = job["started_at"]
            if job.get("finished_at"):
                doc["finished_at"] = job["finished_at"]
            if job["status"] == JobStatus.FAILED:
                doc["error"] = job["error"]
            if include_result_metrics_only and job.get("result"):
                doc["metrics"] = job["result"]["metrics"]
            self._db.collection("users").document(job["uid"]).collection("jobs").document(job_id).set(doc, merge=True)
        except Exception as exc:
            logger.warning(f"Failed to mirror job {job_id} status to Firestore (non-fatal): {exc}")


# Populated by main.py at startup once `db` and settings are available.
training_queue: TrainingJobQueue | None = None
