"""Shared maintenance activity inventory and safe image-queue reconciliation.

1.0.41 fixes cancelled queued image jobs that could remain persisted as ``queued``
after a restart and permanently block update/restart/uninstall checks.
"""
import contextlib, json, sqlite3
from pathlib import Path

BUSY = {'running','queued','cancelling','preparing','loading_model','downloading','validating','verifying','installing'}


def _jobs_table(con):
    return 'jobs' in {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _job_columns(con):
    return {r[1] for r in con.execute('PRAGMA table_info(jobs)')}


def reconcile_cancelled_image_jobs(home):
    """Finalize image jobs that were cancelled while still queued.

    This is safe to run while the app is online: the UPDATE only touches rows that
    are still ``queued`` and already have ``cancel_requested=1``. A worker that has
    actually started a job changes it to ``running`` first and is therefore not
    affected. The function intentionally does *not* infer that running work is
    stale; startup recovery owns that decision.
    """
    home = Path(home)
    db = home/'shared/data/sdxl_studio.sqlite3'
    if not db.is_file():
        return 0
    with contextlib.closing(sqlite3.connect(db, timeout=3)) as con:
        if not _jobs_table(con):
            return 0
        cols = _job_columns(con)
        if 'cancel_requested' not in cols:
            return 0
        cur = con.execute("""
            UPDATE jobs
               SET status='cancelled', phase='cancelled',
                   completed_at=COALESCE(completed_at,CURRENT_TIMESTAMP),
                   updated_at=CURRENT_TIMESTAMP
             WHERE status='queued' AND COALESCE(cancel_requested,0)=1
        """)
        con.commit()
        return int(cur.rowcount or 0)


def busy_jobs(home):
    home = Path(home)
    """Read active work without treating already-cancelled queued rows as busy.

    Corrupt known state still fails closed. ``running`` remains busy even when a
    cancellation was requested because the worker may still be unwinding GPU work.
    """
    result = []
    db = home/'shared/data/sdxl_studio.sqlite3'
    if db.is_file():
        with contextlib.closing(sqlite3.connect(db.resolve().as_uri()+'?mode=ro', uri=True, timeout=3)) as con:
            if _jobs_table(con):
                cols = _job_columns(con)
                if 'cancel_requested' in cols:
                    sql = """SELECT id FROM jobs
                             WHERE status='running'
                                OR status='cancelling'
                                OR (status='queued' AND COALESCE(cancel_requested,0)=0)"""
                else:
                    sql = "SELECT id FROM jobs WHERE status IN ('running','queued','cancelling')"
                result.extend('image:'+str(row[0]) for row in con.execute(sql))
    for pattern in ('shared/training/jobs/*/progress.json', 'shared/identity/jobs/*/progress.json',
                    'jobs/vision-actions/*.json', 'shared/downloads/model-hub/*.json',
                    'shared/config/instantid_code_install.json', 'shared/vision/link-imports/*.json',
                    'shared/config/sdxl_base_install.json'):
        for path in home.glob(pattern):
            d = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(d,dict):raise ValueError('Invalid job state: '+str(path.relative_to(home)))
            if d.get('status') in BUSY: result.append(str(path.relative_to(home)))
    return result
