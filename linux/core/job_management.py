"""Project-scoped, non-destructive job controls for Image Lab.

All mutations are checked inside a SQLite write transaction.  The global GPU
queue remains single-worker and queued order is resolved from the database.
Hiding history never deletes benchmark jobs or generated images.
"""
from __future__ import annotations
import json
from core.db import con
from core.qwen_request import effective_prompt

TERMINAL = ('completed', 'failed', 'cancelled', 'interrupted')

class JobControlError(ValueError):
    def __init__(self, message, code=400):
        super().__init__(message)
        self.code=code


def require_project(pid):
    if not isinstance(pid,str) or not pid:
        raise JobControlError('Select a project first.',400)
    with con() as c:
        row=c.execute('SELECT id FROM projects WHERE id=? AND COALESCE(trashed,0)=0',(pid,)).fetchone()
    if not row:raise JobControlError('Project not found.',404)


def _job(c,pid,jid):
    row=c.execute('SELECT * FROM jobs WHERE id=? AND project_id=?',(jid,pid)).fetchone()
    if not row:raise JobControlError('Job not found in this project.',404)
    return dict(row)


def change_prompt(pid,jid,text):
    """Only a pending, non-benchmark text prompt can be edited.

    Component snapshots and all inference settings remain untouched.
    """
    if not isinstance(text,str):raise JobControlError('Prompt must be text.')
    with con() as c:
        c.execute('BEGIN IMMEDIATE')
        row=_job(c,pid,jid)
        if row['status']!='queued' or int(row.get('cancel_requested') or 0):
            raise JobControlError('Only queued jobs can be edited.',409)
        p=json.loads(row.get('params_json') or '{}')
        if p.get('_benchmark_id') or p.get('prompt_enhancer') or p.get('_prompt_enhancer') or row.get('tool') in ('qwen_prompt_enhance','qwen_decode','qwen_roundtrip','qwen_vae_test'):
            raise JobControlError('This job belongs to a controlled test or tool; edit the original request instead.',409)
        try:effective=effective_prompt(text,bool(p.get('transparent_output')))
        except ValueError as e:raise JobControlError(str(e),400) from e
        p['user_prompt']=text.strip()
        c.execute("UPDATE jobs SET prompt=?,params_json=?,updated_at=CURRENT_TIMESTAMP WHERE id=? AND project_id=? AND status='queued'",
                  (effective,json.dumps(p,ensure_ascii=False),jid,pid))
    return {'id':jid,'prompt':effective}


def move_queued(pid,jid,direction):
    """Swap queue priorities with adjacent queued job of the SAME project.

    An unrelated project's queued jobs are never edited. Global chronological
    GPU dispatch still uses queue_order across every project.
    """
    if direction not in ('up','down'):raise JobControlError('Direction must be up or down.')
    with con() as c:
        c.execute('BEGIN IMMEDIATE')
        row=_job(c,pid,jid)
        if row['status']!='queued' or int(row.get('cancel_requested') or 0):
            raise JobControlError('Only queued jobs can be reordered.',409)
        items=c.execute("SELECT id,queue_order FROM jobs WHERE project_id=? AND status='queued' AND COALESCE(cancel_requested,0)=0 ORDER BY queue_order,created_at,id",(pid,)).fetchall()
        ids=[x['id'] for x in items]; i=ids.index(jid); j=i+(-1 if direction=='up' else 1)
        if j<0 or j>=len(items):return {'changed':False,'id':jid}
        a,b=items[i],items[j]
        c.execute('UPDATE jobs SET queue_order=?,updated_at=CURRENT_TIMESTAMP WHERE id=? AND project_id=?',(b['queue_order'],a['id'],pid))
        c.execute('UPDATE jobs SET queue_order=?,updated_at=CURRENT_TIMESTAMP WHERE id=? AND project_id=?',(a['queue_order'],b['id'],pid))
    return {'changed':True,'id':jid}


def hide_terminal(pid,jid):
    with con() as c:
        c.execute('BEGIN IMMEDIATE')
        row=_job(c,pid,jid)
        if row['status'] not in TERMINAL:
            raise JobControlError('Cancel the job and wait until it stops before removing it from history.',409)
        c.execute('UPDATE jobs SET history_hidden=1,updated_at=CURRENT_TIMESTAMP WHERE id=? AND project_id=?',(jid,pid))
    return {'hidden':True,'id':jid}


def clear_terminal(pid):
    with con() as c:
        c.execute('BEGIN IMMEDIATE')
        cur=c.execute("UPDATE jobs SET history_hidden=1,updated_at=CURRENT_TIMESTAMP WHERE project_id=? AND status IN ('completed','failed','cancelled','interrupted') AND COALESCE(history_hidden,0)=0",(pid,))
        n=cur.rowcount
    return {'hidden':n}


def restore_history(pid):
    with con() as c:
        c.execute('BEGIN IMMEDIATE')
        cur=c.execute('UPDATE jobs SET history_hidden=0,updated_at=CURRENT_TIMESTAMP WHERE project_id=? AND COALESCE(history_hidden,0)=1',(pid,))
        n=cur.rowcount
    return {'restored':n}


def delete_terminal_record(pid,jid):
    """Delete one finished, ordinary job record (not a generated image).

    Benchmark job IDs must remain in the database for historic report comparisons
    and retry; they can only be hidden in Jobs, or deleted from Benchmark Lab.
    """
    with con() as c:
        c.execute('BEGIN IMMEDIATE')
        row=_job(c,pid,jid)
        if row['status'] not in TERMINAL:
            raise JobControlError('Wait until the job is finished before deleting its record.',409)
        params=json.loads(row.get('params_json') or '{}')
        if params.get('_benchmark_id'):
            raise JobControlError('This job is linked to a benchmark; hide it here or manage it from Benchmark Lab.',409)
        c.execute('DELETE FROM jobs WHERE project_id=? AND id=?',(pid,jid))
    # Original PNG, linked inputs, cached latents and other project data remain.
    # Progress/cancel markers are small caches and can be deleted when safe.
    from core.paths import SHARED_ROOT
    for path in (SHARED_ROOT/'cache'/'job_progress'/(jid+'.json'),SHARED_ROOT/'cache'/'job_cancel'/(jid+'.cancel')):
        try:path.unlink(missing_ok=True)
        except OSError:pass
    return {'deleted':True,'id':jid,'artifacts_kept':True}
