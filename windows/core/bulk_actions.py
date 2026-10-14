"""Scoped bulk operations for Library and Jobs (1.0.62).

These actions keep the existing individual-job and individual-artifact semantics.
A selection never crosses project boundaries; hide only changes Library visibility.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from core.db import con
from core.paths import ARTIFACTS_DIR, UPLOADS_DIR
from core.job_management import JobControlError, TERMINAL, hide_terminal, delete_terminal_record

_ID=re.compile(r'^[a-f0-9]{32}$')
MAX_SELECTION=400


def validate_ids(ids):
    if not isinstance(ids,list) or not (1<=len(ids)<=MAX_SELECTION):
        raise JobControlError(f'Select between 1 and {MAX_SELECTION} items.',400)
    if any(not isinstance(i,str) or not _ID.fullmatch(i) for i in ids):
        raise JobControlError('Invalid selected item ID.',400)
    if len(set(ids))!=len(ids):
        raise JobControlError('Duplicate items in selection.',400)
    return ids


def _owned_rows(c,table,pid,ids):
    placeholders=','.join('?' for _ in ids)
    found={r['id']:dict(r) for r in c.execute(
        f'SELECT * FROM {table} WHERE project_id=? AND id IN ({placeholders})',
        (pid,*ids)).fetchall()}
    if len(found)!=len(ids):
        raise JobControlError('Selection contains items outside this project or no longer present. Refresh and retry.',409)
    return found


def _user_owned_file(path,pid):
    if not path:return None
    p=Path(path).resolve()
    for root in (ARTIFACTS_DIR,UPLOADS_DIR):
        try:
            p.relative_to((root/pid).resolve())
            return p
        except ValueError:pass
    raise JobControlError("File is outside this project\'s managed Library storage; deletion refused.",409)


def library_bulk(pid,ids,action):
    ids=validate_ids(ids)
    if action not in ('hide','restore','delete'):
        raise JobControlError('Unsupported Library action.',400)
    with con() as c:
        c.execute('BEGIN IMMEDIATE')
        found=_owned_rows(c,'artifacts',pid,ids)
        if action in ('hide','restore'):
            for aid in ids:
                row=found[aid]
                try:metadata=json.loads(row.get('metadata_json') or '{}')
                except (ValueError,TypeError):metadata={}
                if not isinstance(metadata,dict):metadata={}
                if action=='hide':metadata['library_hidden']=True
                else:metadata.pop('library_hidden',None)
                c.execute('UPDATE artifacts SET metadata_json=? WHERE id=? AND project_id=?',
                          (json.dumps(metadata,ensure_ascii=False),aid,pid))
            return {'processed':len(ids),'errors':[],'action':action}
        # Validate ALL file locations before deleting ANY; no external model files.
        targets={aid:_user_owned_file(found[aid].get('path'),pid) for aid in ids}
        errors=[];done=0
        for aid in ids:
            p=targets[aid]
            try:
                if p is not None and p.exists():
                    if not p.is_file():raise OSError('Not a regular file')
                    p.unlink()
                c.execute('UPDATE jobs SET output_artifact_id=NULL WHERE project_id=? AND output_artifact_id=?',(pid,aid))
                c.execute("UPDATE project_state SET active_artifact_id='' WHERE project_id=? AND active_artifact_id=?",(pid,aid))
                c.execute('DELETE FROM artifacts WHERE id=? AND project_id=?',(aid,pid))
                done+=1
            except OSError as exc:errors.append({'id':aid,'error':str(exc)})
        return {'processed':done,'errors':errors,'action':action}


def jobs_bulk(pid,ids,action,cancel_job):
    ids=validate_ids(ids)
    if action not in ('hide','delete','cancel'):
        raise JobControlError('Unsupported Jobs action.',400)
    # Validate all memberships before any individual action.
    with con() as c:
        found=_owned_rows(c,'jobs',pid,ids)
    processed=0;errors=[]
    for jid in ids:
        row=found[jid]
        try:
            if action=='hide':hide_terminal(pid,jid)
            elif action=='delete':delete_terminal_record(pid,jid)
            else:
                if row['status'] not in ('queued','running','cancelling'):
                    raise JobControlError('This job is already finished.',409)
                cancel_job(jid)
            processed+=1
        except (JobControlError,ValueError) as exc:
            errors.append({'id':jid,'error':str(exc)})
    return {'processed':processed,'errors':errors,'action':action}
