"""Bounded persistent login reservations. No passwords, usernames or raw IPs stored."""
from __future__ import annotations
import contextlib, hashlib, math, sqlite3, time
from pathlib import Path
from core.private_files import private_directory,protect

class LoginThrottle:
    def __init__(self,path,clock=time.time,window=60,per_peer=8,total=60):
        self.path=Path(path);self.clock=clock;self.window=window;self.per_peer=per_peer;self.total=total
    def reserve(self,peer):
        private_directory(self.path.parent)
        if not self.path.exists():
            try:
                with self.path.open('xb'):pass
            except FileExistsError:pass
        protect(self.path)
        now=float(self.clock());key=hashlib.sha256(str(peer or 'unknown').encode('utf-8')).hexdigest()
        with contextlib.closing(sqlite3.connect(self.path,timeout=3)) as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('CREATE TABLE IF NOT EXISTS attempts (stamp REAL NOT NULL, peer TEXT NOT NULL)')
            db.execute('DELETE FROM attempts WHERE stamp <= ? OR stamp > ?', (now-self.window,now+self.window))
            events=list(db.execute('SELECT stamp,peer FROM attempts ORDER BY stamp'))
            same=[stamp for stamp,k in events if k==key]
            wait=0
            if len(events)>=self.total:wait=max(wait,math.ceil(events[0][0]+self.window-now))
            if len(same)>=self.per_peer:wait=max(wait,math.ceil(same[0]+self.window-now))
            if not wait:db.execute('INSERT INTO attempts VALUES (?,?)',(now,key))
            db.commit()
        return max(0,wait)

    def success(self,peer):
        key=hashlib.sha256(str(peer or 'unknown').encode('utf-8')).hexdigest()
        with contextlib.closing(sqlite3.connect(self.path,timeout=3)) as db:
            db.execute('DELETE FROM attempts WHERE peer=?',(key,));db.commit()
