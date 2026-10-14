import sqlite3, json, uuid
from pathlib import Path
from contextlib import contextmanager

from core.paths import DB

@contextmanager
def con():
    c=sqlite3.connect(DB,timeout=30)
    c.row_factory=sqlite3.Row
    try:
        with c:yield c
    finally:c.close()

def init():
    DB.parent.mkdir(parents=True,exist_ok=True)
    with con() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS projects(
            id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS artifacts(
            id TEXT PRIMARY KEY,
            project_id TEXT,
            type TEXT,
            path TEXT,
            parent_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            metadata_json TEXT DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS jobs(
            id TEXT PRIMARY KEY,
            project_id TEXT,
            tool TEXT,
            model_id TEXT,
            status TEXT,
            prompt TEXT,
            params_json TEXT,
            output_artifact_id TEXT,
            error TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS conversations(
            id TEXT PRIMARY KEY,
            project_id TEXT,
            role TEXT,
            content TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS project_state(
            project_id TEXT PRIMARY KEY,
            active_artifact_id TEXT DEFAULT '',
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS recipes(
            id TEXT PRIMARY KEY,
            name TEXT,
            kind TEXT,
            model_id TEXT,
            params_json TEXT DEFAULT '{}',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cols={r["name"] for r in c.execute("PRAGMA table_info(projects)").fetchall()}
        if "description" not in cols:
            c.execute("ALTER TABLE projects ADD COLUMN description TEXT DEFAULT ''")
        if "updated_at" not in cols:
            c.execute("ALTER TABLE projects ADD COLUMN updated_at TEXT DEFAULT CURRENT_TIMESTAMP")
        if "trashed" not in cols:
            c.execute("ALTER TABLE projects ADD COLUMN trashed INTEGER DEFAULT 0")
        if "trashed_at" not in cols:
            c.execute("ALTER TABLE projects ADD COLUMN trashed_at TEXT")
        if "sort_order" not in cols:
            c.execute("ALTER TABLE projects ADD COLUMN sort_order INTEGER DEFAULT 0")
            existing=c.execute("SELECT id FROM projects ORDER BY created_at,id").fetchall()
            for idx,row in enumerate(existing,1):
                c.execute("UPDATE projects SET sort_order=? WHERE id=?",(idx*10,row["id"]))

        # v0.1.8: persistent serial queue/progress/timing fields.
        job_cols={r["name"] for r in c.execute("PRAGMA table_info(jobs)").fetchall()}
        additions={
            "phase":"TEXT DEFAULT 'queued'",
            "progress":"REAL DEFAULT 0",
            "step_current":"INTEGER DEFAULT 0",
            "step_total":"INTEGER DEFAULT 0",
            "started_at":"TEXT",
            "completed_at":"TEXT",
            "duration_seconds":"REAL DEFAULT 0",
            "queue_order":"INTEGER DEFAULT 0",
            "cancel_requested":"INTEGER DEFAULT 0"
        }
        for name,decl in additions.items():
            if name not in job_cols:
                c.execute(f"ALTER TABLE jobs ADD COLUMN {name} {decl}")
        # Job-history visibility is presentation state.  Benchmark reports, output
        # artifacts and original job records must survive clearing the Jobs view.
        if "history_hidden" not in job_cols:
            c.execute("ALTER TABLE jobs ADD COLUMN history_hidden INTEGER NOT NULL DEFAULT 0")
        if not c.execute("SELECT 1 FROM projects LIMIT 1").fetchone():
            pid=uuid.uuid4().hex
            c.execute("INSERT INTO projects(id,name,description,sort_order) VALUES(?,?,?,?)",(pid,"Default Project","",10))
            c.execute("INSERT OR IGNORE INTO project_state(project_id) VALUES(?)",(pid,))
        if not c.execute("SELECT 1 FROM recipes LIMIT 1").fetchone():
            defaults=[
                ("cinematic","Cinematic Still","image","qwen_image_21",{"ratio":"16:9","quality":"balanced","steps":28,"cfg":4.0}),
                ("photoreal","Photoreal","image","qwen_image_21",{"ratio":"3:2","quality":"maximum"}),
                ("anime","Anime Illustration","image","qwen_image_21",{"ratio":"2:3","quality":"balanced"}),
                ("logo","Brand / Typography","image","qwen_image_21",{"ratio":"1:1","quality":"balanced"}),
            ]
            for rid,name,kind,model,params in defaults:
                c.execute("INSERT INTO recipes(id,name,kind,model_id,params_json) VALUES(?,?,?,?,?)",
                          (rid,name,kind,model,json.dumps(params)))

def rows(sql,args=()):
    with con() as c:
        return [dict(x) for x in c.execute(sql,args).fetchall()]

def one(sql,args=()):
    with con() as c:
        x=c.execute(sql,args).fetchone()
        return dict(x) if x else None

def execute(sql,args=()):
    with con() as c:
        c.execute(sql,args)

def create_job(project_id,tool,model,prompt,params):
    jid=uuid.uuid4().hex
    with con() as c:
        next_order=c.execute("SELECT COALESCE(MAX(queue_order),0)+1 AS n FROM jobs").fetchone()["n"]
        c.execute("""INSERT INTO jobs(id,project_id,tool,model_id,status,phase,progress,prompt,params_json,queue_order,cancel_requested)
                     VALUES(?,?,?,?,?,?,?,?,?,?,0)""",
                  (jid,project_id,tool,model,"queued","queued",0.0,prompt,json.dumps(params,ensure_ascii=False),next_order))
        c.execute("UPDATE projects SET updated_at=CURRENT_TIMESTAMP WHERE id=?",(project_id,))
    return jid

def add_message(project_id,role,content):
    mid=uuid.uuid4().hex
    with con() as c:
        c.execute("INSERT INTO conversations(id,project_id,role,content) VALUES(?,?,?,?)",
                  (mid,project_id,role,content))
    return mid
