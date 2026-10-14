"""Small additive UI endpoints. All routes use the parent login/origin guards."""
import json, re, secrets, tempfile, uuid, shutil
from pathlib import Path
from flask import request, jsonify, session, send_file
from .video_picker import VideoPicker
from .runtime_env import SHARED_ROOT
from .paths import ARTIFACTS_DIR

def register(app,one,execute,decode_artifact):
    videos=VideoPicker(SHARED_ROOT/'cache/video-picker')
    def owner():
        if not session.get('video_owner'):session['video_owner']=secrets.token_hex(24)
        return session['video_owner']
    @app.post('/api/image-tools/video-picker')
    def video_picker_upload():
        pid=request.form.get('project_id','');f=request.files.get('file')
        if not f or not one('SELECT id FROM projects WHERE id=?',(pid,)):return jsonify(error='Open a project and choose a video'),400
        try:return jsonify(ok=True,video=videos.upload(f.stream,f.filename,pid,owner()))
        except ValueError as e:return jsonify(error=str(e)),400
    @app.post('/api/image-tools/video-picker/<vid>/preview')
    def video_picker_preview(vid):
        d=request.get_json() or {}
        try:return jsonify(ok=True,preview=videos.preview(vid,owner(),d.get('mode','timestamp'),d.get('timestamp',0)))
        except ValueError as e:return jsonify(error=str(e)),400
    @app.get('/api/image-tools/video-picker/<vid>/preview/<preview_id>')
    def video_picker_png(vid,preview_id):
        try:
            path,_,_=videos.selected(vid,owner(),preview_id)
            r=send_file(path,mimetype='image/png');r.headers['Cache-Control']='no-store';return r
        except ValueError as e:return jsonify(error=str(e)),404
    @app.post('/api/image-tools/video-picker/<vid>/save')
    def video_picker_save(vid):
        d=request.get_json() or {};aid=uuid.uuid4().hex;dest=None
        try:
            with videos.lock:
                src,meta,info=videos.selected(vid,owner(),d.get('preview_id',''));pid=meta['project_id']
                if not one('SELECT id FROM projects WHERE id=?',(pid,)):raise ValueError('Project no longer exists')
                dest=ARTIFACTS_DIR/pid/(aid+'.png');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
            m={'source':'media_tools','operation':'extract_frame','original_name':meta['filename']+' · frame','extraction':info,'width':info['width'],'height':info['height']}
            execute('INSERT INTO artifacts(id,project_id,type,path,parent_id,metadata_json) VALUES(?,?,?,?,?,?)',(aid,pid,'image',str(dest),None,json.dumps(m)))
            return jsonify(ok=True,artifact=decode_artifact(one('SELECT * FROM artifacts WHERE id=?',(aid,))))
        except (ValueError,OSError) as e:
            if dest:dest.unlink(missing_ok=True)
            return jsonify(error=str(e)),400
    @app.delete('/api/image-tools/video-picker/<vid>')
    def video_picker_discard(vid):
        try:videos.discard(vid,owner());return jsonify(ok=True)
        except ValueError as e:return jsonify(error=str(e)),404
