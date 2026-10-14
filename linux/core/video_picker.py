"""Short-lived, authenticated video preview storage for selecting an actual PNG."""
import json, math, re, shutil, subprocess, threading, time, uuid
from pathlib import Path
from .frame_extract import extract_frame, FORMATS
from .runtime_env import atomic_json
from core.platform_support import hidden_kwargs

class VideoPicker:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.lock=threading.Lock()
    def cleanup(self):
        for folder in self.root.iterdir():
            if folder.is_dir() and time.time()-folder.stat().st_mtime>7200:shutil.rmtree(folder,ignore_errors=True)
    def get(self,vid,owner):
        if not re.fullmatch('[a-f0-9]{32}',str(vid)):raise ValueError('Video session not found')
        folder=self.root/vid
        try:meta=json.loads((folder/'meta.json').read_text())
        except (OSError,ValueError):raise ValueError('Video session expired; upload again') from None
        if meta['owner']!=owner or time.time()-meta['created_at']>7200:raise ValueError('Video session expired; upload again')
        return folder,meta
    def upload(self,stream,name,pid,owner):
        self.cleanup()
        if sum(f.stat().st_size for f in self.root.rglob('input.*'))>1024**3:raise ValueError('Temporary video storage is full')
        ext=Path(name).suffix.lower()
        if ext not in {'.mp4','.mov','.webm','.mkv','.avi'}:raise ValueError('Choose MP4, MOV, MKV, WebM or AVI')
        if not shutil.which('ffprobe') or not shutil.which('ffmpeg'):raise ValueError('FFmpeg and ffprobe are required')
        vid=uuid.uuid4().hex;folder=self.root/vid;folder.mkdir()
        try:
            total=0;source=folder/('input'+ext)
            with source.open('wb') as f:
                while True:
                    chunk=stream.read(1024**2)
                    if not chunk:break
                    total+=len(chunk)
                    if total>120*1024**2:raise ValueError('Video limit: 120 MiB')
                    f.write(chunk)
            args=['ffprobe','-v','error','-protocol_whitelist','file,pipe','-format_whitelist',FORMATS,'-select_streams','v:0','-show_entries','stream=width,height,duration:format=duration','-of','json',str(source)]
            r=subprocess.run(args,check=True,capture_output=True,timeout=20, **hidden_kwargs());p=json.loads(r.stdout);s=p['streams'][0]
            duration=float(s.get('duration') or p.get('format',{}).get('duration') or 0);w=int(s.get('width',0));h=int(s.get('height',0))
            if not 0<duration<=3600 or not math.isfinite(duration) or not 0<w*h<=16000000:raise ValueError('Use a video up to one hour and 16 megapixels')
            meta={'id':vid,'project_id':pid,'owner':owner,'filename':Path(name).name,'input':source.name,'duration':duration,'width':w,'height':h,'created_at':time.time()}
            atomic_json(folder/'meta.json',meta)
            return {k:v for k,v in meta.items() if k not in {'owner','input'}}
        except Exception as e:
            shutil.rmtree(folder,ignore_errors=True)
            if isinstance(e,ValueError):raise
            raise ValueError('Invalid or unsupported video') from None
    def preview(self,vid,owner,mode,timestamp):
        folder,meta=self.get(vid,owner)
        if not self.lock.acquire(False):raise ValueError('A video preview is running; retry in a moment')
        try:
            for p in folder.glob('*.png'):p.unlink(missing_ok=True)
            preview_id=uuid.uuid4().hex;out=folder/(preview_id+'.png')
            info=extract_frame(folder/meta['input'],out,mode,timestamp)
            from PIL import Image
            with Image.open(out) as im:info.update(width=im.width,height=im.height)
            atomic_json(folder/'preview.json',{'id':preview_id,'info':info})
            return {'id':preview_id,'info':info}
        finally:self.lock.release()
    def selected(self,vid,owner,preview_id):
        folder,meta=self.get(vid,owner)
        try:p=json.loads((folder/'preview.json').read_text())
        except (OSError,ValueError):raise ValueError('Preview a frame first') from None
        if p['id']!=preview_id:raise ValueError('Selection changed; preview again')
        return folder/(preview_id+'.png'),meta,p['info']
    def discard(self,vid,owner):
        with self.lock:
            folder,_=self.get(vid,owner);shutil.rmtree(folder,ignore_errors=True)
