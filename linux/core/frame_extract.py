"""Optional local FFmpeg image extraction. No video generator/runtime required."""
import json, math, shutil, subprocess
from pathlib import Path
from PIL import Image
from core.platform_support import hidden_kwargs

FORMATS='mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi'

def timestamp_seconds(value):
    text=str(value).strip()
    try:
        parts=[float(x) for x in text.split(':')]
        if len(parts)>3 or not parts or any(not math.isfinite(x) or x<0 for x in parts):raise ValueError()
        if len(parts)>1 and any(x>=60 for x in parts[1:]):raise ValueError()
        result=sum(x*60**i for i,x in enumerate(reversed(parts)))
        if result>86400:raise ValueError()
        return result
    except (TypeError,ValueError):raise ValueError('Usa secondi (es. 1.5) oppure HH:MM:SS.mmm.') from None

def extract_frame(source,destination,mode='first',timestamp=0):
    ffmpeg=shutil.which('ffmpeg');ffprobe=shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:raise ValueError('Installa FFmpeg e ffprobe con il gestore pacchetti Linux. Nessun modello AI è necessario.')
    if mode not in ('first','last','timestamp'):raise ValueError('Scegli primo frame, ultimo frame o istante.')
    safety=['-protocol_whitelist','file,pipe','-format_whitelist',FORMATS]
    try:
        probe=subprocess.run([ffprobe,'-v','error',*safety,'-select_streams','v:0','-show_entries','stream=width,height,duration:format=duration','-of','json',str(source)],capture_output=True,timeout=20,check=True, **hidden_kwargs())
        data=json.loads(probe.stdout);stream=data.get('streams',[{}])[0];w=int(stream.get('width') or 0);h=int(stream.get('height') or 0)
        duration=float(stream.get('duration') or data.get('format',{}).get('duration') or 0)
        if not 0<w*h<=16000000:raise ValueError('Video non valido o risoluzione oltre 16 megapixel.')
        if not math.isfinite(duration) or duration<0 or duration>3600:raise ValueError('Usa un video di durata non superiore a un’ora.')
        args=[ffmpeg,'-hide_banner','-loglevel','error','-y','-nostdin']
        seek=0
        if mode=='timestamp':
            seek=timestamp_seconds(timestamp)
            if duration and seek>=duration:raise ValueError('L’istante scelto è oltre la fine del video.')
        elif mode=='last' and duration>5:seek=duration-5
        if seek:args+=['-ss',str(seek)]
        args+=safety+['-i',str(source),'-map','0:v:0','-an','-sn','-dn']
        if mode=='last':
            # Overwrite the output with EVERY decoded frame in the tail; the retained
            # image is the final decoded frame, not duration minus a guessed interval.
            args+=['-vsync','0','-update','1']
        else:args+=['-frames:v','1']
        args+=[str(destination)]
        subprocess.run(args,capture_output=True,timeout=180,check=True, **hidden_kwargs())
        with Image.open(destination) as image:
            image.verify()
        return {'mode':mode,'timestamp':seek if mode=='timestamp' else None,'width':w,'height':h,'source_duration':duration,'last_frame_exact':mode=='last'}
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired,IndexError,json.JSONDecodeError,OSError):
        Path(destination).unlink(missing_ok=True)
        raise ValueError('Video non leggibile, istante non disponibile o estrazione scaduta. Usa MP4, MKV, WebM o AVI locale.') from None
