"""Image-only utilities. Qwen edition media utilities; FFmpeg frame extraction is preserved."""
import json, uuid
from pathlib import Path
from PIL import Image, ImageOps
from core.db import execute
from core.paths import ARTIFACTS_DIR

def run_media_tool(project_id,src,operation,params):
    if src.get('type')!='image':raise ValueError('This image operation accepts image assets only.')
    if operation not in ('resize_image','convert_image','crop_image','flip_image','rotate_image'):raise ValueError('Image operation not supported.')
    with Image.open(src['path']) as im:
        image=ImageOps.exif_transpose(im).copy()
    if operation=='resize_image':
        w=int(params.get('width') or image.width);h=int(params.get('height') or image.height)
        if not (16<=w<=16384 and 16<=h<=16384) or w*h>64000000:raise ValueError('Invalid image size (maximum 64 MP).')
        if params.get('keep_aspect'):image=ImageOps.contain(image,(w,h),Image.Resampling.LANCZOS)
        else:image=image.resize((w,h),Image.Resampling.LANCZOS)
    elif operation=='crop_image':
        x=max(0,int(params.get('x') or 0));y=max(0,int(params.get('y') or 0))
        w=int(params.get('width') or image.width);h=int(params.get('height') or image.height)
        if w<1 or h<1 or x+w>image.width or y+h>image.height:raise ValueError('Crop rectangle outside image.')
        image=image.crop((x,y,x+w,y+h))
    elif operation=='flip_image':image=ImageOps.mirror(image)
    elif operation=='rotate_image':image=image.rotate(float(params.get('degrees') or 90),expand=True)
    ext=str(params.get('format') or 'png').lower().lstrip('.')
    if ext not in ('png','jpg','jpeg','webp'):raise ValueError('Choose PNG, JPG or WebP.')
    aid=uuid.uuid4().hex;out=ARTIFACTS_DIR/project_id/f'{aid}.{ext}';out.parent.mkdir(parents=True,exist_ok=True)
    if ext in ('jpg','jpeg'):image=image.convert('RGB')
    image.save(out,quality=max(1,min(100,int(params.get('quality') or 95))))
    meta={'source':'media_tools','operation':operation,'params':params,'source_artifact_id':src['id'],'width':image.width,'height':image.height}
    execute('INSERT INTO artifacts(id,project_id,type,path,parent_id,metadata_json) VALUES(?,?,?,?,?,?)',(aid,project_id,'image',str(out),src['id'],json.dumps(meta)))
    return {'artifact_id':aid,'output_path':str(out),'operation':operation,'type':'image'}
