"""Chunked browser uploads for Qwen Image 2.1 transformer/LoRA/VAE files.

Files selected on the user's browser are copied into the app's standard model folders.
This is deliberately separate from local/server-path linking, which never copies weights.
"""
from __future__ import annotations
import json
import os
import shutil
import time
import uuid
import re
from pathlib import Path
from core.qwen_assets import QwenAssetError as LocalModelError,DESTINATIONS
from core.runtime_env import atomic_json

MAX_UPLOAD_BYTES = 32 * 1024**3
MAX_CHUNK_BYTES = 24 * 1024**2
STALE_SECONDS = 24 * 3600


class BrowserModelUploads:
    def __init__(self, models_root, local_models):
        self.models_root = Path(models_root).expanduser().resolve()
        self.local_models = local_models
        self.root = self.models_root / '.incoming' / 'browser-uploads'
        self.root.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, uid):
        if not isinstance(uid, str) or len(uid) != 32 or any(c not in '0123456789abcdef' for c in uid):
            raise LocalModelError('Invalid upload identifier.')
        path=self.root/f'{uid}.json'
        if path.is_symlink():raise LocalModelError('Linked upload state is not accepted.')
        return path

    def _load(self, uid):
        p = self._meta_path(uid)
        try:
            d = json.loads(p.read_text(encoding='utf-8'))
        except (OSError, ValueError) as e:
            raise LocalModelError('Upload not found or expired.') from e
        if not isinstance(d, dict) or d.get('id') != uid or d.get('kind') not in ('transformer','lora','vae','text_encoder_file','prompt_enhancer_t2i','prompt_enhancer_i2i'):
            raise LocalModelError('Invalid upload state.')
        self._stage(d)
        if type(d.get('size')) is not int or not 0<d['size']<=MAX_UPLOAD_BYTES:raise LocalModelError('Invalid upload size in state.')
        return d

    def _stage(self, d):
        # Stage keeps its data-only suffix and must remain in this private directory.
        uid=d.get('id');name=d.get('filename')
        if not isinstance(uid,str) or not re.fullmatch('[a-f0-9]{32}',uid):raise LocalModelError('Invalid upload identity.')
        if not isinstance(name,str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]{0,219}',name) or not name.lower().endswith(('.safetensors','.gguf')):raise LocalModelError('Invalid upload filename.')
        path=self.root/(uid+'-'+name)
        if path.is_symlink():raise LocalModelError('Linked upload data is not accepted.')
        return path

    def cleanup(self):
        now = time.time()
        for p in self.root.glob('*.json'):
            try:
                d = self._load(p.stem)
                if now - float(d.get('created_at', 0)) <= STALE_SECONDS:
                    continue
                stage = self._stage(d)
                stage.unlink(missing_ok=True)
                p.unlink(missing_ok=True)
            except Exception:
                continue

    def create(self, filename, size, kind):
        self.cleanup()
        if kind not in ('transformer','lora','vae','text_encoder_file','prompt_enhancer_t2i','prompt_enhancer_i2i'):
            raise LocalModelError('Select a Qwen transformer, LoRA or VAE.')
        if not isinstance(filename, str):
            raise LocalModelError('Choose a .safetensors or transformer .gguf file.')
        raw_name = Path(filename).name
        name = re.sub(r'[^A-Za-z0-9._-]+', '_', raw_name).strip('._')
        if not name or len(name)>220 or not name.lower().endswith(('.safetensors','.gguf')):
            raise LocalModelError('Only safetensors or transformer GGUF files are accepted.')
        if kind.startswith('prompt_enhancer_') and not name.lower().endswith('.safetensors'):
            raise LocalModelError('Prompt Enhancer uploads require a .safetensors file.')
        try:
            size = int(size)
        except (TypeError, ValueError):
            raise LocalModelError('Invalid file size.') from None
        if size <= 0 or size > MAX_UPLOAD_BYTES:
            raise LocalModelError('File size must be between 1 byte and 32 GiB.')
        free = shutil.disk_usage(self.models_root).free
        if free < size + 512 * 1024**2:
            raise LocalModelError('Not enough free disk space for this upload.')
        uid = uuid.uuid4().hex
        d = {'id': uid, 'filename': name, 'size': size, 'kind': kind, 'created_at': time.time(), 'received': 0}
        stage = self._stage(d)
        stage.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidentally appending to stale content.
        with open(stage, 'xb'):
            pass
        atomic_json(self._meta_path(uid), d)
        return self.public(d)

    def public(self, d):
        return {k: d.get(k) for k in ('id', 'filename', 'size', 'kind', 'created_at', 'received')}

    def append(self, uid, offset, file_storage):
        d = self._load(uid)
        try:
            offset = int(offset)
        except (TypeError, ValueError):
            raise LocalModelError('Invalid upload offset.') from None
        stage = self._stage(d)
        try:
            current = stage.stat().st_size
        except OSError as e:
            raise LocalModelError('Upload staging file is missing.') from e
        if offset != current or offset != int(d.get('received', current)):
            raise LocalModelError(f'Upload offset mismatch. Expected {current}.')
        if file_storage is None:
            raise LocalModelError('Missing upload chunk.')
        declared = getattr(file_storage, 'content_length', None)
        if declared and declared > MAX_CHUNK_BYTES:
            raise LocalModelError('Upload chunk is too large.')
        remaining = int(d['size']) - current
        if remaining <= 0:
            return self.public(d)
        written = 0
        try:
            with open(stage, 'ab') as out:
                while True:
                    block=file_storage.stream.read(min(1024*1024,MAX_CHUNK_BYTES-written+1,remaining-written+1))
                    if not block:break
                    if written+len(block)>MAX_CHUNK_BYTES or written+len(block)>remaining:
                        raise LocalModelError('Upload chunk exceeds the allowed size.')
                    out.write(block);written+=len(block)
                out.flush();os.fsync(out.fileno())
            if written<=0:raise LocalModelError('Empty upload chunk.')
        except BaseException:
            # A rejected request must leave the offset retryable, not corrupt the upload.
            with open(stage,'r+b') as out:out.truncate(current)
            raise
        d['received'] = current + written
        d['updated_at'] = time.time()
        atomic_json(self._meta_path(uid), d)
        return self.public(d)

    def finalize(self, uid, accept_unknown=False):
        d = self._load(uid); stage = self._stage(d)
        try:
            size = stage.stat().st_size
        except OSError as e:
            raise LocalModelError('Upload staging file is missing.') from e
        if size != int(d['size']):
            raise LocalModelError(f'Upload incomplete: {size} of {d["size"]} bytes received.')
        detail = self.local_models.inspect(str(stage), d['kind'])
        if detail.get('needs_confirmation') and accept_unknown is not True:
            return {'needs_confirmation': True, 'preview': {**detail, 'name': d['filename']}, 'upload': self.public(d)}
        target_root = self.models_root / DESTINATIONS[d['kind']]
        target_root.mkdir(parents=True, exist_ok=True)
        target = target_root / d['filename']
        if target.exists():
            stem, suffix = target.stem, target.suffix
            for n in range(2, 10000):
                candidate = target_root / f'{stem}-{n}{suffix}'
                if not candidate.exists():
                    target = candidate; break
            else:
                raise LocalModelError('Too many files with the same name in the destination folder.')
        if not target_root.resolve().is_relative_to(self.models_root):raise LocalModelError('Refusing a linked external upload destination.')
        os.link(stage,target);stage.unlink()
        self.local_models.add(str(target),d['kind'],accept_unknown=True)
        self._meta_path(uid).unlink(missing_ok=True)
        return {'needs_confirmation': False, 'path': str(target), 'name': target.name, 'kind': d['kind'], 'copied': True,
                'note': 'Uploaded from this device into the app model folder.'}

    def cancel(self, uid):
        d = self._load(uid)
        self._stage(d).unlink(missing_ok=True)
        self._meta_path(uid).unlink(missing_ok=True)
        return {'cancelled': uid}
