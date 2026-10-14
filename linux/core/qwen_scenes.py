"""Persistent user-managed automatic benchmark scenes (no inference changes).

Built-ins remain shipped in the UI. Only per-user custom scenes and overrides
are stored; benchmark plans keep their existing immutable prompt snapshots.
"""
from __future__ import annotations

import copy
import json
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from core import file_lock
from core.source_util import atomic_json

# IDs of the existing built-in scenes, not their contents. Their original
# prompts and names live in the unchanged default scene list in qwen-studio.js.
BUILTIN_IDS = frozenset({'photo', 'instruction', 'poster', 'complex', 'diagram'})
SCHEMA = 1
MAX_CUSTOM = 200


class BenchmarkScenes:
    def __init__(self, root):
        self.root = Path(root).absolute()

    def _guard(self):
        if any(part.is_symlink() for part in (self.root, *self.root.parents)):
            raise ValueError('Linked benchmark scene directory is not allowed.')
        for name in ('scenes.json', 'scenes.lock'):
            if (self.root / name).is_symlink():
                raise ValueError('Linked benchmark scene files are not allowed.')

    @contextmanager
    def _locked(self):
        self._guard()
        self.root.mkdir(parents=True, exist_ok=True)
        self._guard()
        with (self.root / 'scenes.lock').open('a+') as lock:
            file_lock.flock(lock.fileno(), file_lock.LOCK_EX)
            try:
                yield
            finally:
                file_lock.flock(lock.fileno(), file_lock.LOCK_UN)

    def _read(self):
        self._guard()
        path = self.root / 'scenes.json'
        if not path.exists():
            return {'schema': SCHEMA, 'custom': [], 'overrides': []}
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or data.get('schema') != SCHEMA:
            raise ValueError('Invalid benchmark scenes data.')
        if not isinstance(data.get('custom'), list) or not isinstance(data.get('overrides'), list):
            raise ValueError('Invalid benchmark scenes catalogue.')
        # Do not silently discard malformed user data.
        ids = set()
        for category in ('custom', 'overrides'):
            for item in data[category]:
                if not isinstance(item, dict) or not isinstance(item.get('id'), str):
                    raise ValueError('Invalid saved benchmark scene.')
                rid = item['id']
                if rid in ids or not (rid in BUILTIN_IDS or re.fullmatch('[a-f0-9]{32}', rid)):
                    raise ValueError('Invalid or duplicate benchmark scene identifier.')
                if (category == 'custom') == (rid in BUILTIN_IDS):
                    raise ValueError('Invalid benchmark scene category.')
                self._validate_text(item)
                ids.add(rid)
        if len(data['custom']) > MAX_CUSTOM:
            raise ValueError('Benchmark scene catalogue exceeds the limit.')
        return data

    @staticmethod
    def _validate_text(item):
        name = item.get('name')
        prompt = item.get('prompt')
        if not isinstance(name, str) or not (1 <= len(name.strip()) <= 80):
            raise ValueError('Scene name must contain 1–80 characters.')
        if not isinstance(prompt, str) or not (1 <= len(prompt.strip()) <= 8000):
            raise ValueError('Scene prompt must contain 1–8000 characters.')
        return name.strip(), prompt.strip()

    def list(self):
        with self._locked():
            return copy.deepcopy(self._read())

    def save(self, value, *, rid=None):
        if not isinstance(value, dict) or set(value) - {'name', 'prompt'}:
            raise ValueError('Invalid benchmark scene payload.')
        name, prompt = self._validate_text(value)
        if rid is not None and not isinstance(rid, str):
            raise ValueError('Invalid benchmark scene identifier.')
        if rid is not None and rid not in BUILTIN_IDS and not re.fullmatch('[a-f0-9]{32}', rid):
            raise ValueError('Invalid benchmark scene identifier.')
        with self._locked():
            data = self._read()
            target = 'overrides' if rid in BUILTIN_IDS else 'custom'
            old = next((x for x in data[target] if x['id'] == rid), None)
            if rid and target == 'custom' and old is None:
                raise ValueError('Custom benchmark scene not found.')
            if rid is None:
                if len(data['custom']) >= MAX_CUSTOM:
                    raise ValueError('Maximum 200 custom benchmark scenes.')
                rid = uuid.uuid4().hex
            now = time.time()
            row = {'id': rid, 'name': name, 'prompt': prompt,
                   'created_at': old.get('created_at', now) if old else now,
                   'updated_at': now}
            if old:
                data[target][data[target].index(old)] = row
            else:
                data[target].append(row)
            atomic_json(self.root / 'scenes.json', data)
            return copy.deepcopy(row)

    def delete(self, rid):
        if not isinstance(rid, str) or not (rid in BUILTIN_IDS or re.fullmatch('[a-f0-9]{32}', rid)):
            raise ValueError('Invalid benchmark scene identifier.')
        with self._locked():
            data = self._read()
            target = 'overrides' if rid in BUILTIN_IDS else 'custom'
            existing = next((x for x in data[target] if x['id'] == rid), None)
            if existing is None:
                raise ValueError('Scene override not found.' if target == 'overrides' else 'Custom benchmark scene not found.')
            data[target].remove(existing)
            atomic_json(self.root / 'scenes.json', data)
            return {'deleted': rid, 'restored_builtin': target == 'overrides'}
