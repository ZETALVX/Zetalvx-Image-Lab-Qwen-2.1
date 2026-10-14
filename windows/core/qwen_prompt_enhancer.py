"""Optional Qwen 2.1 prompt rewriting: configuration and data-only validation.

PE is not a diffusion text encoder. No model imports, downloads, executable code
from model files, or changes to inference configuration occur on this web host.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
import re
import threading
from pathlib import Path
from core.source_util import atomic_json
from core.qwen_external_runtime import runtime_paths,discover_runtimes

PE_KINDS = {'t2i': 'prompt_enhancer_t2i', 'i2i': 'prompt_enhancer_i2i'}
MAX_PROMPT = 30000
MAX_RESULT = 256 * 1024
DEFAULTS = {
    'schema': 1, 'mode': 'off', 'comfy_root': '', 'comfy_python': '',
    'runtime_confirmed': False, 't2i_model_id': '', 'i2i_model_id': '',
    'max_tokens': 512, 'seed': 314159, 'thinking': False,
    'sampling': False, 'temperature': 0.7, 'top_p': 0.95,
    'template_mode': 'task', 'system_prompt': '', 'timeout_seconds': 600,
}
_LOCK = threading.RLock()


def check_string(value, label, limit=4096, allow_empty=True):
    if not isinstance(value, str) or '\0' in value or len(value) > limit:
        raise ValueError('Invalid ' + label + '.')
    value = value.strip()
    if not value and not allow_empty:
        raise ValueError('Missing ' + label + '.')
    return value


def normalize_config(raw, previous=None):
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError('Invalid Prompt Enhancer configuration.')
    out = copy.deepcopy(DEFAULTS)
    if previous:
        out.update({k: v for k, v in previous.items() if k in DEFAULTS})
    out.update(raw)
    out['schema'] = 1
    if out['mode'] not in ('off', 't2i', 'i2i'):
        raise ValueError('Choose Off, PE-T2I or PE-I2I.')
    for field in ('runtime_confirmed', 'thinking', 'sampling'):
        if type(out[field]) is not bool:
            raise ValueError(field + ' must be boolean.')
    for field in ('comfy_root', 'comfy_python'):
        value = check_string(out[field], field)
        if value:
            p = Path(value).expanduser()
            if not p.is_absolute():
                raise ValueError('Use an absolute server path for ' + field + '.')
            # Do not resolve a venv python symlink: its location identifies the venv.
            value = str(p.absolute())
        out[field] = value
    for field in ('t2i_model_id', 'i2i_model_id'):
        value = check_string(out[field], field, 120)
        if value and not re.fullmatch(r'[A-Za-z0-9_.-]+', value):
            raise ValueError('Invalid PE catalog identifier.')
        out[field] = value
    for field, lo, hi in (('max_tokens', 64, 4096), ('seed', 0, 2**53-1),
                          ('timeout_seconds', 30, 1800)):
        value = out[field]
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError(f'{field} must be between {lo} and {hi}.')
    for field, lo, hi in (('temperature', 0.05, 2), ('top_p', 0.01, 1)):
        value = out[field]
        if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError('Invalid ' + field + '.')
        out[field] = float(value)
    if out['template_mode'] not in ('task', 'model', 'custom'):
        raise ValueError('Choose the model template or a custom system instruction.')
    out['system_prompt'] = check_string(out['system_prompt'], 'system instruction', 16000)
    if out['template_mode'] == 'custom' and not out['system_prompt']:
        raise ValueError('Enter the custom system instruction or use the model template.')
    # Never carry permission to execute a different interpreter across a path change.
    if previous and any(out[k] != previous.get(k, '') for k in ('comfy_root', 'comfy_python')):
        if raw.get('runtime_confirmed') is not True:
            out['runtime_confirmed'] = False
    return out



def inspect_pe_file(path, kind):
    from core.qwen_weight_format import inspect_weights
    if kind not in PE_KINDS.values():
        raise ValueError('Select PE-T2I or PE-I2I.')
    p = Path(path).expanduser()
    if not p.is_absolute() or not p.is_file() or p.suffix.lower() != '.safetensors':
        raise ValueError('Select the Prompt Enhancer .safetensors file, not its containing folder.')
    info = inspect_weights(p)
    keys = info['keys']
    if not any('linear_attn.' in key for key in keys) or not any('embed_tokens.weight' in key for key in keys):
        raise ValueError('This file is not a supported Qwen3.5 Prompt Enhancer. Qwen3-VL diffusion encoders belong in Text encoder.')
    if kind == PE_KINDS['i2i'] and not any('visual.' in key for key in keys):
        raise ValueError('PE-I2I requires a Qwen3.5 model with its vision weights.')
    opposite = 'pe_t2i' if kind == PE_KINDS['i2i'] else 'pe_i2i'
    if opposite in p.name.lower():
        raise ValueError('The filename identifies the other PE task. Choose the matching PE-T2I / PE-I2I component.')
    return {'kind': kind, 'family': 'qwen21', 'format': 'safetensors',
            'quantized': info['quantized'], 'bytes': info['bytes'],
            'header_sha256': info['header_sha256'],
            'note': 'Qwen3.5 file structure checked. Confirm the PE task and model licence from its source. Native loading uses the selected external ComfyUI runtime.'}


def build_request(cfg, catalog, prompt, refs):
    cfg = normalize_config(cfg)
    if cfg['mode'] == 'off':
        raise ValueError('Prompt Enhancer is Off. Choose PE-T2I or PE-I2I to rewrite a prompt.')
    prompt = check_string(prompt, 'prompt', MAX_PROMPT, allow_empty=False)
    if not isinstance(refs, list) or len(refs) > 10 or any(not isinstance(x, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', x) for x in refs):
        raise ValueError('Use up to 10 ordered reference images.')
    if len(set(refs)) != len(refs):
        raise ValueError('Duplicate reference image identifiers are not allowed.')
    if cfg['mode'] == 'i2i' and not refs:
        raise ValueError('PE-I2I needs at least one input image. Add an image to Create / Edit first.')
    if cfg['mode'] == 't2i' and refs:
        raise ValueError('PE-T2I rewrites text without images. Choose PE-I2I or remove the reference images.')
    rid = cfg[cfg['mode'] + '_model_id']
    entry = next((x for x in catalog if x.get('id') == rid and x.get('kind') == PE_KINDS[cfg['mode']]), None)
    if not entry or not entry.get('present') or entry.get('selectable') is False:
        raise ValueError('Select an available Prompt Enhancer from its dedicated catalog.')
    inspect_pe_file(entry['path'], PE_KINDS[cfg['mode']])
    root, exe = runtime_paths(cfg)
    return {'config': {**cfg, 'comfy_root': str(root), 'comfy_python': str(exe)},
            'model_path': str(Path(entry['path']).resolve()), 'model_name': entry['name'],
            'model_id': entry['id'], 'original_prompt': prompt, 'reference_artifact_ids': list(refs)}


def parse_output(raw):
    """Strip reasoning and parse PE JSON, never pass metadata/reasoning to DiT."""
    if not isinstance(raw, str) or len(raw.encode('utf-8')) > MAX_RESULT:
        raise ValueError('The PE output is missing or too large.')
    text = raw.strip()
    # Some tokenizers suppress the opening special token but leave the closing one.
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1].strip()
    elif '<think>' in text:
        raise ValueError('PE stopped during thinking. Disable thinking or increase the PE token limit, then retry.')
    text = re.sub(r'<\|(?:im_end|endoftext)\|>\s*$', '', text).strip()
    if not text:
        raise ValueError('PE returned no usable prompt. Increase the PE token limit and retry.')
    decoder = json.JSONDecoder()
    # Output sometimes has a fenced JSON block or a short preamble.
    obj = None
    for match in re.finditer(r'\{', text):
        try:
            candidate, _ = decoder.raw_decode(text[match.start():])
        except (ValueError, RecursionError):
            continue
        if isinstance(candidate, dict) and isinstance(candidate.get('rewritten_prompt'), str):
            obj = candidate
            break
    if obj is not None:
        rewritten = check_string(obj['rewritten_prompt'], 'rewritten prompt', MAX_PROMPT, allow_empty=False)
        if any(x in rewritten for x in ('<think>', '</think>', '<|im_start|>')):
            raise ValueError('PE returned control/reasoning tokens instead of a usable prompt.')
        ratio = str(obj.get('wh_ratio') or '')[:64]
        follow = str(obj.get('ratio_follow') or '')[:64]
        return {'rewritten_prompt': rewritten, 'wh_ratio': ratio, 'ratio_follow': follow, 'output_format': 'pe_json'}
    if text.startswith(('{', '[', '```')) or '"rewritten_prompt"' in text:
        raise ValueError('PE returned incomplete or unsupported JSON. Increase the PE token limit and retry; the original prompt is unchanged.')
    if any(token in text for token in ('<think>', '</think>', '<|im_start|>')):
        raise ValueError('PE returned control/reasoning tokens instead of a usable prompt.')
    rewritten = check_string(text, 'rewritten prompt', MAX_PROMPT, allow_empty=False)
    return {'rewritten_prompt': rewritten, 'wh_ratio': '', 'ratio_follow': '', 'output_format': 'text'}


class PEStore:
    def __init__(self, shared):
        self.shared = Path(shared)
        self.config_path = self.shared / 'config/qwen-prompt-enhancer.json'
        self.results = self.shared / 'prompt-enhancer/results'
    def config(self):
        if not self.config_path.exists():
            return copy.deepcopy(DEFAULTS)
        if self.config_path.is_symlink() or self.config_path.stat().st_size > 100000:
            raise ValueError('Invalid PE configuration file.')
        return normalize_config(json.loads(self.config_path.read_text(encoding='utf-8')))
    def save(self, raw):
        with _LOCK:
            cfg = normalize_config(raw, self.config())
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(self.config_path, cfg)
        return cfg
    def result_path(self, jid):
        if not isinstance(jid, str) or not re.fullmatch(r'[a-f0-9]{32}', jid):
            raise ValueError('Invalid PE job identifier.')
        p = self.results / (jid + '.json')
        if p.is_symlink() or self.results.is_symlink():
            raise ValueError('Linked PE result paths are not accepted.')
        return p
    def put_result(self, jid, result):
        p = self.result_path(jid)
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(p, result)
    def result(self, jid):
        p = self.result_path(jid)
        if not p.is_file():
            return None
        if p.stat().st_size > MAX_RESULT:
            raise ValueError('Oversized PE result.')
        return json.loads(p.read_text(encoding='utf-8'))


def provenance(store, raw, project_id, prompt, references):
    """Attach only a completed PE result from this project, never client-supplied text history."""
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != {'job_id'}:
        raise ValueError('Invalid Prompt Enhancer provenance.')
    item = store.result(raw['job_id'])
    if not item or item.get('project_id') != project_id:
        raise ValueError('Prompt Enhancer result is unavailable for this project.')
    if item.get('reference_artifact_ids', []) != references:
        raise ValueError('Input images changed after prompt enhancement. Clear the applied PE result or rewrite again.')
    return {'job_id':raw['job_id'], 'mode':item['mode'], 'model_name':item['model_name'],
            'original_prompt':item['original_prompt'], 'rewritten_prompt':item['rewritten_prompt'],
            'effective_prompt':prompt, 'user_edited':prompt.strip()!=item['rewritten_prompt'].strip(),
            'reference_artifact_ids':list(references), 'engine':'external_comfyui',
            'options':item.get('options',{}), 'timing':item.get('timing',{}),
            'runtime':item.get('runtime',{}), 'model_identity':item.get('model_identity',{})}
