"""1.0.72 no-torch automated local LoRA-folder import tests."""
from __future__ import annotations
import json, tempfile, struct, sys
from pathlib import Path

from core.qwen_assets import QwenAssets, QwenAssetError, scan_model_folder

def write_safe(path, keys):
    header={}
    start=0
    for key in keys:
        header[key]={'dtype':'F32','shape':[1], 'data_offsets':[start,start+4]}
        start+=4
    encoded=json.dumps(header).encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(struct.pack('<Q',len(encoded))+encoded+b'\x00'*start)

with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    folder=root/'qwen_misc_adapters'
    folder.mkdir()
    # A normal LoRA with generic name (classification cannot depend on filename).
    write_safe(folder/'turbo8_step2500.safetensors',[
        'transformer.transformer_blocks.0.attn.to_q.lora_A.weight',
        'transformer.transformer_blocks.0.attn.to_q.lora_B.weight',
    ])
    # Nested LoRA and a sidecar config. Hidden cache should not be scanned.
    write_safe(folder/'loras'/'rapid.safetensors',[
        'diffusion_model.transformer_blocks.0.attn.to_q.lora_down.weight',
        'diffusion_model.transformer_blocks.0.attn.to_q.lora_up.weight',
    ])
    write_safe(folder/'.cache'/'skip.safetensors',[
        'diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight',
    ])
    # A standalone transformer is not a LoRA and must be excluded by inspection.
    write_safe(folder/'not_a_lora.safetensors',[
        'transformer.transformer_blocks.0.attn.to_q.weight'
    ])
    (folder/'notes.txt').write_text('not a model')
    (folder/'link.safetensors').symlink_to(folder/'turbo8_step2500.safetensors')
    catalog=QwenAssets(root/'managed',root/'state.json')
    result=catalog.add_lora_folder(str(folder),accept_unknown=True)
    assert result['scanned']==3, result
    assert result['linked']==2, result
    assert result['failed']==1, result
    assert 'not_a_lora' in result['errors'][0]['path'], result
    inv=catalog.inventory()
    assert len(inv)==2, inv
    assert all(e['kind']=='lora' and e['present'] for e in inv),inv
    assert not any(e['path']==str(folder) for e in inv)
    assert {Path(e['path']).name for e in inv}=={'turbo8_step2500.safetensors','rapid.safetensors'}
    assert not any(str(folder/'.cache') in e['path'] for e in inv)
    again=catalog.add_lora_folder(str(folder),accept_unknown=True)
    assert again['linked']==2 and len(catalog.inventory())==2,'must be idempotent'
    no_confirmation=catalog.add_lora_folder(str(folder),accept_unknown=False)
    assert no_confirmation['linked']==0 and no_confirmation['failed']==3,no_confirmation
    try:catalog.add_lora_folder(str(root/'uncreated'),accept_unknown=True)
    except QwenAssetError:pass
    else:raise AssertionError('missing folder was accepted')
    print('PASS: flat/nested LoRA, bad file rejection, symlink/cache skipping, catalog idempotency, licences and missing folder')
