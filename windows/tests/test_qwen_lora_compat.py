from __future__ import annotations
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

class DummyLoraConfig:
    def __init__(self, r=8, lora_alpha=8, target_modules=None, peft_version=None):
        pass

sys.modules['peft'] = types.SimpleNamespace(LoraConfig=DummyLoraConfig)

# Work with module paths provided by test runner's PYTHONPATH.
from services.qwen_lora_compat import (
    _diff_target, _safe_peft_metadata, _read_compatible_header,
    load_compatible_lora, apply_additive_plans,
)

class FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.norm_k = torch.nn.LayerNorm(4)
        block.attn.norm_q = torch.nn.LayerNorm(4)
        self.transformer_blocks = torch.nn.ModuleList([block])
        self.txt_in = torch.nn.Module()
        self.txt_in.text_norm = torch.nn.LayerNorm(4)

class FakePipe:
    def __init__(self):
        self.transformer = FakeTransformer()
        self.calls = []
        self.meta = {}
    def load_lora_weights(self, sd, **kwargs):
        self.calls.append(('normal', sd, kwargs))
    def lora_state_dict(self, source, **kwargs):
        self.calls.append(('state', source, kwargs))
        if isinstance(source, str):
            return ({'transformer.test.lora_A.weight': torch.zeros(2, 3)}, self.meta)
        return ({'transformer.test.lora_A.weight': torch.zeros(2, 3)}, None)
    def load_lora_into_transformer(self, sd, **kwargs):
        self.calls.append(('inject', sd, kwargs))

class CompatibilityTests(unittest.TestCase):
    def test_target_whitelist(self):
        self.assertEqual(_diff_target('diffusion_model.transformer_blocks.11.attn.norm_k.diff'),
                         'transformer_blocks.11.attn.norm_k.weight')
        self.assertEqual(_diff_target('diffusion_model.txt_in.text_norm.diff'), 'txt_in.text_norm.weight')
        with self.assertRaises(ValueError):
            _diff_target('diffusion_model.transformer_blocks.0.attn.to_q.diff')

    def test_peft_null_filter(self):
        meta = {'transformer.r':128, 'transformer.use_bdlora':None, 'transformer.peft_version':'0.21.0'}
        clean, removed = _safe_peft_metadata(meta)
        self.assertEqual(clean, {'transformer.r':128,'transformer.peft_version':'0.21.0'})
        self.assertEqual(removed, ('transformer.use_bdlora',))
        with self.assertRaises(ValueError):
            _safe_peft_metadata({'transformer.velora_config':{'enabled':True}})

    def test_official_diff_uses_standard_converter_and_applies_only_after_activation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'official.safetensors'
            save_file({
                'diffusion_model.transformer_blocks.0.attn.norm_k.diff': torch.tensor([.1,.2,.3,.4]),
                'diffusion_model.txt_in.text_norm.diff': torch.tensor([.01,.02,.03,.04]),
                'diffusion_model.transformer_blocks.0.attn.to_q.lora_down.weight': torch.zeros(2,4),
                'diffusion_model.transformer_blocks.0.attn.to_q.lora_up.weight': torch.zeros(4,2),
            },str(p),metadata={'format':'pt'})
            pipe=FakePipe()
            initial = pipe.transformer.transformer_blocks[0].attn.norm_k.weight.detach().clone()
            planned = load_compatible_lora(pipe,p,'test0',eager=True)
            self.assertEqual(len(planned),2)
            self.assertTrue(torch.equal(initial,pipe.transformer.transformer_blocks[0].attn.norm_k.weight))
            self.assertEqual(pipe.calls[0][0],'normal')
            self.assertEqual(len(pipe.calls[0][1]),2)
            self.assertFalse(any(k.endswith('.diff') for k in pipe.calls[0][1]))
            self.assertEqual(apply_additive_plans([(planned,0.5)]),2)
            torch.testing.assert_close(pipe.transformer.transformer_blocks[0].attn.norm_k.weight,
                initial+torch.tensor([.05,.1,.15,.2]),rtol=0,atol=1e-6)
            torch.testing.assert_close(pipe.transformer.txt_in.text_norm.weight,
                torch.ones(4)+torch.tensor([.005,.01,.015,.02]),rtol=0,atol=1e-6)

    def test_unknown_additive_target_aborts_before_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.safetensors'
            save_file({'diffusion_model.invalid.diff':torch.ones(4),'a.lora_A.weight':torch.ones(1,4),
                       'a.lora_B.weight':torch.ones(4,1)},str(p))
            pipe=FakePipe()
            with self.assertRaises(ValueError):
                load_compatible_lora(pipe,p,'n',eager=True)
            self.assertEqual(pipe.calls,[])

    def test_unknown_delta_shape_aborts_before_injection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'badshape.safetensors'
            save_file({'diffusion_model.transformer_blocks.0.attn.norm_k.diff':torch.ones(5),
                       'a.lora_A.weight':torch.ones(1,4),'a.lora_B.weight':torch.ones(4,1)},str(p))
            pipe=FakePipe()
            with self.assertRaisesRegex(ValueError,'Shape mismatch'):
                load_compatible_lora(pipe,p,'n',eager=True)
            self.assertEqual(pipe.calls,[])

    def test_turbo8_filters_only_null_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'turbo8.safetensors'
            meta={'transformer.r':128,'transformer.lora_alpha':128,
                  'transformer.target_modules':['to_q'], 'transformer.peft_version':'0.21.0',
                  'transformer.kasa_config':None,'transformer.lora_ga_config':None,
                  'transformer.monteclora_config':None,'transformer.use_bdlora':None,
                  'transformer.velora_config':None}
            save_file({'transformer.test.lora_A.weight':torch.zeros(2,3),
                       'transformer.test.lora_B.weight':torch.zeros(3,2)},str(p),
                      metadata={'format':'pt','lora_adapter_metadata':json.dumps(meta)})
            pipe=FakePipe();pipe.meta=meta
            self.assertEqual(load_compatible_lora(pipe,p,'turbo8',eager=True),[])
            self.assertEqual([item[0] for item in pipe.calls], ['state','inject'])
            clean=pipe.calls[-1][2]['metadata']
            self.assertEqual(set(clean),set(meta)-{k for k in meta if meta[k] is None})
            self.assertEqual(clean['transformer.peft_version'],'0.21.0')
            self.assertEqual(pipe.calls[-1][2]['low_cpu_mem_usage'], False)

    def test_failure_rollback_restores_previous_params(self):
        fake = FakeTransformer()
        p = fake.transformer_blocks[0].attn.norm_k.weight
        original = p.clone().detach()
        # Shape error at add_ triggers transaction rollback even if the first
        # additive operation succeeded.
        with self.assertRaises(RuntimeError):
            apply_additive_plans([([('norm', p, torch.ones(4)),
                                    ('norm2', p, torch.ones(5))], 1.0)])
        self.assertTrue(torch.equal(p, original))

    def test_default_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'ordinary.safetensors'
            save_file({'transformer.test.lora_A.weight':torch.zeros(2,3),
                       'transformer.test.lora_B.weight':torch.zeros(3,2)},str(p))
            pipe=FakePipe()
            self.assertEqual(load_compatible_lora(pipe,p,'base',eager=True),[])
            self.assertEqual(pipe.calls[0][0], 'normal')
            self.assertEqual(pipe.calls[0][1],str(p))

if __name__=='__main__': unittest.main()
