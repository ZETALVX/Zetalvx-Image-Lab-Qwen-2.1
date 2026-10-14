"""Lightweight tests for LoRA as an existing Benchmark Lab matrix dimension."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from core.qwen_benchmarks import BenchmarkStore, _benchmark_lora_variants


class AssetCatalog:
    def __init__(self, items):
        self.items = items

    def inventory(self, cfg=None):
        return self.items


class BenchmarkLoraTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        assets = []
        self.paths = {}
        for kind, names in [('lora', ('turbo8', 'viggle')), ('transformer', ('float2',))]:
            for name in names:
                path = root / f'{name}.safetensors'
                path.write_bytes(b'fixture')
                self.paths[name] = path
                assets.append({'id':name, 'name':path.name, 'path':str(path),
                               'kind':kind, 'present':True, 'selectable':True})
        self.assets = AssetCatalog(assets)
        self.cfg = {'model_dir':str(root), 'model_path':'', 'text_encoder_path':'',
                    'vae_path':'', 'quantized_loading':'reject'}
        self.store = BenchmarkStore(root/'reports', None, self.assets, lambda:{},
                                    {'validate_model':lambda x:{'ready':True}}, None)
        self.store.project = lambda pid:None
        self.store.preflight = lambda entries:{'ready':True,'jobs_total':len(entries),'jobs_ready':len(entries)}
        self.resolve = patch('core.qwen_benchmarks.resolve_request',side_effect=lambda *args: ('', '', '') if False else (dict(self.cfg), [], {'preset_name':'Baseline'}))
        self.resolve.start(); self.addCleanup(self.resolve.stop)
        self.weightcheck = patch('core.qwen_weight_format.require_supported_weights',return_value=None)
        self.weightcheck.start();self.addCleanup(self.weightcheck.stop)

    def request(self, **kwargs):
        result = dict(project_id='project',mode='auto',profile_ids=['@configured'],
                      transformers=['@baseline'],encoders=['@baseline'],vaes=['@baseline'],
                      prompts=[{'name':'Photo','prompt':'Test the adapter against the same baseline.'}],
                      params={'width':1024,'height':1024,'steps':8,'seed':123},
                      auto_strategy='full',lora_ids=['turbo8','viggle'],
                      lora_strengths={'turbo8':1.,'viggle':.75},
                      lora_combinations='single')
        result.update(kwargs)
        return result

    def test_full_baseline_plus_single_loras(self):
        p=self.store.plan(self.request())
        self.assertEqual(p['count'],3)
        self.assertEqual([len(x['params']['loras']) for x in p['entries']],[0,1,1])
        self.assertTrue(all(x['params']['seed']==123 for x in p['entries']))
        self.assertEqual([x['params']['loras'][0]['strength'] for x in p['entries'][1:]], [1.,.75])
        self.assertIn('No LoRA',p['entries'][0]['label'])
        self.assertIn('turbo8',p['entries'][1]['label'])

    def test_cartesian_multiplier(self):
        p=self.store.plan(self.request(transformers=['@baseline','float2']))
        self.assertEqual(p['count'],6)
        self.assertEqual(sum(not x['params']['loras'] for x in p['entries']),2)

    def test_screening_only_anchor_varies_loras(self):
        p=self.store.plan(self.request(transformers=['@baseline','float2'],auto_strategy='screening'))
        self.assertEqual(p['count'],4)
        self.assertEqual(sum(bool(x['params']['loras']) for x in p['entries']),2)

    def test_all_subsets(self):
        p=self.store.plan(self.request(lora_combinations='all'))
        self.assertEqual(p['count'],4)
        self.assertEqual(sorted(len(x['params']['loras']) for x in p['entries']),[0,1,1,2])

    def test_native_only_skips_loras_and_keeps_baseline(self):
        self.cfg['model_path']=str(Path(self.tmp.name)/'qwen.gguf')
        self.cfg['text_encoder_path']=str(self.paths['float2'])
        self.cfg['vae_path']=str(self.paths['viggle'])
        self.cfg['quantized_loading']='native_comfyui'
        p=self.store.plan(self.request())
        self.assertEqual(p['count'],1)
        self.assertFalse(p['entries'][0]['params']['loras'])
        self.assertEqual(len(p['skipped_incompatible_combinations']),2)
        self.assertTrue(all('native ComfyUI' in x['reason'] for x in p['skipped_incompatible_combinations']))

    def test_missing_adapter_or_invalid_strength(self):
        with self.assertRaises(ValueError):
            self.store.plan(self.request(lora_ids=['unregistered'],lora_strengths={}))
        with self.assertRaises(ValueError):
            self.store.plan(self.request(lora_strengths={'turbo8':3.,'viggle':1.}))
        with self.assertRaises(ValueError):
            self.store.plan(self.request(lora_ids=['turbo8','turbo8'],lora_strengths={'turbo8':1.}))

    def test_no_selected_loras_preserves_legacy(self):
        p=self.store.plan(self.request(lora_ids=[],lora_strengths={}))
        self.assertEqual(p['count'],1)
        self.assertFalse(p['entries'][0]['params']['loras'])

if __name__=='__main__':unittest.main()
