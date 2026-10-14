"""Non-GPU regression checks for the final benchmark scene-manager addition."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from core.qwen_scenes import BenchmarkScenes, BUILTIN_IDS


class BenchmarkScenesTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'persistent' / 'benchmarks' / 'qwen21-scenes'
        self.store = BenchmarkScenes(self.path)

    def test_original_scenes_available_without_writing_data(self):
        self.assertEqual(self.store.list()['custom'], [])
        self.assertEqual(self.store.list()['overrides'], [])
        self.assertFalse((self.path/'scenes.json').exists())
        self.assertEqual(BUILTIN_IDS, {'photo','instruction','poster','complex','diagram'})

    def test_create_edit_persist_and_delete_user_scene(self):
        saved = self.store.save({'name':'Test caption', 'prompt':'Create a custom scene.'})
        self.assertEqual(saved['name'], 'Test caption')
        self.assertRegex(saved['id'], '^[a-f0-9]{32}$')
        second = BenchmarkScenes(self.path)
        self.assertEqual(second.list()['custom'][0]['prompt'], 'Create a custom scene.')
        updated = second.save({'name':'Renamed', 'prompt':'Another custom scene'},rid=saved['id'])
        self.assertEqual(updated['created_at'],saved['created_at'])
        self.assertEqual(len(self.store.list()['custom']), 1)
        self.assertEqual(self.store.delete(saved['id']),{'deleted':saved['id'],'restored_builtin':False})
        self.assertEqual(second.list()['custom'], [])

    def test_modify_builtin_then_restore_original(self):
        self.store.save({'name':'Modified watchmaker','prompt':'A different test scene'},rid='photo')
        self.assertEqual(self.store.list()['overrides'][0]['id'], 'photo')
        self.assertTrue(self.store.delete('photo')['restored_builtin'])
        self.assertEqual(self.store.list()['overrides'],[])
        self.assertEqual(self.store.list()['custom'],[])

    def test_reject_invalid_input_and_identifiers(self):
        for bad in ({'name':'no prompt'}, {'name':'x','prompt':''}, {'name':'a'*81,'prompt':'test'},
                    {'name':'ok','prompt':'x'*8001}, {'name':'ok','prompt':'ok','unknown':'bad'}):
            with self.subTest(bad=list(bad)), self.assertRaises(ValueError):self.store.save(bad)
        for rid in ('/../escape','', 'deadbeef'*4):
            with self.subTest(rid=rid), self.assertRaises(ValueError):self.store.save({'name':'ok','prompt':'ok'},rid=rid)
        with self.assertRaises(ValueError):self.store.delete('diagram')

    def test_no_symlink_catalogue(self):
        self.path.mkdir(parents=True)
        outside = Path(self.temp.name)/'outside.json'
        outside.write_text('{"schema":1,"custom":[],"overrides":[]}')
        (self.path/'scenes.json').symlink_to(outside)
        with self.assertRaises(ValueError):self.store.list()
        with self.assertRaises(ValueError):self.store.save({'name':'x','prompt':'y'})
        self.assertEqual(outside.read_text(),'{"schema":1,"custom":[],"overrides":[]}')

    def test_invalid_existing_user_data_is_not_overwritten(self):
        self.path.mkdir(parents=True)
        doc=self.path/'scenes.json'
        doc.write_text('{"schema":1,"custom":[{"id":"../../bad"}],"overrides":[]}')
        with self.assertRaises(ValueError):self.store.save({'name':'x','prompt':'y'})
        self.assertIn('../../bad',doc.read_text())

if __name__ == '__main__':
    unittest.main()
