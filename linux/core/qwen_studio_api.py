"""Preset/benchmark API; inherits application auth, Origin and model-hub CSRF."""
import json
from flask import request,jsonify,Response
from core.runtime_env import SHARED_ROOT
from core.qwen_studio import PresetLibrary,resolve_request,export_portable,import_portable
from core.qwen_benchmarks import BenchmarkStore
from core.qwen_scenes import BenchmarkScenes


def register(app,cb):
    current=lambda:dict((cb['model_by_id']('qwen_image_21') or {}).get('config') or {})
    assets=app.extensions['model_hub']['assets'];profiles=app.extensions['qwen_lab']['profiles']
    lib=PresetLibrary(SHARED_ROOT/'presets/qwen-studio',profiles,cb,current)
    bench=BenchmarkStore(SHARED_ROOT/'benchmarks/qwen21',lib,assets,current,cb,app.extensions['qwen_lab']['latents'])
    scenes=BenchmarkScenes(SHARED_ROOT/'benchmarks/qwen21-scenes')
    app.extensions['qwen_studio']={'library':lib,'assets':assets,'benchmarks':bench,'scenes':scenes}
    base='/api/model-hub/studio'
    def attachment(data,name,mime='application/json'):
        text=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False) if mime=='application/json' else data
        return Response(text,mimetype=mime,headers={'Content-Disposition':'attachment; filename="'+name+'"','Cache-Control':'no-store'})
    def body():
        d=request.get_json()
        if not isinstance(d,dict):raise ValueError('Expected a JSON object.')
        return d
    @app.get(base+'/presets')
    def studio_list():return jsonify(ok=True,presets=lib.list(),catalog=assets.inventory(current()),config=current())
    @app.post(base+'/resolve')
    def studio_resolve():
        cfg,loras,label=resolve_request(body().get('model_setup') or {},lib,assets,current())
        return jsonify(ok=True,config=cfg,loras=loras,label=label)
    @app.post(base+'/presets')
    def studio_save():
        d=body();return jsonify(ok=True,preset=lib.save(d,rid=d.get('id')))
    @app.delete(base+'/presets/<kind>/<rid>')
    def studio_delete(kind,rid):return jsonify(ok=True,**lib.delete(kind,rid))
    @app.post(base+'/presets/<kind>/<rid>/duplicate')
    def studio_duplicate(kind,rid):
        item=lib.get(kind,rid);item['name']=(item['name']+' (copy)')[:120]
        return jsonify(ok=True,preset=lib.save(item))
    @app.get(base+'/presets/<kind>/<rid>/export')
    def studio_export(kind,rid):return attachment(export_portable(lib.get(kind,rid)),'qwen21-'+kind+'-preset.json')
    @app.post(base+'/import')
    def studio_import():
        d=body();item,missing=import_portable(d.get('payload'),assets,current(),d.get('bindings'))
        if missing:return jsonify(ok=True,needs_bindings=missing)
        if d.get('confirm') is not True:return jsonify(ok=True,preview=item,needs_bindings=[])
        return jsonify(ok=True,preset=lib.save(item),needs_bindings=[])
    @app.get(base+'/benchmark-scenes')
    def studio_benchmark_scenes():return jsonify(ok=True,**scenes.list())
    @app.post(base+'/benchmark-scenes')
    def studio_create_scene():return jsonify(ok=True,scene=scenes.save(body()))
    @app.put(base+'/benchmark-scenes/<rid>')
    def studio_update_scene(rid):return jsonify(ok=True,scene=scenes.save(body(),rid=rid))
    @app.delete(base+'/benchmark-scenes/<rid>')
    def studio_remove_scene(rid):return jsonify(ok=True,**scenes.delete(rid))
    @app.post(base+'/benchmarks/plan')
    def studio_benchmark_plan():return jsonify(ok=True,plan=bench.plan(body()))
    @app.get(base+'/benchmarks')
    def studio_benchmarks():return jsonify(ok=True,benchmarks=bench.list(request.args.get('project_id')))
    @app.get(base+'/benchmarks/library')
    def studio_benchmark_library():return jsonify(ok=True,benchmarks=bench.archive(request.args.get('project_id') or None))
    @app.post(base+'/benchmarks/compare')
    def studio_benchmark_compare():return jsonify(ok=True,comparison=bench.comparison(body().get('ids')))
    @app.patch(base+'/benchmarks/<rid>/visibility')
    def studio_benchmark_visibility(rid):
        d=body();return jsonify(ok=True,**bench.set_visibility(rid,d.get('job_id'),d.get('hidden')))
    @app.get(base+'/benchmarks/<rid>')
    def studio_benchmark(rid):return jsonify(ok=True,report=bench.report(rid))
    @app.patch(base+'/benchmarks/<rid>')
    def studio_benchmark_rename(rid):
        d=body();return jsonify(ok=True,benchmark=bench.rename(rid,d.get('name')))
    @app.get(base+'/benchmarks/<rid>/preflight')
    def studio_benchmark_preflight(rid):return jsonify(ok=True,preflight=bench.preflight_report(rid))
    @app.post(base+'/benchmarks/<rid>/launch')
    def studio_launch(rid):
        if body().get('confirm') is not True:raise ValueError('Review the plan and explicitly confirm before queueing.')
        return jsonify(ok=True,**bench.launch(rid)),202
    @app.post(base+'/benchmarks/<rid>/retry')
    def studio_retry(rid):
        d=body();return jsonify(ok=True,**bench.retry(rid,include_cancelled=bool(d.get('include_cancelled',False)))),202
    @app.post(base+'/benchmarks/<rid>/cancel')
    def studio_cancel(rid):return jsonify(ok=True,**bench.cancel(rid))
    @app.delete(base+'/benchmarks/<rid>')
    def studio_remove_bench(rid):return jsonify(ok=True,**bench.delete(rid))
    @app.post(base+'/benchmarks/<rid>/delete')
    def studio_remove_bench_with_contents(rid):
        d=body();return jsonify(ok=True,**bench.delete(rid,delete_contents=bool(d.get('delete_contents',False))))
    @app.get(base+'/benchmarks/<rid>/export')
    def studio_report_export(rid):return attachment(bench.public_report(rid),'qwen21-benchmark-'+rid+'.json')
    @app.get(base+'/benchmarks/<rid>/csv')
    def studio_report_csv(rid):return attachment(bench.csv(rid),'qwen21-benchmark-'+rid+'.csv','text/csv')
