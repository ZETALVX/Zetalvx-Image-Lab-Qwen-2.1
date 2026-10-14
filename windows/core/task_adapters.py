IMAGE_PROVIDERS={'qwen_image21'}
LEVEL_RANK={'native':3,'fallback':2,'experimental':1,'unavailable':0}
def task_from_tool(tool):return {'generate_image':'generate','edit_image':'edit','img2img':'edit'}.get(tool,tool)
def task_support(model,task):
    level='unavailable';reason='Unsupported Qwen Image 2.1 task.'
    if model.get('provider')=='qwen_image21':
        if task in ('generate','edit','reference','multi_image'):level='native';reason='Qwen Image 2.1 unified generation and editing.'
        elif task=='inpaint':level='fallback';reason='Qwen semantic editing followed by local mask compositing.'
    return {'level':level,'rank':LEVEL_RANK[level],'executable':level!='unavailable','reason':reason}
def support_matrix(model):return {t:task_support(model,t) for t in ('generate','edit','reference','multi_image','inpaint')}
def resolve_strategy(model,tool,params=None):
    task=task_from_tool(tool);support=task_support(model,task)
    if not support['executable']:raise ValueError(support['reason'])
    return {'task':task,'worker_task':task,'support':support}
