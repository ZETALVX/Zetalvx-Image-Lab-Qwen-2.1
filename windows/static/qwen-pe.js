/* Optional, explicit prompt preview; no automatic model execution. */
(()=>{'use strict';
 const $=s=>document.querySelector(s),t=(s,v)=>window.ZI18n?.t(s,v)||s;
 const panel=$('#qwenPEPanel');if(!panel)return;
 const state={loaded:false,loading:false,config:null,catalog:[],runtime:null,busy:false,job:null,result:null,applied:null,context:'',polling:false};
 const path='/api/model-hub/pe';
 const project=()=>typeof S!=='undefined'?S.project?.id:null;
 const refs=()=>typeof S!=='undefined'?[...(S.references||[])]:[];
 const context=()=>JSON.stringify([project(),refs()]);
 const runtimeReady=()=>!!(state.runtime?.verified&&state.runtime?.capabilities?.prompt_enhancer_imports);
 function status(msg,error=false){$('#peStatus').textContent=t(msg);$('#peStatus').classList.toggle('error',error)}
 function failure(e){status('Prompt Enhancer failed. Open technical details.',true);$('#peErrorDetails').hidden=false;$('#peError').textContent=String(e?.message||e)}
 async function call(p,method='GET',data){return api(path+p,method==='GET'?{}:{method,headers:{'Content-Type':'application/json'},body:JSON.stringify(data||{})})}
 function form(){const c={...(state.config||{})};c.mode=$('#peMode').value;c.comfy_root='';c.comfy_python='';c.runtime_confirmed=false;
 c.max_tokens=Number($('#peTokens').value);c.seed=Number($('#peSeed').value);c.thinking=$('#peThinking').checked;c.sampling=$('#peSampling').checked;c.temperature=Number($('#peTemperature').value);c.top_p=Number($('#peTopP').value);c.timeout_seconds=Number($('#peTimeout').value);c.template_mode=$('#peTemplate').value;c.system_prompt=$('#peSystem').value;
 if(c.mode!=='off')c[c.mode+'_model_id']=$('#peModel').value;return c}
 function choices(){const mode=$('#peMode').value;const id=state.config?.[mode+'_model_id']||'';const sel=$('#peModel');
 const items=state.catalog.filter(e=>e.kind==='prompt_enhancer_'+mode&&e.present&&e.selectable!==false);
 sel.replaceChildren(new Option(t('Select a PE model…'),''),...items.map(e=>new Option(e.name,e.id)));sel.value=items.some(e=>e.id===id)?id:'';sel.disabled=mode==='off';controls()}
 function controls(){const mode=$('#peMode').value;$('#peModeBadge').textContent=mode==='off'?t('Off'):'PE-'+mode.toUpperCase();$('#peRun').disabled=mode==='off'||state.busy||!$('#peModel').value||!runtimeReady();$('#peCancel').disabled=!state.job||!state.busy;
 $('#peSystemWrap').hidden=$('#peTemplate').value!=='custom';$('#peModelPath').textContent=(state.catalog.find(x=>x.id===$('#peModel').value)||{}).path||'';
 $('#peTemperature').disabled=!$('#peSampling').checked;$('#peTopP').disabled=!$('#peSampling').checked}
 function fill(c){state.config=c;$('#peMode').value=c.mode;
 for(const [id,key] of [['peTokens','max_tokens'],['peSeed','seed'],['peTemperature','temperature'],['peTopP','top_p'],['peTimeout','timeout_seconds'],['peTemplate','template_mode'],['peSystem','system_prompt']])$('#'+id).value=c[key];
 $('#peThinking').checked=c.thinking;$('#peSampling').checked=c.sampling;choices()}
 async function load(){if(state.loading)return;state.loading=true;try{const d=await call('/config');state.catalog=d.catalog||[];state.runtime=d.runtime||null;fill(d.config);const rs=$('#peRuntimeState');if(rs)rs.textContent=runtimeReady()?t('Verified ComfyUI runtime'):(state.runtime?.verified?t('Prompt Enhancer runtime check unavailable'):t('ComfyUI runtime not verified'));state.loaded=true;await history()}catch(e){failure(e)}finally{state.loading=false}}
 async function history(){if(!project())return;const pid=project();const d=await call('/jobs?project_id='+encodeURIComponent(pid));if(project()!==pid)return;const sel=$('#peHistory'),old=sel.value;sel.replaceChildren(new Option(t('Select a request…'),''),...(d.jobs||[]).map(j=>new Option((j.created_at||'')+' · '+t(j.status)+' · '+j.id.slice(0,8),j.id)));if([...sel.options].some(x=>x.value===old))sel.value=old;
 const pending=(d.jobs||[]).find(x=>['queued','running'].includes(x.status));if(pending&&!state.job){state.job={id:pending.id,pid};state.busy=true;controls()}}
 function display(result){state.result=result;$('#peResult').hidden=false;$('#peOriginal').textContent=result.original_prompt;$('#peOutput').value=result.rewritten_prompt;
 const sec=result.timing?.end_to_end_seconds;$('#peResultInfo').textContent=[result.model_name,result.execution_mode||'external ComfyUI',sec!==undefined?Number(sec).toFixed(1)+' s':'',result.output_tokens!==undefined?result.output_tokens+' '+t('tokens'):'',result.wh_ratio?'wh_ratio: '+result.wh_ratio:'',result.ratio_follow?'ratio_follow: '+result.ratio_follow:''].filter(Boolean).join(' · ');
 $('#peTokenWarning').hidden=!result.token_limit_reached;$('#peApplied').hidden=state.applied?.job_id!==result.job_id;status('Preview ready. Review it before applying.');}
 async function poll(){if(state.polling||!state.job)return;state.polling=true;const current={...state.job};try{const d=await call('/jobs/'+current.id+'?project_id='+encodeURIComponent(current.pid));if(!state.job||state.job.id!==current.id||project()!==current.pid)return;
 if(['completed','failed','cancelled'].includes(d.job.status)){state.busy=false;state.job=null;controls();if(d.job.status==='completed'&&d.result)display(d.result);else if(d.job.status==='failed')failure(d.job.error||'PE failed');else status('PE request cancelled.');await history()}else status(d.job.status==='queued'?'PE queued. It shares the image job queue.':'PE is working. The original prompt is unchanged.')
 }catch(e){failure(e)}finally{state.polling=false}}
 function detach(){state.applied=null;$('#peApplied').hidden=true}
 function guardContext(){const key=context();if(key!==state.context){state.context=key;detach();state.job=null;state.busy=false;state.result=null;$('#peResult').hidden=true;controls();if(state.loaded&&panel.open)history().catch(failure)}}
 panel.addEventListener('toggle',()=>{if(panel.open&&!state.loaded)load();else if(panel.open)history().catch(failure)});
 $('#peMode').onchange=()=>{if(state.config){const old=state.config.mode;if(old!=='off')state.config[old+'_model_id']=$('#peModel').value;state.config.mode=$('#peMode').value}choices();if($('#peMode').value!=='off'&&!runtimeReady())$('#peSettings').open=true};
 $('#peModel').onchange=()=>{if(state.config&&$('#peMode').value!=='off')state.config[$('#peMode').value+'_model_id']=$('#peModel').value;controls()};
 $('#peTemplate').onchange=controls;$('#peSampling').onchange=controls;
 $('#peAddModel').onclick=()=>{const mode=$('#peMode').value;showView('models');qwenOpenAddForKind('prompt_enhancer_'+(mode==='t2i'?'t2i':'i2i'))};
 $('#peSave').onclick=async()=>{try{state.config=(await call('/config','POST',form())).config;controls();status('PE settings saved.');$('#peErrorDetails').hidden=true}catch(e){failure(e)}};
 $('#peDefaults').onclick=()=>{for(const [id,val] of [['peTokens',512],['peSeed',314159],['peTemperature',.7],['peTopP',.95],['peTimeout',600],['peTemplate','task'],['peSystem','']])$('#'+id).value=val;$('#peThinking').checked=false;$('#peSampling').checked=false;controls();status('PE defaults restored. Runtime and model selections are kept.');};
 $('#peRun').onclick=async()=>{if(state.busy)return;const pid=project();try{if(!pid)throw Error(t('Open a project first.'));if(!$('#prompt').value.trim())throw Error(t('Enter a prompt or edit instruction.'));state.busy=true;controls();$('#peErrorDetails').hidden=true;const config=form();const d=await call('/jobs','POST',{project_id:pid,prompt:$('#prompt').value,reference_artifact_ids:refs(),config,confirm:true});
 if(pid===project()){state.config=config;state.job={id:d.id,pid};state.busy=true;status('PE queued. It shares the image job queue.');startPolling();await history();controls();await poll()}
 }catch(e){state.busy=false;controls();failure(e);if(pid===project())try{await history();controls()}catch(_){}}};
 $('#peCancel').onclick=async()=>{if(state.job){try{await call('/jobs/'+state.job.id+'/cancel','POST',{project_id:state.job.pid});status('Cancellation requested. Waiting for the PE process to stop.')}catch(e){failure(e)}}};
 $('#peApply').onclick=()=>{const r=state.result;if(!r)return;try{if(r.project_id!==project()||JSON.stringify(r.reference_artifact_ids)!==JSON.stringify(refs()))throw Error(t('Project or images changed. Rewrite again for the current inputs.'));
 const output=$('#peOutput').value.trim();if(!output)throw Error(t('Enter a prompt or edit instruction.'));
 const current=$('#prompt').value.trim();if(current!==r.original_prompt.trim()&&current!==output&&!confirm(t('The main prompt changed. Replace it with this reviewed preview?')))return;
 $('#prompt').value=output;state.applied={job_id:r.job_id,pid:project(),refs:JSON.stringify(refs())};$('#peApplied').hidden=false;saveUiSession();qwenControlState();status('Rewritten prompt applied. Generate when you are ready.');
 }catch(e){failure(e)}};
 $('#peRestore').onclick=()=>{if(!state.result)return;if(state.result.project_id!==project())return;if(!confirm(t('Replace the current prompt with the original PE input?')))return;$('#prompt').value=state.result.original_prompt;detach();saveUiSession();status('Original prompt restored.');};
 $('#peDetach').onclick=()=>{detach();status('Prompt kept. PE provenance cleared.');};
 $('#peHistoryLoad').onclick=async()=>{try{const jid=$('#peHistory').value;if(!jid||!project())return;const pid=project();const d=await call('/jobs/'+jid+'?project_id='+encodeURIComponent(pid));if(project()!==pid)return;if(d.result)display(d.result);else if(['queued','running'].includes(d.job.status)){state.job={id:jid,pid};state.busy=true;controls();await poll()}else failure(d.job.error||t(d.job.status))}catch(e){failure(e)}};
 $('#peRefresh').onclick=()=>history().catch(failure);
 window.addEventListener('qwen-hub-updated',e=>{if(!state.loaded)return;state.catalog=(e.detail?.local_catalog||[]).filter(x=>x.kind.startsWith('prompt_enhancer_'));const c=form();state.config=c;choices()});
 window.addEventListener('languagechange',()=>{if(state.loaded)choices()});window.addEventListener('qwen-comfy-runtime-updated',e=>{state.runtime=e.detail||null;const rs=$('#peRuntimeState');if(rs)rs.textContent=runtimeReady()?t('Verified ComfyUI runtime'):(state.runtime?.verified?t('Prompt Enhancer runtime check unavailable'):t('ComfyUI runtime not verified'));controls()});
 window.QwenPE={async selectEntry(e){if(!state.loaded)await load();if(!state.config)return;state.config.mode=e.kind==='prompt_enhancer_t2i'?'t2i':'i2i';state.config[state.config.mode+'_model_id']=e.id;if(!state.catalog.some(x=>x.id===e.id))state.catalog.push(e);$('#peMode').value=state.config.mode;choices();showView('image');panel.open=true;panel.scrollIntoView({behavior:'smooth',block:'start'})},
 provenance(){guardContext();const a=state.applied;return a&&a.pid===project()&&a.refs===JSON.stringify(refs())?{job_id:a.job_id}:null}};
 state.context=context();setInterval(()=>{guardContext();if(state.job)poll()},1600);controls();
})();
