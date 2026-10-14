/* Modified for Zetalvx Image Lab Qwen 1.0.27: catalog-first profiles; Apache-2.0, retained NOTICE. */
/* Qwen Components 1.0.22: explicit profiles and preserved-latent decoding. */
let qwenProfiles=[],qwenLatents=[],qwenLabBusy=false,qwenTestTimer=null,qwenTestId='',qwenLabEpoch=0;
const QWEN22_DEFAULTS={decode_device:'configured',vae_dtype:'auto',vae_decode_mode:'auto',tile_height:1024,tile_width:1024,tile_stride_height:768,tile_stride_width:768,retain_latents:true};
function q22SelectedProfile(){const id=$('#qwenProfileSelect').value,p=qwenProfiles.find(x=>x.id===id);if(!p)throw Error(tr('Choose a model profile first.'));return p}
async function q22RefreshProfiles(){const d=await hubApi('profiles'),sel=$('#qwenProfileSelect'),old=sel.value;qwenProfiles=d.profiles||[];sel.replaceChildren(new Option(tr('Choose a model profile…'),''),...qwenProfiles.map(p=>new Option(p.name,p.id)));if(qwenProfiles.some(p=>p.id===old))sel.value=old;}
function q22ProfileEditor(){const p=qwenProfiles.find(x=>x.id===$('#qwenProfileSelect').value);if(p){$('#qwenProfileName').value=p.name;$('#qwenProfileDescription').value=p.description||'';setStatus('qwenProfileStatus',tr('Selected profile: {name} · not activated until you confirm.',{name:p.name}));}if(typeof qwenRenderProfileComponentSummary==='function')qwenRenderProfileComponentSummary();}
function q22FormProfile(){const c=readQwenModelForm();if($('#qwenModelFormat').value==='original')c.model_path='';return {schema:1,edition:'qwen21',name:$('#qwenProfileName').value.trim(),description:$('#qwenProfileDescription').value,config:c,loras:(S.loras||[]).map(x=>({...x}))};}
function q22ApplyModelDecode(cfg){
 const merged={...QWEN22_DEFAULTS,...cfg};const ids={decode_device:'qwenJobDecodeDevice',vae_dtype:'qwenJobDecodeDtype',vae_decode_mode:'vaeDecodeMode',tile_height:'qwenJobTileHeight',tile_width:'qwenJobTileWidth',tile_stride_height:'qwenJobTileStrideHeight',tile_stride_width:'qwenJobTileStrideWidth'};
 for(const [key,id] of Object.entries(ids))$('#'+id).value=merged[key];$('#qwenRetainLatents').checked=merged.retain_latents;saveUiSession();
}
function q22Catalog(){
 const sel=$('#qwenDecodeVae'),old=sel.value;sel.replaceChildren(new Option(tr('Current model configuration'),'configured'),new Option(tr('Official repository VAE'),'original'),...(qwenHub?.local_catalog||[]).filter(x=>x.kind==='vae'&&x.present).map(x=>new Option(x.name,x.id)));
 if([...sel.options].some(x=>x.value===old))sel.value=old;
}
function q22DecodeOptions(){return {decode_device:$('#qwenLabDecodeDevice').value,vae_dtype:$('#qwenLabDecodeDtype').value,vae_decode_mode:$('#qwenLabDecodeMode').value,tile_height:Number($('#qwenLabTileHeight').value),tile_width:Number($('#qwenLabTileWidth').value),tile_stride_height:Number($('#qwenLabTileStrideHeight').value),tile_stride_width:Number($('#qwenLabTileStrideWidth').value)};}
function q22CurrentLatent(){return qwenLatents.find(x=>x.id===$('#qwenLatentSelect').value)}
function q22RenderLatent(){
 const item=q22CurrentLatent();$('#qwenRedecode').disabled=!item;$('#qwenLatentDelete').disabled=!item;
 $('#qwenLatentInfo').textContent=item?`${item.width} × ${item.height} · ${tr('Seed {seed}',{seed:item.seed})} · ${tr('{steps} sampling steps already completed',{steps:item.steps})}\n${item.prompt||''}\nSHA-256: ${item.sha256}`:tr('No saved latents yet. Enable Keep latents and generate an image first.');
 $('#qwenDecodeResults').innerHTML=item?(item.decode_results||[]).filter(x=>x.url&&x.ok).slice(-6).map(x=>`<figure><a href="${esc(x.url)}" target="_blank" rel="noopener"><img class="qwen-alpha-grid" src="${esc(x.url)}" alt="VAE decode comparison" loading="lazy"/></a><figcaption>${esc((x.decode?.vae_path||'').split('/').pop()||'VAE')} · ${esc(tr(x.decode?.decode_device||''))} · ${esc(tr(({auto:'Auto',full_frame:'Full frame',tiled:'Tiled',tiled_custom:'Custom tiles'})[x.decode?.vae_decode_mode]||x.decode?.vae_decode_mode||''))}<br>${Number(x.decode?.decode_seconds||0).toFixed(1)} s · <a data-safe-download href="${esc(x.url)}?download=1">PNG</a></figcaption></figure>`).join(''):'';
}
async function q22RefreshLatents(){
 const pid=S.project?.id;if(!pid){qwenLatents=[];q22RenderLatent();return}
 if(qwenLabBusy)return;qwenLabBusy=true;const epoch=++qwenLabEpoch;
 try{const d=await hubApi('latents?project_id='+encodeURIComponent(pid));if(S.project?.id!==pid||epoch!==qwenLabEpoch)return;
  qwenLatents=d.latents||[];const sel=$('#qwenLatentSelect'),old=sel.value;sel.replaceChildren(new Option(tr('Choose a preserved latent…'),''),...qwenLatents.map(x=>new Option(`${new Date(x.created_at*1000).toLocaleString(window.ZI18n?.language)} · ${x.width}×${x.height} · ${tr('Seed {seed}',{seed:x.seed})}`,x.id)));
  sel.value=qwenLatents.some(x=>x.id===old)?old:(qwenLatents[0]?.id||'');const q=d.quota||{};$('#qwenLatentQuota').textContent=q.unlimited?tr('Latent archive: {count} · {used}. No app quota; {free} free on disk. Deleting latents keeps the PNG images.',{count:q.count||0,used:formatBytes(q.bytes||0),free:formatBytes(q.disk_free_bytes||0)}):tr('Cache: {count}/{max_count} · {used} / {max_bytes}. Deleting latents keeps the PNG images.',{count:q.count||0,max_count:q.max_count||64,used:formatBytes(q.bytes||0),max_bytes:formatBytes(q.max_bytes||1073741824)});q22RenderLatent();
 }finally{qwenLabBusy=false}
}
async function q22PollTest(){
 if(!qwenTestId)return;
 try{const d=await hubApi('components/tests/'+qwenTestId),status=d.job?.status;
  if(status==='completed'&&d.report){setStatus('qwenVaeTestStatus',tr('VAE test passed: strict weights, finite 64×64 RGBA round-trip on CPU. Not a 2048 GPU benchmark.'));showDetails(tr('VAE compatibility test'),d.report);qwenTestId='';sessionStorage.removeItem('qwen22.vaeTest');return}
  if(['failed','cancelled'].includes(status)){setStatus('qwenVaeTestStatus',d.job.error||status,true);qwenTestId='';sessionStorage.removeItem('qwen22.vaeTest');return}
  setStatus('qwenVaeTestStatus',tr('VAE test queued/running on the server. You may leave this page.'));
 }catch(e){setStatus('qwenVaeTestStatus',e.message,true);return}
 qwenTestTimer=setTimeout(q22PollTest,2500);
}
function bindQwenComponents(){
 $('#qwenVaeReset').onclick=()=>{$('#qwenVaePath').value='';$('#qwenVaeConfig').value='';qwenModelDirty=true;if(typeof qwenRenderComponentPickers==='function')qwenRenderComponentPickers();if(typeof qwenRenderProfileComponentSummary==='function')qwenRenderProfileComponentSummary();setStatus('qwenComponentsStatus','Official VAE selected in the form. Save configuration to activate it.');};
 $('#qwenTextureFix').onclick=()=>{qwenModelTab('add');$('#hubUrl').value='https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1';$('#hubKind').value='vae_repository';qwenFormatState();qwenDownloadPreview=null;$('#hubPreview').hidden=true;$('#hubAcceptTerms').checked=false;setStatus('qwenModelStatus','Texture-Fix is optional and not bundled. Its source declares qwen-research / non-commercial research terms. Review the exact revision and licence, then explicitly confirm the download. It is not a guaranteed fix for purple tiling lines or 2048 OOM.');};
 $('#qwenVaeTest').onclick=e=>uiAction(e.currentTarget,async()=>{if(!S.project)throw Error(tr('Open a project first.'));const d=await hubApi('components/vae/test',{method:'POST',body:JSON.stringify({project_id:S.project.id,config:readQwenModelForm()})});qwenTestId=d.id;sessionStorage.setItem('qwen22.vaeTest',qwenTestId);clearTimeout(qwenTestTimer);q22PollTest();projectRecoveryUntil=Date.now()+120000;startPolling();},'qwenVaeTestStatus');
 $('#qwenProfileSelect').onchange=q22ProfileEditor;
 $('#qwenProfileSave').onclick=e=>uiAction(e.currentTarget,async()=>{const body=q22FormProfile();if(!body.name)throw Error(tr('Enter a profile name.'));const d=await hubApi('profiles',{method:'POST',body:JSON.stringify(body)});await q22RefreshProfiles();$('#qwenProfileSelect').value=d.profile.id;setStatus('qwenProfileStatus','Profile saved. It is not automatically activated.');},'qwenProfileStatus');
 $('#qwenProfileReplace').onclick=e=>uiAction(e.currentTarget,async()=>{const p=q22SelectedProfile();if(!confirm(tr('Replace this model profile with the current form?')))return;await hubApi('profiles',{method:'POST',body:JSON.stringify({...q22FormProfile(),id:p.id})});await q22RefreshProfiles();setStatus('qwenProfileStatus','Profile updated.');},'qwenProfileStatus');
 $('#qwenProfileActivate').onclick=e=>uiAction(e.currentTarget,async()=>{const p=q22SelectedProfile();if(!confirm(tr('Activate this component profile? Queued jobs keep their captured configuration. Unsaved model-form changes will be discarded.')))return;const d=await hubApi('profiles/'+p.id+'/activate',{method:'POST',body:JSON.stringify({confirm:true})});qwenModelDirty=false;await refreshModelsHub();q22ApplyModelDecode(d.model.config);S.loras=(d.loras||[]).map(x=>({...x}));qwenControlState();saveUiSession();setStatus('qwenProfileStatus',tr('Active profile: {name}',{name:p.name}));},'qwenProfileStatus');
 $('#qwenProfileDuplicate').onclick=e=>uiAction(e.currentTarget,async()=>{const p=q22SelectedProfile();const d=await hubApi('profiles/'+p.id+'/duplicate',{method:'POST',body:'{}'});await q22RefreshProfiles();$('#qwenProfileSelect').value=d.profile.id;q22ProfileEditor();},'qwenProfileStatus');
 $('#qwenProfileDelete').onclick=e=>uiAction(e.currentTarget,async()=>{const p=q22SelectedProfile();if(!confirm(tr('Delete this profile only? Component files are kept.')))return;await hubApi('profiles/'+p.id,{method:'DELETE',body:'{}'});await q22RefreshProfiles();setStatus('qwenProfileStatus','Profile deleted; files are unchanged.');},'qwenProfileStatus');
 $('#qwenProfileExport').onclick=e=>uiAction(e.currentTarget,async()=>{const p=q22SelectedProfile();const a=document.createElement('a');a.href='/api/model-hub/profiles/'+encodeURIComponent(p.id)+'/download';a.download='qwen-model-profile.json';document.body.append(a);a.click();a.remove();},'qwenProfileStatus');
 $('#qwenProfileImport').onchange=async e=>{try{const f=e.target.files[0];if(!f)return;if(f.size>2*1024**2)throw Error('Profile JSON must be below 2 MiB.');const body=JSON.parse(await f.text());if(!body||Array.isArray(body)||typeof body!=='object')throw Error('Invalid profile JSON.');const d=await hubApi('profiles/import',{method:'POST',body:JSON.stringify(body)});await q22RefreshProfiles();$('#qwenProfileSelect').value=d.profile.id;q22ProfileEditor();setStatus('qwenProfileStatus','Profile imported as a new entry. Review paths and activate explicitly.');}catch(err){setStatus('qwenProfileStatus',err.message,true)}finally{e.target.value=''}};
 $('#qwenDecodeLab').addEventListener('toggle',()=>{if($('#qwenDecodeLab').open)q22RefreshLatents().catch(e=>setStatus('qwenDecodeStatus',e.message,true))});
 $('#qwenLatentsRefresh').onclick=e=>uiAction(e.currentTarget,q22RefreshLatents,'qwenDecodeStatus');$('#qwenLatentSelect').onchange=q22RenderLatent;
 $('#qwenRedecode').onclick=e=>uiAction(e.currentTarget,async()=>{const item=q22CurrentLatent();if(!item||!S.project)throw Error(tr('Select a preserved latent.'));const options=q22DecodeOptions();if(options.vae_decode_mode.startsWith('tiled')&&!confirm(tr('Tiling can reintroduce purple lines/seams. Run this manual tiled comparison?')))return;
  const d=await hubApi('latents/'+item.id+'/decode',{method:'POST',body:JSON.stringify({project_id:S.project.id,vae_selection:$('#qwenDecodeVae').value,options})});setStatus('qwenDecodeStatus',tr('Re-decode queued; 0 sampling steps. Job: {id}',{id:d.id}));projectRecoveryUntil=Date.now()+120000;await openProjectNoPoll(S.project.id);startPolling();},'qwenDecodeStatus');
 $('#qwenLatentDelete').onclick=e=>uiAction(e.currentTarget,async()=>{const item=q22CurrentLatent();if(!item||!S.project)return;if(!confirm(tr('Delete this cached latent? Existing PNG images are kept; re-decode will no longer be possible.')))return;await hubApi('latents/'+item.id,{method:'DELETE',body:JSON.stringify({project_id:S.project.id})});await q22RefreshLatents();},'qwenDecodeStatus');
 window.addEventListener('qwen-hub-updated',()=>{q22Catalog();q22RefreshProfiles().catch(e=>setStatus('qwenProfileStatus',e.message,true));});
 window.addEventListener('studio-project-changed',()=>{qwenLabEpoch++;qwenLatents=[];q22RenderLatent();if($('#qwenDecodeLab').open)q22RefreshLatents().catch(e=>setStatus('qwenDecodeStatus',e.message,true))});
 window.addEventListener('qwen-jobs-updated',()=>{if($('#qwenDecodeLab').open)q22RefreshLatents().catch(e=>setStatus('qwenDecodeStatus',e.message,true))});
 document.addEventListener('change',e=>{if(e.target.id.startsWith('qwenJob')||e.target.id==='qwenRetainLatents')saveUiSession()});
 // Keep name/logo untouched; application header already carries release version.
 try{qwenTestId=sessionStorage.getItem('qwen22.vaeTest')||'';}catch(_){}
 if(qwenTestId)document.addEventListener('DOMContentLoaded',()=>{qwenTestTimer=setTimeout(q22PollTest,3000)});
 q22RenderLatent();
}
bindQwenComponents();
