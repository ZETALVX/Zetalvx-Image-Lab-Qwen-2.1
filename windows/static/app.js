/* Result-card actions: Add input + Reuse settings; editing is available under More actions. */
/* Modified for Zetalvx Image Lab Qwen 1.0.24: interface/localization only; Apache-2.0, retained NOTICE. */
/* Qwen-only form + original Image Lab project/library/job/FFmpeg UI functions. */
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m]));
const tr=(s,vars={})=>window.ZI18n?.t(s,vars)||String(s).replace(/\{(\w+)\}/g,(m,k)=>vars[k]??m);
let S={boot:null,project:null,activeAsset:'',sourceAsset:'',maskAsset:'',references:[],loras:[],poll:null,mediaSettings:{},mask:{sourceId:null,dirty:false,history:[]}};
const creatorViewKey='qwen21_last_view',creatorInitialQuery=new URLSearchParams(location.search);
let qwenSessionTokens=null,qwenSessionTokenFetch=null;
let projectSelectionEpoch=0,projectRefreshEpoch=0,pendingProjectId='',projectLoadController=null,projectPollInFlight=false,projectRecoveryUntil=0,projectRenameId='';
function projectSessionKey(){return S.project?.id||'_global'}
function csrfScope(url,opt){
 if(!['POST','PUT','PATCH','DELETE'].includes((opt.method||'GET').toUpperCase()))return '';
 const p=new URL(url,document.baseURI).pathname;
 if(p.startsWith('/api/model-hub/')||p.startsWith('/api/download-manager/')||p.startsWith('/api/models/'))return 'model-hub';
 if(p==='/api/setup')return 'vision'; // Existing account-bound CSRF scope.
 return '';
}
function generationFeedback(kind,message,bad=false){const el=$('#imageRequestStatus');if(!el)return;el.dataset.sourceMessage=message;el.textContent=tr(message);el.classList.toggle('is-error',bad);el.hidden=!message;}
function showView(name){
 if(!validCreatorView(name))return;rememberCreatorView(name);
 $$('.view').forEach(v=>v.classList.toggle('active',v.id==='view-'+name));$$('.nav').forEach(b=>b.classList.toggle('active',b.dataset.view===name));
 if(window.QwenUI)window.QwenUI.closeMore(false);else{$('#mobileMoreMenu')?.classList.add('hidden');$('#mobileMoreToggle')?.setAttribute('aria-expanded','false');}
 if(name==='library')renderLibraryAssets();if(name==='user-presets')refreshPresets().catch(e=>setStatus('presetStatus',e.message,true));
 if(name==='models')refreshModelsHub().catch(e=>setStatus('qwenModelStatus',e.message,true));if(name==='settings')loadQwenSettings();
 window.dispatchEvent(new CustomEvent('creator-view-change',{detail:{view:name}}));
}
function artifactKind(a){const m=a.metadata||{};if(m.source==='upload')return m.original_name||'Imported image';if(m.source==='media_tools')return ({extract_frame:'Video frame',resize_image:'Resize',convert_image:'Convert',crop_image:'Crop',rotate_image:'Rotate',flip_image:'Flip'})[m.operation]||'Image tool';return m.model_id?'Qwen Image 2.1':m.original_name||'Image'}
function artifactIsImageResult(a){return a?.type==='image'&&a.metadata?.provider==='qwen_image21'}
const bulkState={project:'',library:{enabled:false,selected:new Set(),showHidden:false},jobs:{enabled:false,selected:new Set()}};
function checkBulkProject(){
 const pid=S.project?.id||'';
 if(bulkState.project===pid)return;
 bulkState.project=pid;
 for(const kind of ['library','jobs']){bulkState[kind].enabled=false;bulkState[kind].selected.clear();}
 bulkState.library.showHidden=false;
 const box=$('#libraryShowHidden');if(box)box.checked=false;
}
function bulkCount(kind){const state=bulkState[kind];const bar=$('#'+kind+'BulkActions'),toggle=$('#'+kind+'SelectBtn'),counter=$('#'+kind+'SelectedCount');
 if(bar)bar.hidden=!state.enabled;
 if(toggle){toggle.textContent=tr(state.enabled?'Done':'Select');toggle.setAttribute('aria-pressed',String(state.enabled));}
 if(counter)counter.textContent=tr('{count} selected',{count:state.selected.size});
 const compare=$('#libraryCompareSelected');if(compare)compare.disabled=kind!=='library'||state.selected.size<2||state.selected.size>4||[...state.selected].some(id=>!artifactById(id)||artifactById(id).type!=='image');const ab=$('#libraryCompareAB');if(ab)ab.disabled=kind!=='library'||state.selected.size!==2||[...state.selected].some(id=>!artifactById(id)||artifactById(id).type!=='image');
}
function artifactCard(a,library=false){
 const seed=actualSeed(a),r=a.metadata?.runtime_result||{};
 const isRedecode=a.metadata?.tool==='qwen_decode';
 const label=isRedecode?'Re-decoded image':a.metadata?.source==='upload'?'Input image':a.metadata?.source==='media_tools'?'Image tool':'Qwen image';
 const sel=library&&bulkState.library.enabled?`<label class="bulk-check" title="${esc(tr('Select image'))}"><input type="checkbox" data-bulk-artifact="${esc(a.id)}" aria-label="${esc(tr('Select image'))}" ${bulkState.library.selected.has(a.id)?'checked':''}/></label>`:'';
 const hidden=library&&a.metadata?.library_hidden;
 return `<article class="asset-card ${hidden?'is-library-hidden':''}" data-aid="${esc(a.id)}">${sel}
 ${mediaPreview(a)}<div class="asset-info"><b data-no-i18n>${esc(artifactDisplayTitle(a))}</b>
 <div class="qwen-asset-meta"><span>${tr(label)}</span>${hidden?`<span class="bulk-hidden-badge">${tr('Hidden')}</span>`:''}<code>#${esc(a.id.slice(0,8))}</code>${r.has_transparency?'<span>RGBA</span>':''}</div>
 <small data-no-i18n>${esc(a.created_at)}${seed!==null?' · '+esc(tr('Seed {seed}',{seed})):''}</small>
 ${r.warning?`<small class="qwen-warning" data-no-i18n>${esc(tr(r.warning))}</small>`:''}
 <div class="asset-actions qwen-primary-actions"><button type="button" data-aact="reference" data-id="${esc(a.id)}" title="${esc(tr('Add this image to Qwen inputs; no generation starts.'))}">${tr('Add input')}</button><button type="button" data-areuse="${esc(a.id)}" title="${esc(tr("Restore this result's settings in Create / Edit; no generation starts."))}">${tr('Reuse settings')}</button></div>
 <details class="qwen-asset-actions"><summary>${tr('More actions')}</summary><div class="asset-actions">
 <button type="button" data-adetails="${esc(a.id)}">${tr('Details')}</button><button type="button" data-amediatools="${esc(a.id)}" title="${esc(tr('Open image / FFmpeg tools with this image; no Qwen generation.'))}">${tr('Edit / tools')}</button><a class="qwen-save-action" data-safe-download href="${esc(a.url)}?download=1" download>${tr('Save PNG')}</a><button type="button" class="danger" data-adelete="${esc(a.id)}">${tr('Delete')}</button>
 </div></details></div></article>`;
}
function useAssetAsImageReference(id){addQwenReference(id);showView('image')}
function artifactAction(action,id){if(!artifactById(id))return;if(action==='edit'){sendArtifactToMediaTools(id);return}addQwenReference(id);S.activeAsset=id;renderInputs();showView('image')}
function setStatus(id,message,bad=false){const e=$('#'+id);if(e){e.textContent=tr(message);e.classList.toggle('is-error',bad)}}
async function uiAction(button,fn,status='qwenModelStatus'){
 if(button?.disabled)return;button&&(button.disabled=true);
 try{return await fn()}catch(e){console.error('[Qwen UI]',e);setStatus(status,e.message,true)}finally{button&&(button.disabled=false)}
}
function formatDuration(sec){sec=Number(sec||0);if(sec<60)return `${sec.toFixed(1)}s`;const m=Math.floor(sec/60),s=Math.round(sec%60);return `${m}m ${s}s`}

function formatBytes(n){n=Number(n||0);if(n<1024)return `${n} B`;if(n<1024**2)return `${(n/1024).toFixed(1)} KB`;if(n<1024**3)return `${(n/1024**2).toFixed(1)} MB`;return `${(n/1024**3).toFixed(2)} GB`}

function pct(v){return Math.max(0,Math.min(100,Math.round(Number(v||0)*100)))}

async function sessionTokens(force=false){
 if(force)qwenSessionTokens=null;
 if(qwenSessionTokens)return qwenSessionTokens;
 if(!qwenSessionTokenFetch)qwenSessionTokenFetch=ZetalvxRequestRecovery.fetchText('/api/session/tokens',{cache:'no-store',credentials:'same-origin'}).then(({response:r,text})=>{
  const d=JSON.parse(text);if(!r.ok)throw Error(d.error||'Sign in again');return qwenSessionTokens=d.tokens;
 }).finally(()=>qwenSessionTokenFetch=null);
 return qwenSessionTokenFetch;
}

async function api(url,opt={}){
 const scope=csrfScope(url,opt);
 for(let attempt=0;attempt<2;attempt++){
  const headers=new Headers(opt.headers||{});
  if(scope){const tokens=await sessionTokens(attempt===1);headers.set('X-CSRF-Token',tokens[scope]);}
  const method=String(opt.method||'GET').toUpperCase();
  const body=scope&&opt.body===undefined&&!['GET','HEAD'].includes(method)?'{}':opt.body;
  if(scope&&body!==undefined&&!(body instanceof FormData)&&!headers.has('Content-Type'))headers.set('Content-Type','application/json');
  const {response:r,text}=await ZetalvxRequestRecovery.fetchText(url,{...opt,headers,body});let d={};
  if(text){try{d=JSON.parse(text)}catch(e){throw new Error(`Invalid JSON from ${url} (HTTP ${r.status}): ${text.slice(0,220)}`)}}
  if(r.status===401&&d?.auth_required){window.location.href=`/login?next=${encodeURIComponent(location.pathname+location.search)}`;throw new Error('Authentication required')}
  // Retry ONLY a request rejected by the CSRF guard, before it performed any action.
  if(scope&&attempt===0&&r.status===403&&d.code==='csrf_failed')continue;
  if(!r.ok){const e=new Error(d.error||`HTTP ${r.status}`);e.code=d.code;e.data=d;e.status=r.status;throw e;}return d;
 }
}

function validCreatorView(name){return typeof name==='string'&&/^[a-z][a-z0-9-]*$/.test(name)&&!!document.getElementById('view-'+name);}

function rememberedCreatorView(){try{const v=sessionStorage.getItem(creatorViewKey);return validCreatorView(v)?v:'home'}catch(_){return 'home'}}

function rememberCreatorView(name){try{sessionStorage.setItem(creatorViewKey,name)}catch(_){}}

function consumeCreatorOnboarding(){
 try{const u=new URL(window.location.href);u.searchParams.delete('onboarding');u.searchParams.delete('network_restart');history.replaceState(history.state,'',u.pathname+u.search+u.hash)}catch(_){}
}

function projectSelectionUi(message=''){
 const active=S.project?.id||'';
 $$('[data-open-project]').forEach(el=>{
  const id=el.dataset.openProject,card=el.closest('.project-card');
  el.setAttribute('aria-pressed',String(id===active));el.setAttribute('aria-busy',String(id===pendingProjectId));
  card?.classList.toggle('active',id===active);card?.classList.toggle('pending',id===pendingProjectId);
 });
 const select=$('#libraryProjectSelect');if(select)select.value=pendingProjectId||active;
 const msg=$('#projectSwitchStatus');if(msg)msg.textContent=window.ZI18n?.t(message)||message;
 $('#libraryGrid')?.setAttribute('aria-busy',String(!!pendingProjectId));
}

async function openProject(pid){
 const epoch=++projectSelectionEpoch;projectRefreshEpoch++;
 projectLoadController?.abort();const controller=new AbortController();projectLoadController=controller;
 clearInterval(S.poll);pendingProjectId=pid;projectSelectionUi('Apertura progetto…');
 try{
  const d=await api(`/api/projects/${pid}`,{signal:controller.signal});
  if(epoch!==projectSelectionEpoch)return false;
  if(!d.project||d.project.id!==pid)throw Error('Risposta progetto non valida');
  const changed=S.project?.id!==pid;
  S.project=d.project;S.activeAsset=S.project.state?.active_artifact_id||'';
  if(changed){S.sourceAsset='';S.maskAsset='';S.references=[];S.mask.sourceId=null;S.mask.dirty=false;S.mask.history=[];}
  pendingProjectId='';
  try{localStorage.setItem('qwen21_project_id',pid)}catch(_){}
  $('#topProjectName').textContent=S.project.name;
  renderAssets();renderJobs();renderProjects();renderMediaTools();
  if(changed){renderInputs();window.dispatchEvent(new CustomEvent('studio-project-changed',{detail:{id:pid}}));}
  projectSelectionUi();startPolling();return true;
 }catch(e){
  if(epoch!==projectSelectionEpoch||e.name==='AbortError')return false;
  pendingProjectId='';projectSelectionUi(e.message);startPolling();throw e;
 }
}

async function selectLibraryProject(pid){
 try{if(await openProject(pid)){showView('library');const w=$('.workspace');if(w)w.scrollTop=0;if(matchMedia('(max-width:900px)').matches)window.scrollTo({top:0,behavior:'auto'});}}
 catch(e){if(e?.name==='NetworkError'||e?.name==='TimeoutError'){projectSelectionUi('Connessione temporaneamente interrotta. Riprova tra qualche secondo.');console.warn('[PROJECT RECONNECT]',e);return;}alert(e.message)}
}

function artifactSignature(project){return (project?.artifacts||[]).map(a=>a.id).join("|")}

function generationStatusRecovered(kind){
 const node=$('#'+kind+'RequestStatus');
 if(node?.classList.contains('is-error')&&(/reconnect|not confirmed|Connection interrupted/i).test(node.dataset.sourceMessage||''))
  generationFeedback(kind,'Queue updated. Check the latest job status below.');
}

function startPolling(){
 clearInterval(S.poll);
 S.lastPollArtifacts=artifactSignature(S.project);
 S.lastPollBusy=(S.project?.jobs||[]).some(j=>["queued","running"].includes(j.status));
 S.poll=setInterval(async()=>{
   if(!S.project||pendingProjectId||projectPollInFlight||document.hidden)return;
   const busy=(S.project.jobs||[]).some(j=>["queued","running"].includes(j.status));
   if(!busy&&!S.lastPollBusy&&Date.now()>projectRecoveryUntil)return;
   projectPollInFlight=true;
   try{await pollProjectStatus(S.project.id);S.projectPollFailures=0}catch(e){S.projectPollFailures=(S.projectPollFailures||0)+1;if(S.projectPollFailures===1)generationFeedback('image','Connessione temporaneamente interrotta. Riconnessione automatica in corso.',true);if(S.projectPollFailures===1||S.projectPollFailures%10===0)console.warn('[POLL RETRY]',e)}finally{projectPollInFlight=false}
 },2500)
}

async function pollProjectStatus(pid){
 if(S.project?.id!==pid||pendingProjectId)return;
 const epoch=projectSelectionEpoch,refresh=++projectRefreshEpoch;
 const d=await api(`/api/projects/${pid}`);
 if(epoch!==projectSelectionEpoch||refresh!==projectRefreshEpoch||S.project?.id!==pid||pendingProjectId)return;
 const oldArtifacts=S.lastPollArtifacts;
 const oldActive=S.activeAsset;
 S.project=d.project;
 generationStatusRecovered('image');
 S.activeAsset=S.project.state?.active_artifact_id||S.activeAsset;
 const newArtifacts=artifactSignature(S.project);
 const busy=(S.project.jobs||[]).some(j=>["queued","running"].includes(j.status));

 // Only the queue is redrawn continuously. This avoids image/video reloads,
 // scroll jumps and form disruption while a generation is running.
 renderJobs();
 window.dispatchEvent(new CustomEvent('qwen-jobs-updated'));

 // Refresh media UI one time when an output actually appears.
 if(newArtifacts!==oldArtifacts||S.activeAsset!==oldActive){
   renderStudioResults();
   
 }
 S.lastPollArtifacts=newArtifacts;
 S.lastPollBusy=busy;
}

async function openProjectNoPoll(pid){
 if(S.project?.id!==pid||pendingProjectId)return false;
 const epoch=projectSelectionEpoch,refresh=++projectRefreshEpoch;
 const d=await api(`/api/projects/${pid}`);
 if(epoch!==projectSelectionEpoch||refresh!==projectRefreshEpoch||S.project?.id!==pid||pendingProjectId)return false;
 S.project=d.project;S.activeAsset=S.project.state?.active_artifact_id||S.activeAsset;
 S.lastPollArtifacts=artifactSignature(S.project);
 S.lastPollBusy=(S.project.jobs||[]).some(j=>["queued","running"].includes(j.status));
 renderAssets();renderJobs();renderMediaTools();
 
}

function renderProjects(){
  const box=$("#projectsList") || $("#projectGrid");
  if(!box) return;
  const projects=(S.boot?.projects||[]);
  const active=S.project?.id||"";
  const alive=projects.filter(p=>!Number(p.trashed||0)).sort((a,b)=>(Number(a.sort_order||0)-Number(b.sort_order||0))||String(a.created_at||'').localeCompare(String(b.created_at||'')));
  const trashed=projects.filter(p=>Number(p.trashed||0));
  box.innerHTML=`
    <div class="project-section">
      <div class="section-title">Projects <span class="project-count">${alive.length}</span></div>
      ${alive.map((p,i)=>`
        <div class="project-card ${p.id===active?"active":""}">
          <button type="button" class="project-main" data-open-project="${p.id}" aria-pressed="${p.id===active}">
            <b>${esc(p.name)}</b>
            <small data-no-i18n>${esc(p.description||"")}</small>
          </button>
          <div class="project-actions">
            <button class="ghost small project-icon" data-move-project="${p.id}" data-direction="up" ${i===0?'disabled':''} title="Move up" aria-label="Move up">↑</button>
            <button class="ghost small project-icon" data-move-project="${p.id}" data-direction="down" ${i===alive.length-1?'disabled':''} title="Move down" aria-label="Move down">↓</button>
            <button class="ghost small project-icon" data-rename-project="${p.id}" title="Rename" aria-label="Rename">✎</button>
            <button class="danger ghost small project-icon" data-trash-project="${p.id}" title="Move to Trash" aria-label="Move to Trash">×</button>
          </div>
        </div>`).join("") || '<div class="empty-mini">No active projects.</div>'}
    </div>
    <div class="project-section">
      <div class="section-title">Trash <span class="project-count">${trashed.length}</span></div>
      ${trashed.map(p=>`
        <div class="project-card trashed">
          <div class="project-main">
            <b>${esc(p.name)}</b>
            <small>${p.trashed_at?`Trashed · ${esc(p.trashed_at)}`:"Trashed"}</small>
          </div>
          <div class="project-actions">
            <button class="ghost small" data-restore-project="${p.id}">Restore</button>
            <button class="danger ghost small" data-delete-project="${p.id}">Delete forever</button>
          </div>
        </div>`).join("") || '<div class="empty-mini">Trash is empty.</div>'}
    </div>`;
  const select=$('#libraryProjectSelect');if(select){select.replaceChildren(...alive.map(p=>new Option(p.name,p.id)));select.value=active;select.disabled=!alive.length;}
  $$('[data-open-project]').forEach(el=>el.onclick=()=>selectLibraryProject(el.dataset.openProject));
  $$('.project-card').forEach(card=>{const opener=card.querySelector('[data-open-project]');if(!opener)return;card.onclick=e=>{if(e.target.closest('.project-actions'))return;if(e.target.closest('[data-open-project]'))return;opener.click();};});
  projectSelectionUi();
  $$('[data-trash-project]').forEach(el=>el.onclick=()=>trashProject(el.dataset.trashProject));
  $$('[data-restore-project]').forEach(el=>el.onclick=()=>restoreProject(el.dataset.restoreProject));
  $$('[data-delete-project]').forEach(el=>el.onclick=()=>deleteProjectForever(el.dataset.deleteProject));
  $$('[data-rename-project]').forEach(el=>el.onclick=()=>openProjectRename(el.dataset.renameProject));
  $$('[data-move-project]').forEach(el=>el.onclick=()=>moveProject(el.dataset.moveProject,el.dataset.direction));
}

function mediaPreview(a,cls=""){return '<img src="'+esc(a.url)+'" class="'+esc(cls)+'" alt="Image output" loading="lazy" decoding="async">'}

function actualSeed(a){
 const m=a?.metadata||{},p=m.params||{},r=m.runtime_result||{};
 const s=r.seed!==undefined&&r.seed!==null?r.seed:p.seed;
 return s!==undefined&&s!==null&&Number(s)>=0?Number(s):null;
}

function artifactOrdinal(a){
 const arts=[...(S.project?.artifacts||[])].sort((x,y)=>String(x.created_at||'').localeCompare(String(y.created_at||''))||String(x.id).localeCompare(String(y.id)));
 const i=arts.findIndex(x=>x.id===a.id);return i>=0?i+1:0;
}

function artifactDisplayTitle(a){const n=artifactOrdinal(a),kind=artifactKind(a),label=a.metadata?.original_name?kind:tr(kind);return `${n?String(n).padStart(3,'0')+' · ':''}${label}`}

function bindAssetCards(){
 $$('.asset-card').forEach(x=>x.onclick=e=>{if(e.target.closest('.asset-actions,.qwen-asset-actions,.bulk-check,video,audio'))return;window.CreatorImageViewer?.show(x.querySelector('img'),x)});
 $$('[data-aact]').forEach(b=>b.onclick=e=>{e.stopPropagation();artifactAction(b.dataset.aact,b.dataset.id)});
 $$('#libraryGrid [data-bulk-artifact]').forEach(b=>b.onchange=()=>{if(b.checked)bulkState.library.selected.add(b.dataset.bulkArtifact);else bulkState.library.selected.delete(b.dataset.bulkArtifact);bulkCount('library')});
 for(const [attr,fn] of [['adelete',deleteArtifact],['adetails',showArtifactDetails],['areuse',reuseArtifact],['amediatools',sendArtifactToMediaTools]])$$('[data-'+attr+']').forEach(b=>b.onclick=e=>{e.stopPropagation();fn(b.dataset[attr])});
}

function renderAssetGrid(container,items,empty,library=false){
 if(!container)return;
 // Keep an open action drawer stable when a completed job refreshes the grid.
 const opened=new Set([...container.querySelectorAll('.asset-card')].filter(c=>c.querySelector('.qwen-asset-actions')?.open).map(c=>c.dataset.aid));
 container.innerHTML=items.map(a=>artifactCard(a,library)).join('')||`<div class="empty-state">${tr(empty)}</div>`;
 container.querySelectorAll('.asset-card').forEach(c=>{if(opened.has(c.dataset.aid))c.querySelector('.qwen-asset-actions').open=true});
}
function renderImageStudioResults(){
 const arts=(S.project?.artifacts||[]).filter(artifactIsImageResult);
 renderCreatorPreview(arts);
 renderAssetGrid($("#imageHistoryGrid"),arts.slice(0,12),'No image generations yet.');
}

function renderLibraryAssets(){
 checkBulkProject();
 const arts=S.project?.artifacts||[];
 const visible=arts.filter(a=>bulkState.library.showHidden||!a.metadata?.library_hidden);
 bulkState.library.selected=new Set([...bulkState.library.selected].filter(id=>visible.some(a=>a.id===id)));
 renderAssetGrid($("#libraryGrid"),visible,'No assets in this project.',true);
 renderAssetGrid($("#homeRecent"),arts.filter(a=>!a.metadata?.library_hidden).slice(0,10),'No assets yet.');
 bulkCount('library');
 bindAssetCards();
}

function renderStudioResults(){renderImageStudioResults();bindAssetCards()}

function renderAssets(){renderLibraryAssets();renderStudioResults()}

function sendArtifactToMediaTools(aid){
 const a=artifactById(aid);
 if(!a||!["image"].includes(a.type)){alert("Choose an image asset.");return}
 S.activeAsset=aid;
 // Keep the existing operation/options, but select the image that was clicked.
 captureMediaToolState();
 mediaState().source_ids=[aid];
 renderMediaTools([aid],false);
 saveUiSession();
 showView("media-tools");if($("#imageToolsPanel"))$("#imageToolsPanel").open=true;
 api(`/api/projects/${S.project.id}/active-artifact`,{
   method:"PUT",
   headers:{"Content-Type":"application/json"},
   body:JSON.stringify({artifact_id:aid})
 }).catch(()=>{});
}

function currentArtifact(){return (S.project?.artifacts||[]).find(a=>a.id===S.activeAsset)}


function jobDetailsObject(j){
 return {id:j.id,status:j.status,phase:j.phase,model_id:j.model_id,tool:j.tool,prompt:j.prompt,
         duration_seconds:j.duration_seconds,created_at:j.created_at,started_at:j.started_at,completed_at:j.completed_at,
         output_artifact_id:j.output_artifact_id,params:j.params||{},error:j.error||""}
}

let jobsStatusFilter='all';
function progressJob(j,compact=false){
 const p=pct(j.progress),cur=Number(j.step_current||0),tot=Number(j.step_total||0);
 const elapsed=Number(j.elapsed_seconds||j.duration_seconds||0);
 const queued=j.status==='queued',running=['running','cancelling'].includes(j.status);
 const terminal=['completed','failed','cancelled','interrupted'].includes(j.status);
 const select=!compact&&bulkState.jobs.enabled?`<label class="bulk-check jobs-bulk-check" title="${esc(tr('Select job'))}"><input type="checkbox" data-bulk-job="${esc(j.id)}" aria-label="${esc(tr('Select job'))}" ${bulkState.jobs.selected.has(j.id)?'checked':''}/></label>`:'';
 const tool=j.tool==='qwen_prompt_enhance'?tr('Prompt Enhancer'):(j.model_id==='qwen_image_21'?tr('Image'):j.model_id);
 return `<div class="queue-job ${esc(j.status)}" data-job-id="${esc(j.id)}">
   <div class="queue-job-head">${select}<div><b>${esc(tool)}</b><small data-no-i18n>${esc(j.prompt||'').slice(0,150)}</small></div><span>${esc(tr(j.phase||j.status))} · ${p}%</span></div>
   ${j.params?.auto_test?`<div class="job-test-label"><b>${esc(window.ZI18n?.t('Test {index}/{total}',{index:j.params.test_index,total:j.params.test_total})||`Test ${j.params.test_index}/${j.params.test_total}`)}</b> · ${esc(j.params.test_label||'')}</div>`:''}
   <div class="progress-track"><div class="progress-fill" style="width:${p}%"></div></div>
   <div class="queue-job-foot"><span>${tot?tr('{current}/{total} steps',{current:cur,total:tot})+' · ':''}${formatDuration(elapsed)}</span>
   <div class="jobs-row-actions"><button class="mini" type="button" data-jdetails="${esc(j.id)}">${tr('Details')}</button>
   ${queued&&!compact?`<button class="mini" data-jmove="up" data-jid="${esc(j.id)}" title="${esc(tr('Move earlier in queue'))}">↑</button><button class="mini" data-jmove="down" data-jid="${esc(j.id)}" title="${esc(tr('Move later in queue'))}">↓</button><button class="mini" data-jedit="${esc(j.id)}">${tr('Edit prompt')}</button>`:''}
   ${queued||running?`<button class="mini danger" data-jcancel="${esc(j.id)}">${tr('Cancel')}</button>`:''}
   ${terminal&&!compact?`<button class="mini" data-jhide="${esc(j.id)}">${tr('Remove from list')}</button>${!j.params?._benchmark_id?`<button class="mini danger" data-jdelete="${esc(j.id)}">${tr('Delete record')}</button>`:''}`:''}
   </div></div>
 </div>`
}

function sortActiveJobs(jobs){return jobs.slice().sort((a,b)=>(a.queue_order||0)-(b.queue_order||0));}
function renderLiveQueue(){
 const jobs=(S.project?.jobs||[]).filter(j=>['queued','running','cancelling'].includes(j.status));
 $('#queueCount').textContent=tr('{count} jobs',{count:jobs.length});
 const active=jobs.filter(j=>j.status==='running'||j.status==='cancelling');
 const queued=sortActiveJobs(jobs.filter(j=>j.status==='queued'));
 $('#liveQueue').innerHTML=jobs.length?[...active,...queued].map(j=>progressJob(j,true)).join(''):'<div class="empty-mini">Queue empty.</div>';
 bindJobButtons();
}

function renderJobs(){
 checkBulkProject();
 const jobs=S.project?.jobs||[];
 bulkState.jobs.selected=new Set([...bulkState.jobs.selected].filter(id=>jobs.some(j=>j.id===id)));
 const active=sortActiveJobs(jobs.filter(j=>j.status==='queued'||j.status==='running'||j.status==='cancelling'));
 const finished=jobs.filter(j=>!active.includes(j));
 const ordered=[...active,...finished];
 const visible=ordered.filter(j=>jobsStatusFilter==='all'||(jobsStatusFilter==='active'&&['queued','running','cancelling'].includes(j.status))||
   (jobsStatusFilter==='completed'&&j.status==='completed')||
   (jobsStatusFilter==='failed'&&['failed','interrupted','cancelled'].includes(j.status)));
 $('#jobsList').innerHTML=visible.length?visible.map(j=>progressJob(j)).join(''):'<div class="empty-state">No jobs in this view.</div>';
 renderLiveQueue();bindJobButtons();bulkCount('jobs');
}

async function refreshProjectJobs(){
 if(!S.project)return;
 await openProjectNoPoll(S.project.id);
}
function bindJobButtons(){
 const pid=S.project?.id;if(!pid)return;
 $$('#jobsList [data-bulk-job]').forEach(b=>b.onchange=()=>{if(b.checked)bulkState.jobs.selected.add(b.dataset.bulkJob);else bulkState.jobs.selected.delete(b.dataset.bulkJob);bulkCount('jobs')});
 $$('[data-jcancel]').forEach(b=>b.onclick=async()=>{if(!S.project||S.project.id!==pid)return;
   if(!confirm(tr('Cancel this job?')))return;
   try{await api(`/api/projects/${pid}/jobs/${b.dataset.jcancel}/cancel`,{method:'POST'});await refreshProjectJobs();}catch(e){setStatus('jobsActionStatus',e.message,true)}
 });
 $$('[data-jdetails]').forEach(b=>b.onclick=()=>{const j=(S.project?.jobs||[]).find(x=>x.id===b.dataset.jdetails);if(j)showDetails(`${tr('Job')} #${j.id.slice(0,8)}`,jobDetailsObject(j))});
 $$('[data-jmove]').forEach(b=>b.onclick=async()=>{
   if(S.project?.id!==pid)return;
   try{await api(`/api/projects/${pid}/jobs/${b.dataset.jid}/move`,{method:'POST',body:JSON.stringify({direction:b.dataset.jmove})});await refreshProjectJobs()}catch(e){setStatus('jobsActionStatus',e.message,true)}
 });
 $$('[data-jedit]').forEach(b=>b.onclick=async()=>{
   const j=(S.project?.jobs||[]).find(x=>x.id===b.dataset.jedit);if(!j||S.project?.id!==pid)return;
   const promptNew=prompt(tr('Edit queued job prompt'),j.params?.user_prompt||j.prompt||'');
   if(promptNew===null||promptNew===j.params?.user_prompt||promptNew===j.prompt)return;
   try{await api(`/api/projects/${pid}/jobs/${j.id}`,{method:'PATCH',body:JSON.stringify({prompt:promptNew})});await refreshProjectJobs()}catch(e){setStatus('jobsActionStatus',e.message,true)}
 });
 $$('[data-jdelete]').forEach(b=>b.onclick=async()=>{
   if(S.project?.id!==pid||!confirm(tr('Permanently delete this finished job record? The generated image remains in your Library.')))return;
   try{await api(`/api/projects/${pid}/jobs/${b.dataset.jdelete}`,{method:'DELETE'});await refreshProjectJobs()}catch(e){setStatus('jobsActionStatus',e.message,true)}
 });
 $$('[data-jhide]').forEach(b=>b.onclick=async()=>{
   if(S.project?.id!==pid||!confirm(tr('Remove this job from the Jobs list? Generated images and benchmark reports will be kept.')))return;
   try{await api(`/api/projects/${pid}/jobs/${b.dataset.jhide}/hide`,{method:'POST'});await refreshProjectJobs()}catch(e){setStatus('jobsActionStatus',e.message,true)}
 });
}

function bulkVisibleIds(kind){
 const container=kind==='jobs'?'#jobsList':'#libraryGrid';
 return [...new Set($$(container+' [data-bulk-'+(kind==='jobs'?'job':'artifact')+']').map(el=>el.dataset[kind==='jobs'?'bulkJob':'bulkArtifact']))];
}
function configureBulkSelection(){
 for(const kind of ['jobs','library']){
  $('#'+kind+'SelectBtn').onclick=()=>{checkBulkProject();const s=bulkState[kind];s.enabled=!s.enabled;s.selected.clear();kind==='jobs'?renderJobs():renderLibraryAssets()};
  $('#'+kind+'SelectAll').onclick=()=>{const s=bulkState[kind];bulkVisibleIds(kind).forEach(id=>s.selected.add(id));kind==='jobs'?renderJobs():renderLibraryAssets()};
  $('#'+kind+'DeselectAll').onclick=()=>{bulkState[kind].selected.clear();kind==='jobs'?renderJobs():renderLibraryAssets()};
 }
 const openLibraryComparison=mode=>{
  checkBulkProject();const arts=[...bulkState.library.selected].map(artifactById).filter(Boolean);
  if(arts.length<2||arts.length>4||(mode==='ab'&&arts.length!==2)||arts.some(x=>x.type!=='image')){
   setStatus('libraryBulkStatus','Select 2–4 images (exactly 2 for A/B).',true);return;
  }
  if(!window.QwenComparison?.compareLibrary){setStatus('libraryBulkStatus','Comparison viewer is not available.',true);return;}
  try{window.QwenComparison.compareLibrary(arts,mode)}catch(e){setStatus('libraryBulkStatus',e.message,true)}
 };
 $('#libraryCompareSelected').onclick=()=>openLibraryComparison('grid');
 $('#libraryCompareAB').onclick=()=>openLibraryComparison('ab');
 $('#libraryShowHidden').onchange=e=>{checkBulkProject();bulkState.library.showHidden=e.target.checked;bulkState.library.selected.clear();renderLibraryAssets()};
 $$('[data-jobs-bulk],[data-library-bulk]').forEach(button=>button.onclick=async()=>{
  const kind=button.dataset.jobsBulk?'jobs':'library',action=button.dataset.jobsBulk||button.dataset.libraryBulk;
  const pid=S.project?.id,ids=[...bulkState[kind].selected];if(!pid||pid!==bulkState.project||!ids.length)return;
  const description=({jobs:{cancel:'Request cancellation of selected jobs?',hide:'Hide selected finished jobs? Generated images and benchmark reports will be kept.',delete:'Permanently delete the selected finished job records? Generated images will be kept; benchmark records cannot be deleted here.'},library:{hide:'Hide selected images from the Library? Files will be kept.',restore:'Restore selected hidden images?',delete:'Permanently delete the selected images and files? This cannot be undone.'}})[kind][action];
  if(!confirm(tr(description)))return;
  button.disabled=true;
  try{
   const response=await api(`/api/projects/${pid}/${kind==='jobs'?'jobs':'artifacts'}/bulk`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ids,action})});
   if(S.project?.id!==pid)return;
   const errors=response.errors||[],failedIds=new Set(errors.map(e=>e.id));
   bulkState[kind].selected=new Set(ids.filter(id=>failedIds.has(id)));
   if(kind==='library'&&action==='delete'){
    const removed=new Set(ids.filter(id=>!failedIds.has(id)));
    if(removed.has(S.activeAsset))S.activeAsset='';
    if(removed.has(S.sourceAsset))S.sourceAsset='';
    if(removed.has(S.maskAsset))S.maskAsset='';
    S.references=S.references.filter(id=>!removed.has(id));
   }
   await openProjectNoPoll(pid);
   if(kind==='library'&&action==='delete')renderInputs();
   const msg=`${response.processed} ${tr('items processed')}${errors.length?'; '+errors.length+' '+tr('not processed')+': '+errors.slice(0,4).map(e=>e.id.slice(0,8)+' '+e.error).join(' | '):''}`;
   setStatus(kind==='jobs'?'jobsActionStatus':'libraryBulkStatus',msg,Boolean(errors.length));
  }catch(err){setStatus(kind==='jobs'?'jobsActionStatus':'libraryBulkStatus',err.message,true)}
  finally{button.disabled=false;}
 });
}

async function uploadFile(file,role){
 if(!S.project)throw new Error("Open a project first");
 const fd=new FormData();fd.append("file",file);fd.append("role",role);
 const d=await api(`/api/projects/${S.project.id}/upload`,{method:"POST",body:fd});
 await openProjectNoPoll(S.project.id);return d.artifact;
}

function artifactById(id){return (S.project?.artifacts||[]).find(a=>a.id===id)}

function artifactExists(id){return !!(S.project?.artifacts||[]).find(a=>a.id===id)}

async function deleteArtifact(id){
 const a=(S.project?.artifacts||[]).find(x=>x.id===id);if(!a)return;
 if(!confirm(tr('Delete image #{id} and its file?',{id:id.slice(0,8)})))return;
 await api(`/api/artifacts/${id}`,{method:"DELETE"});
 if(S.activeAsset===id)S.activeAsset="";
 if(S.sourceAsset===id)S.sourceAsset="";
 if(S.maskAsset===id)S.maskAsset="";
 S.references=S.references.filter(x=>x!==id);
 await openProjectNoPoll(S.project.id);renderInputs();
}

/* Structured, user-friendly details. Full original JSON stays available in the
   opt-in technical section for support and advanced debugging. */
function prettyInfoKey(k){
 const labels={id:'ID',model_id:'Model',model_name:'Model',tool:'Operation',status:'Status',phase:'Phase',prompt:'Prompt',
 created_at:'Created',started_at:'Started',completed_at:'Completed',duration_seconds:'Duration',
 output_artifact_id:'Output image',type:'Type',path:'Server file',actual_seed:'Actual seed',requested_seed:'Requested seed',
 width:'Width',height:'Height',output_width:'Width',output_height:'Height',steps:'Steps',cfg:'CFG',cfg_enabled:'CFG enabled',
 negative_prompt:'Negative prompt',seed:'Seed',loras:'LoRA',strength:'Strength',vae_decode_mode:'VAE decode',
 qwen21_transformer_mode:'Transformer mode',qwen21_transformer_path:'Transformer file',
 qwen21_text_encoder_path:'Text Encoder',qwen21_vae_path:'VAE',peak_rss_bytes:'Peak RAM',peak_cuda_allocated_bytes:'Peak VRAM (allocated)',
 peak_cuda_reserved_bytes:'Peak VRAM (reserved)',execution_backend:'Backend',precision_policy:'Precision',
 memory_mode:'Memory mode',gpu_memory:'GPU memory limit',cpu_memory:'RAM limit'};
 return tr(labels[k]||String(k).replace(/^_+/,'').replace(/_/g,' ').replace(/\b\w/g,m=>m.toUpperCase()));
}
function prettyInfoValue(key,value){
 if(value===null||value===undefined||value==='')return '—';
 if(typeof value==='boolean')return value?tr('Yes'):tr('No');
 if(key==='duration_seconds'&&typeof value==='number')return formatDuration(value);
 if((key.endsWith('_bytes')||['peak_rss_bytes','peak_cuda_allocated_bytes'].includes(key))&&typeof value==='number')return formatBytes(value);
 if(Array.isArray(value)){
   if(key==='loras')return value.length?value.map(x=>typeof x==='object'?`${x.name||x.path||'LoRA'} (${x.strength??1})`:String(x)).join(' · '):tr('None');
   return value.length?value.map(x=>typeof x==='object'?JSON.stringify(x):String(x)).join(' · '):tr('None');
 }
 if(typeof value==='object')return JSON.stringify(value);
 return String(value);
}
function infoRows(data,keys=null){
 if(!data||typeof data!=='object')return '';
 const entries=(keys||Object.keys(data)).filter(k=>Object.prototype.hasOwnProperty.call(data,k)&&data[k]!==null&&data[k]!==undefined&&data[k]!=='');
 return entries.map(k=>`<div class="info-pair"><dt>${esc(prettyInfoKey(k))}</dt><dd data-no-i18n>${esc(prettyInfoValue(k,data[k]))}</dd></div>`).join('');
}
function infoSection(title,data,keys=null,open=true){
 const content=infoRows(data,keys);if(!content)return '';
 return `<section class="info-section"><h4>${esc(tr(title))}</h4><dl class="info-grid">${content}</dl></section>`;
}
function showDetails(title,obj){
 $('#detailsTitle').textContent=title;
 $('#detailsBody').textContent=JSON.stringify(obj,null,2);
 const meta=obj?.metadata||{},params=obj?.params||meta.params||{};
 const result=meta.runtime_result||obj?.runtime_result||{};
 const a=(S.project?.artifacts||[]).find(x=>x.id===obj?.id||x.id===obj?.output_artifact_id);
 let html='';
 if(a?.type==='image'&&a.url){
   html+=`<div class="info-image-preview"><img loading="lazy" alt="${esc(tr('Generated image'))}" src="${esc(a.url)}"></div>`;
 }
 if(Array.isArray(obj)){
   html+=obj.map((entry,i)=>infoSection(`${tr('Item')} ${i+1}`,typeof entry==='object'?entry:{value:entry})).join('');
 }else if(obj&&typeof obj==='object'){
   html+=infoSection('General information',obj,['id','status','phase','type','model_name','model_id','tool','prompt','created_at','started_at','completed_at','duration_seconds','output_artifact_id','actual_seed','requested_seed']);
   html+=infoSection('Generation settings',params,['width','height','steps','cfg_enabled','cfg','negative_prompt','seed','loras','strength','qwen21_transformer_mode','qwen21_transformer_path','qwen21_text_encoder_path','qwen21_vae_path','vae_decode_mode','decode_device','vae_dtype']);
   html+=infoSection('Image information',meta,['prompt','model_name','provider','output_width','output_height','requested_width','requested_height','duration_seconds','source_artifact_id','reference_artifact_ids']);
   html+=infoSection('Runtime',result,['execution_backend','backend','precision_policy','native_int8_benchmark','worker_seconds','sampling_seconds','decode_seconds','load_seconds']);
   html+=infoSection('Runtime and memory',result.metrics||result,['peak_rss_bytes','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','peak_gpu_process_bytes','gpu','memory_scope']);
   html+=infoSection('Memory settings',params._model_config_snapshot,['memory_mode','dtype','cpu_offload','gpu_memory','cpu_memory','vae_decode_mode','decode_device']);
   if(obj.error)html+=`<section class="info-section info-error"><h4>${esc(tr('Error'))}</h4><p data-no-i18n>${esc(String(obj.error).split('\nTraceback')[0])}</p></section>`;
   const common=new Set(['id','status','phase','type','model_name','model_id','tool','prompt','created_at','started_at','completed_at','duration_seconds','output_artifact_id','actual_seed','requested_seed','params','metadata','runtime_result','error']);
   const other={};for(const [k,v] of Object.entries(obj)){if(common.has(k))continue;
     if(v&&typeof v==='object'&&!Array.isArray(v))html+=infoSection(prettyInfoKey(k),v);
     else other[k]=v;
   }
   html+=infoSection('Additional information',other);
   if(!html)html=infoSection('Information',obj);
 }else html=infoSection('Information',{value:obj});
 $('#detailsSummary').innerHTML=html||`<p>${esc(tr('No information available.'))}</p>`;
 $('#detailsModal').querySelector('.details-raw').open=false;
 $('#detailsModal').classList.remove('hidden');
}

function showArtifactDetails(id){
 const a=artifactById(id);if(!a)return;
 showDetails(`${tr('Image')} #${id.slice(0,8)}`,{id:a.id,type:a.type,created_at:a.created_at,path:a.path,actual_seed:actualSeed(a),requested_seed:a.metadata?.params?.seed??null,metadata:a.metadata||{}});
}

function mediaToolAssets(){return (S.project?.artifacts||[]).filter(a=>a.type==='image')}

function mediaToolAssetName(a){return a?.metadata?.original_name||a?.metadata?.model_name||`#${a?.id?.slice(0,8)||"asset"}`}

function mediaToolAllowedOperations(){return ['resize_image','convert_image','crop_image','rotate_image','flip_image']}

function mediaStateKey(){return projectSessionKey()}

function mediaState(){return S.mediaSettings[mediaStateKey()]||(S.mediaSettings[mediaStateKey()]={source_ids:[],operation:"",params:{}})}

function selectedMediaIds(){
 const el=$("#mtSource");if(!el)return [];
 return [...el.selectedOptions].map(o=>o.value).filter(Boolean);
}

function setSelectedMediaIds(ids){
 const wanted=new Set(ids||[]);
 if(!$("#mtSource"))return;
 [...$("#mtSource").options].forEach(o=>o.selected=wanted.has(o.value));
}

function captureMediaToolState(){
 if(!$("#mtSource"))return;const st=mediaState(),selected=selectedMediaIds();
 if(selected.length)st.source_ids=selected;
 st.operation=$("#mtOperation")?.value||st.operation||"";
 const p={...(st.params||{})};
 for(const [id,key] of [["mtWidth","width"],["mtHeight","height"],["mtFps","fps"],["mtFormat","format"],["mtTimestamp","timestamp"],["mtStart","start"],["mtEnd","end"],["mtDuration","duration"],["mtVolumeDb","volume_db"]]){
   const el=$("#"+id);if(el)p[key]=el.value;
 }
 st.params=p;saveUiSession();
}

function mediaOperationIntersection(arts){
 if(!arts.length)return [];
 let common=new Set(mediaToolAllowedOperations(arts[0]));
 for(const a of arts.slice(1))common=new Set([...common].filter(x=>mediaToolAllowedOperations(a).includes(x)));
 return [...common];
}

function renderMediaToolSourcePreview(){
 const box=$("#mtSourcePreview");if(!box)return;
 const arts=selectedMediaIds().map(artifactById).filter(Boolean);
 if(!arts.length){box.innerHTML='<div class="empty-mini">Upload files or choose one or more project assets.</div>';return}
 box.innerHTML=`<div class="media-batch-summary"><b>${arts.length} file${arts.length===1?"":"s"} selected</b><small>${arts.map(a=>esc(mediaToolAssetName(a))).join(" · ")}</small></div>`+
   arts.slice(0,6).map(a=>`<div class="media-tool-selected">${mediaPreview(a,"media-tool-selected-preview")}<div><b>${esc(mediaToolAssetName(a))}</b><small>${esc(a.type)} · #${a.id.slice(0,8)}</small></div></div>`).join("")+
   (arts.length>6?`<div class="empty-mini">+ ${arts.length-6} more</div>`:"");
}

function syncMediaToolOperations(preferred=""){
 const arts=selectedMediaIds().map(artifactById).filter(Boolean),allowed=mediaOperationIntersection(arts),sel=$("#mtOperation");if(!sel)return;
 [...sel.options].forEach(o=>o.hidden=!allowed.includes(o.value));
 const wanted=preferred||mediaState().operation||sel.value;
 sel.value=allowed.includes(wanted)?wanted:(allowed.includes(sel.value)?sel.value:(allowed[0]||""));
 mediaState().operation=sel.value;renderMediaToolFields(true);renderMediaToolSourcePreview();captureMediaToolState();
}

function renderMediaTools(preferIds=[],keepState=true){
 if(!$("#mtSource"))return;if(keepState)captureMediaToolState();
 const arts=mediaToolAssets(),st=mediaState();
 const preferred=Array.isArray(preferIds)?preferIds:(preferIds?[preferIds]:[]);
 const previous=preferred.length?preferred:(st.source_ids||[]);
 $("#mtSource").innerHTML=arts.length?arts.map(a=>`<option value="${a.id}">${esc(mediaToolAssetName(a))} · ${esc(a.type)}</option>`).join(""):'<option value="">No media in this project</option>';
 const existing=previous.filter(id=>arts.some(a=>a.id===id));
 if(existing.length)setSelectedMediaIds(existing);
 else if(arts.length)setSelectedMediaIds([arts[0].id]);
 st.source_ids=selectedMediaIds();
 $("#mtSourceHelp").textContent=arts.length?`${arts.length} media assets available. Ctrl/Cmd-click or tap selections to process multiple compatible files together.`:"No media yet. Upload one or more files.";
 syncMediaToolOperations(st.operation);
}

function renderMediaToolFields(){
 const op=$('#mtOperation')?.value;
 const size='<div class="control-grid"><label>Larghezza<input id="mtWidth" value="1024" type="number" min="1"></label><label>Altezza<input id="mtHeight" value="1024" type="number" min="1"></label></div>';
 $('#mtFields').innerHTML=op==='resize_image'?size:op==='crop_image'?size+'<label>Posizione X<input id="mtX" type="number" value="0" min="0"></label><label>Posizione Y<input id="mtY" type="number" value="0" min="0"></label>':op==='rotate_image'?'<label>Angolo<select id="mtDegrees"><option value="90">90° antiorario</option><option value="-90">90° orario</option><option value="180">180°</option></select></label>':op==='flip_image'?'<small>Specchia l’immagine in orizzontale.</small>':'<label>Formato<select id="mtFormat"><option>png</option><option>jpg</option><option>webp</option></select></label>';
}

async function uploadMediaToolFileQuiet(file){
 if(!S.project)throw new Error("Open a project first");
 const fd=new FormData();fd.append("file",file);fd.append("role","media_tool_source");
 const d=await api(`/api/projects/${S.project.id}/upload`,{method:"POST",body:fd});
 return d.artifact;
}

async function refreshAfterProjectMutation(){
  const b=await api("/api/bootstrap");
  S.boot=b;
  const alive=(b.projects||[]).filter(p=>!Number(p.trashed||0));
  const currentAlive=S.project && alive.some(p=>p.id===S.project.id);
  if(currentAlive){
    await openProject(S.project.id);
  }else if(alive.length){
    await openProject(alive[0].id);
  }else{
    projectSelectionEpoch++;projectRefreshEpoch++;projectLoadController?.abort();pendingProjectId="";clearInterval(S.poll);S.project=null;S.activeAsset="";localStorage.removeItem("qwen21_project_id");
    $("#topProjectName").textContent="—";
    renderProjects();renderAssets();renderJobs();renderMediaTools();
  }
}

function openProjectRename(pid){
 const p=(S.boot?.projects||[]).find(x=>x.id===pid);if(!p)return;
 projectRenameId=pid;$('#projectRenameName').value=p.name||'';$('#projectRenameDescription').value=p.description||'';$('#projectRenameModal').classList.remove('hidden');$('#projectRenameName').focus();
}

async function moveProject(pid,direction){await api(`/api/projects/${pid}/move`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({direction})});await refreshAfterProjectMutation();}

async function trashProject(pid){
  if(!confirm("Move this project to Trash? Its files remain recoverable until you choose Delete forever."))return;
  await api(`/api/projects/${pid}/trash`,{method:"POST"});
  await refreshAfterProjectMutation();
}

async function restoreProject(pid){
  await api(`/api/projects/${pid}/restore`,{method:"POST"});
  await refreshAfterProjectMutation();
}

async function deleteProjectForever(pid){
  if(!confirm("DELETE FOREVER? This permanently removes the project, its generated files, uploads, jobs and conversation history. This cannot be undone."))return;
  await api(`/api/projects/${pid}/delete-forever`,{method:"POST"});
  await refreshAfterProjectMutation();
}
$("#mtSource")?.addEventListener("change",()=>{captureMediaToolState();mediaState().source_ids=selectedMediaIds();syncMediaToolOperations("")});
$("#mtOperation")?.addEventListener("change",()=>{captureMediaToolState();mediaState().operation=$("#mtOperation").value;renderMediaToolFields(true);captureMediaToolState()});
$("#mtFields")?.addEventListener("input",captureMediaToolState);$("#mtFields")?.addEventListener("change",captureMediaToolState);


$("#mtUpload")?.addEventListener("change",async e=>{
 try{
   captureMediaToolState();const files=[...e.target.files];if(!files.length)return;
   const oldOp=mediaState().operation,oldParams={...(mediaState().params||{})},newIds=[];
   $("#mtRunBtn").disabled=true;$("#mtRunBtn").textContent=`Uploading ${files.length}…`;
   for(const f of files){const a=await uploadMediaToolFileQuiet(f);newIds.push(a.id)}
   e.target.value="";await openProjectNoPoll(S.project.id);
   mediaState().source_ids=newIds;mediaState().operation=oldOp;mediaState().params=oldParams;saveUiSession();
   renderMediaTools(newIds,true);showView("media-tools");
 }catch(err){alert(err.message)}
 finally{$("#mtRunBtn").disabled=false;$("#mtRunBtn").textContent="Run media tool"}
});

$("#mtRunBtn")?.addEventListener("click",async()=>{
 try{
   captureMediaToolState();const sourceIds=selectedMediaIds(),operation=$("#mtOperation")?.value||"";
   if(!sourceIds.length)throw new Error("Choose one or more source assets.");
   if(!operation)throw new Error("The selected files do not share a compatible operation.");
   const params={};
   if($("#mtWidth"))params.width=Number($("#mtWidth").value);if($("#mtHeight"))params.height=Number($("#mtHeight").value);
   for(const [id,key] of [['mtX','x'],['mtY','y'],['mtDegrees','degrees']])if($('#'+id))params[key]=Number($('#'+id).value);
   if($("#mtFps"))params.fps=Number($("#mtFps").value);if($("#mtFormat"))params.format=$("#mtFormat").value;
   if($("#mtTimestamp"))params.timestamp=$("#mtTimestamp").value;if($("#mtStart"))params.start=$("#mtStart").value;
   if($("#mtEnd"))params.end=$("#mtEnd").value;if($("#mtDuration"))params.duration=$("#mtDuration").value;
   if($("#mtVolumeDb"))params.volume_db=Number($("#mtVolumeDb").value);
   $("#mtRunBtn").disabled=true;$("#mtRunBtn").textContent=`Processing ${sourceIds.length} file${sourceIds.length===1?"":"s"}…`;
   const keepOp=operation,keepParams={...params},keepIds=[...sourceIds];
   const d=await api("/api/media-tools/run",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({project_id:S.project?.id,source_artifact_ids:sourceIds,operation,params})});
   await openProjectNoPoll(S.project.id);
   mediaState().source_ids=keepIds;mediaState().operation=keepOp;mediaState().params=keepParams;saveUiSession();
   renderMediaTools(keepIds,true);showView("media-tools");
   if((d.errors||[]).length)alert(`Completed ${d.artifacts?.length||0}/${sourceIds.length}. Errors:\\n${d.errors.map(x=>`${x.source_id}: ${x.error}`).join("\\n")}`);
 }catch(err){alert(err.message)}
 finally{$("#mtRunBtn").disabled=false;$("#mtRunBtn").textContent="Run media tool"}
});


$('#cancelProjectRename').onclick=()=>{$('#projectRenameModal').classList.add('hidden');projectRenameId=''};
$('#saveProjectRename').onclick=async()=>{if(!projectRenameId)return;const name=$('#projectRenameName').value.trim();if(!name){alert('Project name cannot be empty');return}await api(`/api/projects/${projectRenameId}`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,description:$('#projectRenameDescription').value})});$('#projectRenameModal').classList.add('hidden');projectRenameId='';await refreshAfterProjectMutation();};


const QWEN_FIELDS={width:'width',height:'height',seed:'seed',steps:'steps',cfg:'cfg',batch:'batch',negative_prompt:'negativePrompt',qwen21_transformer_mode:'qwenTransformerMode',qwen21_transformer_path:'qwenTransformerPath',qwen21_text_encoder_path:'qwenJobTextEncoder',qwen21_vae_path:'qwenJobVae',qwen21_vae_decode_mode:'vaeDecodeMode',decode_device:'qwenJobDecodeDevice',vae_dtype:'qwenJobDecodeDtype',tile_height:'qwenJobTileHeight',tile_width:'qwenJobTileWidth',tile_stride_height:'qwenJobTileStrideHeight',tile_stride_width:'qwenJobTileStrideWidth'};
function qwenParams(){
 const p={};for(const [key,id] of Object.entries(QWEN_FIELDS)){const el=$('#'+id);p[key]=['width','height','seed','steps','cfg','batch','tile_height','tile_width','tile_stride_height','tile_stride_width'].includes(key)?Number(el.value):el.value;}
 p.retain_latents=$('#qwenRetainLatents').checked;p.oom_cpu_decode_fallback=$('#qwenOomCpuFallback').checked;
 p.cfg_enabled=$('#qwenEnableCfg').checked;if(!p.cfg_enabled){p.cfg=1;p.negative_prompt=''}
 p.transparent_output=$('#transparentOutput').checked;p.reference_artifact_ids=[...S.references];p.loras=S.loras.map(x=>({...x}));p.ui_qwen21=true;
 return p;
}
function setQwenParams(p={}){
 if(p.retain_latents!==undefined)$('#qwenRetainLatents').checked=!!p.retain_latents;if(p.oom_cpu_decode_fallback!==undefined)$('#qwenOomCpuFallback').checked=!!p.oom_cpu_decode_fallback;
 for(const [key,id] of Object.entries(QWEN_FIELDS))if(p[key]!==undefined)$('#'+id).value=p[key];
 if(p.vae_decode_mode&&!p.qwen21_vae_decode_mode)$('#vaeDecodeMode').value=p.vae_decode_mode;
 $('#qwenEnableCfg').checked=!!(p.cfg_enabled??(Number(p.cfg)>1&&p.negative_prompt));$('#transparentOutput').checked=!!p.transparent_output;
 S.loras=Array.isArray(p.loras)?p.loras.map(x=>({...x})):[];const size=$('#width').value+'x'+$('#height').value;$('#ratio').value=[...$('#ratio').options].some(x=>x.value===size)?size:'custom';qwenControlState();
}
function qwenControlState(){
 $('#qwenCfgFields').hidden=!$('#qwenEnableCfg').checked;const mode=$('#qwenTransformerMode').value;$('#qwenTransformerPathWrap').hidden=!['gguf','safetensors'].includes(mode);$('#qwenNativeCompanions').hidden=mode!=='gguf';
 $('#qwenEffectiveSummary').textContent=tr('{count} input images · {width} × {height} · {steps} steps · CFG {cfg} · {format}',{count:S.references.length,width:$('#width').value,height:$('#height').value,steps:$('#steps').value,cfg:$('#qwenEnableCfg').checked?$('#cfg').value:tr('Off'),format:$('#transparentOutput').checked?tr('RGBA requested'):'PNG'});
 $('#qwenLoraList').innerHTML=S.loras.map((l,i)=>`<div class="qwen-file-row"><b>${esc(l.name||l.path.split('/').pop())}</b> · ${esc(l.strength)} <button class="ghost" data-qwen-lora-remove="${i}">×</button></div>`).join('');
}
function saveUiSession(){try{sessionStorage.setItem('qwen21_form_v1',JSON.stringify({params:qwenParams(),prompt:$('#prompt').value,mediaSettings:S.mediaSettings}));}catch(e){}}
function loadUiSession(){try{const d=JSON.parse(sessionStorage.getItem('qwen21_form_v1')||'null');if(d){setQwenParams(d.params);$('#prompt').value=d.prompt||'';S.mediaSettings=d.mediaSettings||{}}}catch(e){}}
function renderInputs(){
 $('#qwenInputCount').textContent=S.references.length+' / 10';
 $('#qwenClearInputs').disabled=S.references.length===0;
 $('#qwenReferenceList').innerHTML=S.references.map((id,i)=>{const a=artifactById(id);if(!a)return '';return `<div class="qwen-ref-row"><img src="${esc(a.url)}" alt="${esc(tr('Image {index}',{index:i+1}))}"><div><b>${esc(tr('Image {index}',{index:i+1}))}</b><small data-no-i18n>${esc(a.metadata?.original_name||id.slice(0,8))}</small></div><div class="qwen-ref-actions"><button class="ghost" data-ref-up="${i}" ${i===0?'disabled':''} title="Move up">↑</button><button class="ghost" data-ref-down="${i}" ${i===S.references.length-1?'disabled':''} title="Move down">↓</button><button class="ghost" data-ref-remove="${i}" title="Remove input">×</button></div></div>`}).join('')||`<span class="muted">${tr('No attachments: text-to-image.')}</span>`;
 qwenControlState();
}
function addQwenReference(id){if(S.references.includes(id))return;if(!artifactById(id)||artifactById(id).type!=='image')throw Error(tr('Choose an image in this project.'));if(S.references.length>=10)throw Error(tr('Maximum: 10 input images.'));S.references.push(id);renderInputs()}
async function uploadQwenImages(e){
 const files=[...e.target.files];if(!files.length)return;
 if(!S.project){setStatus('qwenInputStatus','Open a project first.',true);return}
 if(files.length+S.references.length>10){setStatus('qwenInputStatus','Maximum: 10 input images.',true);e.target.value='';return}
 const pid=S.project.id;e.target.disabled=true;$('#runImageBtn').disabled=true;
 try{for(let i=0;i<files.length;i++){if(!files[i].type.startsWith('image/'))throw Error(tr('Only image attachments are accepted.'));setStatus('qwenInputStatus',tr('Uploading image {index}/{total}…',{index:i+1,total:files.length}));const a=await uploadFile(files[i],'qwen21_reference');if(S.project?.id!==pid)throw Error(tr('Project changed; upload is stored in its original project.'));addQwenReference(a.id);}setStatus('qwenInputStatus','Images added in the displayed order.');}
 catch(err){setStatus('qwenInputStatus',err.message,true)}finally{e.target.value='';e.target.disabled=false;$('#runImageBtn').disabled=false;renderInputs()}
}
async function runQwen(){
 const b=$('#runImageBtn');if(b.disabled)return;
 try{
  if(!S.project)throw Error(tr('Open a project first.'));const prompt=$('#prompt').value.trim();if(!prompt)throw Error(tr('Enter a prompt or edit instruction.'));
  if(S.references.length>10)throw Error(tr('Maximum: 10 input images.'));
  const params=qwenParams();const tool=params.reference_artifact_ids.length===0?'generate_image':params.reference_artifact_ids.length===1?'reference':'multi_image';
  b.disabled=true;saveUiSession();generationFeedback('image','Adding to queue…');
  const d=await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project_id:S.project.id,tool,model_id:'qwen_image_21',prompt,params,model_setup:window.QwenStudio?.requestSetup()||{},prompt_enhancer:window.QwenPE?.provenance()||null})});
  projectRecoveryUntil=Date.now()+120000;startPolling();generationFeedback('image',tr('{count} jobs queued. Progress continues on the server.',{count:d.batch||1}));await openProjectNoPoll(S.project.id);
 }catch(e){generationFeedback('image',e.message,true);if(e.submissionUncertain){projectRecoveryUntil=Date.now()+120000;startPolling()}}
 finally{b.disabled=false;}
}
function reuseArtifact(id){const a=artifactById(id);if(!a)return;const m=a.metadata||{},p=m.params||{};setQwenParams(p);$('#prompt').value=p.user_prompt||m.prompt||'';S.references=[...(p.source_artifact_id?[p.source_artifact_id]:[]),...(p.reference_artifact_ids||[])].filter(artifactExists);S.references=[...new Set(S.references)].slice(0,10);const seed=actualSeed(a);if(seed!==null)$('#seed').value=seed;renderInputs();saveUiSession();showView('image')}
async function refreshPresets(){const d=await api('/api/image/presets/user');if(!S.boot)S.boot={};S.boot.user_image_presets=d.presets||[];renderPresets();}
function renderPresets(){const list=S.boot?.user_image_presets||[],sel=$('#quickPreset'),old=sel.value;sel.replaceChildren(new Option(tr('Choose a preset…'),''),...list.map(p=>new Option(p.name,p.id)));if(list.some(x=>x.id===old))sel.value=old;
 $('#qwenPresetList').innerHTML=list.map(p=>`<article class="qwen-preset-card"><b data-no-i18n>${esc(p.name)}</b><small data-no-i18n>${esc(p.description||'')}</small><small>${tr('{count} steps',{count:Number(p.params?.steps||40)})} · ${p.params?.width||1024} × ${p.params?.height||1024} · ${p.params?.transparent_output?'RGBA':'PNG'}</small><div class="inline-actions"><button data-preset-do="apply" data-preset-id="${esc(p.id)}" class="primary">${tr('Apply')}</button><button data-preset-do="rename" data-preset-id="${esc(p.id)}" class="ghost">${tr('Rename')}</button><button data-preset-do="replace" data-preset-id="${esc(p.id)}" class="ghost">${tr('Replace with current form')}</button><button data-preset-do="duplicate" data-preset-id="${esc(p.id)}" class="ghost">${tr('Duplicate')}</button><a data-safe-download href="/api/image/presets/user/${encodeURIComponent(p.id)}/download" class="ghost">${tr('Export JSON')}</a><button data-preset-do="delete" data-preset-id="${esc(p.id)}" class="ghost danger">${tr('Delete')}</button></div></article>`).join('')||tr('No saved presets yet.');
}
function presetFromForm(){const params=qwenParams();delete params.reference_artifact_ids;const include=$('#presetIncludePrompt').checked;if(!include)params.negative_prompt='';return {schema_version:3,edition:'qwen21',model_id:'qwen_image_21',name:$('#presetSaveName').value.trim(),description:$('#presetSaveDescription').value,params,include_prompt:include,prompt:include?$('#prompt').value:'',negative_prompt:include?$('#negativePrompt').value:''};}
function applyQwenPreset(p){const currentPrompt=$('#prompt').value,currentNegative=$('#negativePrompt').value;setQwenParams(p.params||{});if(p.include_prompt){$('#prompt').value=p.prompt||'';$('#negativePrompt').value=p.params?.negative_prompt||p.negative_prompt||''}else{$('#prompt').value=currentPrompt;$('#negativePrompt').value=currentNegative}qwenControlState();saveUiSession();showView('image');generationFeedback('image',tr('Preset applied: {name}',{name:p.name}));}
async function presetAction(action,id){const p=(S.boot?.user_image_presets||[]).find(x=>x.id===id);if(!p)return;
 if(action==='apply'){applyQwenPreset(p);return}
 if(action==='delete'){if(!confirm(tr('Delete this preset? Images and model files are not removed.')))return;await api('/api/image/presets/user/'+encodeURIComponent(id),{method:'DELETE'});}
 else if(action==='duplicate')await api('/api/image/presets/user/'+encodeURIComponent(id)+'/duplicate',{method:'POST'});
 else if(action==='rename'){const name=prompt(tr('Preset name'),p.name);if(!name?.trim())return;const description=prompt(tr('Description (optional)'),p.description||'');if(description===null)return;await api('/api/image/presets/user',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...p,name:name.trim(),description})});}
 else if(action==='replace'){if(!confirm(tr('Replace this preset with the current Create / Edit settings?')))return;await api('/api/image/presets/user',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...presetFromForm(),id:p.id,name:p.name,description:p.description})});}
 await refreshPresets();setStatus('presetStatus','Preset saved.');
}
async function bootstrap(){
 const view=rememberedCreatorView();S.boot=await api('/api/bootstrap');renderProjects();renderMediaTools();renderPresets();loadUiSession();renderInputs();
 const alive=(S.boot.projects||[]).filter(p=>!Number(p.trashed));let last='';try{last=localStorage.getItem('qwen21_project_id')||''}catch(_){}
 const chosen=alive.find(p=>p.id===last)||alive[0];if(chosen)await openProject(chosen.id);
 const onboarding=creatorInitialQuery.get('onboarding')==='1';showView(onboarding?'models':view);if(onboarding)consumeCreatorOnboarding();
 await refreshModelsHub();
}
function resumeGenerationStatus(){if(!document.hidden&&S.project&&!pendingProjectId){projectRecoveryUntil=Date.now()+120000;startPolling()}}
function bindQwenUi(){
 document.addEventListener('click',e=>{
  const nav=e.target.closest('[data-view],[data-goto]');if(nav){showView(nav.dataset.view||nav.dataset.goto);return}
  const ref=e.target.closest('[data-ref-up],[data-ref-down],[data-ref-remove]');if(ref){let i=Number(ref.dataset.refUp??ref.dataset.refDown??ref.dataset.refRemove);if(ref.hasAttribute('data-ref-remove'))S.references.splice(i,1);else{const j=i+(ref.hasAttribute('data-ref-up')?-1:1);if(j>=0&&j<S.references.length)[S.references[i],S.references[j]]=[S.references[j],S.references[i]];}renderInputs();return}
  const l=e.target.closest('[data-qwen-lora-remove]');if(l){S.loras.splice(Number(l.dataset.qwenLoraRemove),1);qwenControlState();saveUiSession();}
  const pb=e.target.closest('[data-preset-do]');if(pb)uiAction(pb,()=>presetAction(pb.dataset.presetDo,pb.dataset.presetId),'presetStatus');
  const topic=e.target.closest('[data-guide-topic]');if(topic){if(window.QwenGuide)window.QwenGuide.open(topic.dataset.guideTopic);else{showView('guide');const d=$('#guide-'+topic.dataset.guideTopic);if(d){d.open=true;d.scrollIntoView({block:'start'})}}}
 });
 $('#libraryProjectSelect').onchange=e=>selectLibraryProject(e.target.value);
 $('#qwenUpload').onchange=uploadQwenImages;$('#qwenClearInputs').onclick=()=>{S.references=[];renderInputs()};$('#runImageBtn').onclick=runQwen;
 $('#qwenEnableCfg').onchange=()=>{if($('#qwenEnableCfg').checked&&Number($('#cfg').value)<=1)$('#cfg').value='2';qwenControlState();saveUiSession()};
 $('#ratio').onchange=()=>{const m=$('#ratio').value.match(/^(\d+)x(\d+)$/);if(m){$('#width').value=m[1];$('#height').value=m[2]}else $('#qwenAdvanced').open=true;qwenControlState();saveUiSession()};
 ['width','height'].forEach(id=>$('#'+id).addEventListener('input',()=>$('#ratio').value='custom'));
 for(const el of $$('#view-image input,#view-image select,#view-image textarea'))el.addEventListener('change',()=>{qwenControlState();saveUiSession()});
 $('#qwenTransformerCatalog').onchange=e=>{const path=e.target.value;if(path){$('#qwenTransformerMode').value=path.toLowerCase().endsWith('.gguf')?'gguf':'safetensors';$('#qwenTransformerPath').value=path;qwenControlState();saveUiSession()}};
 $('#qwenLoraAdd').onclick=()=>{const sel=$('#qwenLoraSelect'),path=sel.value;if(!path)return;if(S.loras.length>=4){alert(tr('Maximum: 4 compatible LoRA adapters.'));return}if(!S.loras.some(x=>x.path===path))S.loras.push({path,name:sel.selectedOptions[0].textContent,strength:Number($('#qwenLoraWeight').value)});qwenControlState();saveUiSession()};
 $('#quickPresetApply').onclick=()=>{const p=(S.boot?.user_image_presets||[]).find(x=>x.id===$('#quickPreset').value);if(p)applyQwenPreset(p)};
 $('#quickPresetSave').onclick=()=>{showView('user-presets');$('#presetSaveName').focus()};$('#presetRefresh').onclick=e=>uiAction(e.currentTarget,refreshPresets,'presetStatus');
 $('#presetSaveNew').onclick=e=>uiAction(e.currentTarget,async()=>{const body=presetFromForm();if(!body.name)throw Error(tr('Enter a preset name.'));await api('/api/image/presets/user',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});await refreshPresets();setStatus('presetStatus','Preset saved.');},'presetStatus');
 $('#presetImport').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;if(file.size>2*1024**2)throw Error('Preset JSON must be below 2 MiB.');const body=JSON.parse(await file.text());await api('/api/image/presets/user/import',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});await refreshPresets();setStatus('presetStatus','Preset imported.')}catch(err){setStatus('presetStatus',err.message,true)}finally{e.target.value=''}};
 $('#newProjectBtn').onclick=()=>$('#projectModal').classList.remove('hidden');$('#cancelProject').onclick=()=>$('#projectModal').classList.add('hidden');$('#createProject').onclick=e=>uiAction(e.currentTarget,async()=>{const d=await api('/api/projects',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:$('#newProjectName').value,description:$('#newProjectDescription').value})});S.boot.projects.unshift(d.project);$('#projectModal').classList.add('hidden');await openProject(d.project.id);showView('image')},'imageRequestStatus');
 $('#closeDetails').onclick=()=>$('#detailsModal').classList.add('hidden');$('#refreshJobs').onclick=()=>S.project&&openProjectNoPoll(S.project.id);
 $('#cleanupJobs').onclick=e=>uiAction(e.currentTarget,async()=>{
  if(!S.project)return;
  if(!confirm(tr('Cancel queued and running jobs in this project only? The running job will stop at a safe cancellation point.')))return;
  const d=await api('/api/jobs/cleanup',{method:'POST',body:JSON.stringify({project_id:S.project.id})});
  setStatus('jobsActionStatus',d.message||tr('Cancellation requested.'));
  projectRecoveryUntil=Date.now()+120000;startPolling();await refreshProjectJobs();
 },'jobsActionStatus');
 $('#clearJobsHistory').onclick=e=>uiAction(e.currentTarget,async()=>{
  if(!S.project||!confirm(tr('Hide finished jobs in this project? Files, generated images and benchmark records will be kept.')))return;
  const d=await api(`/api/projects/${S.project.id}/jobs/clear-history`,{method:'POST'});
  setStatus('jobsActionStatus',`${d.hidden} ${tr('jobs hidden from the list')}`);await refreshProjectJobs();
 },'jobsActionStatus');
 $('#restoreJobsHistory').onclick=e=>uiAction(e.currentTarget,async()=>{
  if(!S.project)return;
  const d=await api(`/api/projects/${S.project.id}/jobs/restore-history`,{method:'POST'});
  setStatus('jobsActionStatus',`${d.restored} ${tr('jobs restored')}`);await refreshProjectJobs();
 },'jobsActionStatus');
 $('#jobsStatusFilter').onchange=e=>{jobsStatusFilter=e.target.value;bulkState.jobs.selected.clear();renderJobs();};
 configureBulkSelection();
 $('#unloadBtn').onclick=e=>uiAction(e.currentTarget,async()=>{await api('/api/runtime/image/unload',{method:'POST'});await refreshModelsHub()},'imageRequestStatus');$('#mobileUnloadBtn').onclick=()=>$('#unloadBtn').click();
 $('#mobileMoreToggle').onclick=()=>{if(window.QwenUI)window.QwenUI.toggleMore();else{const menu=$('#mobileMoreMenu');menu.classList.toggle('hidden');$('#mobileMoreToggle').setAttribute('aria-expanded',String(!menu.classList.contains('hidden')))}};$('#mobileMoreClose').onclick=()=>{if(window.QwenUI)window.QwenUI.closeMore();else{$('#mobileMoreMenu').classList.add('hidden');$('#mobileMoreToggle').setAttribute('aria-expanded','false')}};
 window.addEventListener('online',resumeGenerationStatus);document.addEventListener('visibilitychange',resumeGenerationStatus);window.addEventListener('pagehide',saveUiSession);
 window.addEventListener('languagechange',()=>{qwenControlState();renderPresets();renderInputs();renderAssets();renderJobs();renderProjects();if(typeof q22RenderLatent==='function')q22RenderLatent();if(qwenHub)renderQwenHub(qwenHub);});
}
bindQwenUi();
document.addEventListener('DOMContentLoaded',()=>bootstrap().catch(e=>{generationFeedback('image',e.message,true);setStatus('qwenModelStatus',e.message,true);console.error('[BOOT]',e)}));


// Use the same in-memory project artifacts as the history grid: no second request,
// no upload duplication and no image reconstruction. Updated on project changes and completion.
function renderCreatorPreview(arts){
 const frame=$('#mainPreview'),img=$('#mainPreviewImage'),empty=$('#mainPreviewEmpty'),caption=$('#imagePreviewCaption');
 if(!frame||!img||!empty||!caption)return;
 const latest=arts[0]||null;
 if(!latest||!latest.url){
   img.hidden=true;img.removeAttribute('src');img.dataset.artifactId='';empty.hidden=false;
   empty.textContent=tr('Your generated image will appear here.');caption.textContent='';return;
 }
 if(img.dataset.artifactId!==latest.id||img.getAttribute('src')!==latest.url){
   img.hidden=true;empty.hidden=false;empty.textContent='';
   img.dataset.artifactId=latest.id;
   // Set the image's source once. Polling does not reload an unchanged output.
   img.src=latest.url;
 }
 caption.textContent=artifactDisplayTitle(latest);
}
function initCreatorPreview(){
 const layout=$('#imageCreatorLayout'),toggle=$('#imageLayoutSwapBtn'),img=$('#mainPreviewImage'),empty=$('#mainPreviewEmpty');
 if(!layout||!toggle||!img||!empty)return;
 const key='qwen21.image.preview.side';
 let side='right';try{side=localStorage.getItem(key)==='left'?'left':'right'}catch(_){}
 const paint=()=>{layout.classList.toggle('image-preview-left',side==='left');
   toggle.setAttribute('aria-pressed',String(side==='left'));
   toggle.title=tr('Swap panels');toggle.setAttribute('aria-label',tr('Swap panels'));
   toggle.querySelector('span').textContent=tr('Swap panels');};
 toggle.addEventListener('click',()=>{side=side==='right'?'left':'right';
   try{localStorage.setItem(key,side)}catch(_){}paint();});
 img.addEventListener('load',()=>{if(!img.getAttribute('src'))return;img.hidden=false;empty.hidden=true;});
 img.addEventListener('error',()=>{img.hidden=true;empty.hidden=false;empty.textContent=tr('Cannot load preview. The result remains in history.');});
 img.addEventListener('click',()=>window.CreatorImageViewer?.show(img,img));
 img.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();window.CreatorImageViewer?.show(img,img)}});
 window.addEventListener('languagechange',()=>{paint();renderCreatorPreview((S.project?.artifacts||[]).filter(artifactIsImageResult));});
 paint();renderCreatorPreview((S.project?.artifacts||[]).filter(artifactIsImageResult));
}


if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',initCreatorPreview,{once:true});else initCreatorPreview();
