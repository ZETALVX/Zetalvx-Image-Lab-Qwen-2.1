/* Qwen 2.1 Benchmark Comparison Lab — 1.0.62.
   Reuses all saved benchmark reports and their generated artifacts. */
(()=>{'use strict';
const by=id=>document.getElementById(id), tr=s=>window.ZI18n?.t(s)||s;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const base='/api/model-hub/studio';
const apiReq=(path,method='GET',body)=>{const opt={method};if(body!==undefined){opt.headers={'Content-Type':'application/json'};opt.body=JSON.stringify(body)}return api(base+path,opt)};
const state={library:[],selected:new Set(),comparison:null,filters:{},picked:[],tab:'batch',viewerMode:'grid',duplicateCounts:new Map(),zoom:1,panX:0,panY:0,pointers:new Map(),gesture:null};
const memory=n=>typeof n==='number'?(n/1073741824).toFixed(2)+' GiB':'—';
const seconds=n=>typeof n==='number'?n.toFixed(2)+' s':'—';
const short=s=>String(s||'').replace(/\.safetensors$|\.gguf$/i,'').replace(/^qwen[-_]?image[-_]?2\.1[-_]?/i,'').replace(/^qwen_image_2\.1_/i,'').replace(/^qwen3vl_/i,'Qwen3VL ');
const unique=a=>[...new Set(a.filter(v=>v!==undefined&&v!==null&&String(v)!==''))].sort((a,b)=>String(a).localeCompare(String(b),undefined,{numeric:true}));
const cfgValue=x=>x?.settings?.cfg_enabled===false?'OFF':String(x?.settings?.cfg??'');
const negativeValue=x=>String(x?.settings?.negative_prompt||'').trim()?'on':'off';
function currentProject(){try{return S?.project?.id||''}catch(_){return ''}}
function inject(){
 const view=by('view-benchmark');if(!view||by('qcComparison'))return;
 const header=view.querySelector(':scope > .section-head');const workspace=document.createElement('div');workspace.id='qbBatchWorkspace';
 const children=[...view.children].filter(x=>x!==header);children.forEach(x=>workspace.appendChild(x));
 const tabs=document.createElement('div');tabs.className='qc-tabs segmented';tabs.innerHTML=`<button type="button" data-qc-tab="batch" class="active">${tr('Create / resume batches')}</button><button type="button" data-qc-tab="compare">${tr('Comparison gallery')}</button>`;
 const compare=document.createElement('section');compare.id='qcComparison';compare.hidden=true;compare.innerHTML=`
  <div class="qc-hero"><div><span class="eyebrow">QWEN 2.1</span><h3>${tr('Comparison gallery')}</h3><p>${tr('Browse every saved benchmark by example, compare model components side by side, and manage complete benchmark archives.')}</p></div><div class="qc-hero-actions"><button id="qcRefresh" class="ghost" type="button">${tr('Refresh')}</button><button id="qcLoad" class="primary" type="button">${tr('Load selected')}</button></div></div>
  <div class="qc-layout">
   <details class="qc-library-disclosure"><summary class="qc-library-toggle"><span>▸ ${tr('Benchmark archive')} <small id="qcArchiveSummary"></small></span><span>${tr('Open / close')}</span></summary><aside class="qc-library-panel">
    <div class="qc-library-head"><div><b>${tr('Benchmark archive')}</b><small id="qcLibraryCount"></small></div><div class="inline-actions"><button id="qcSelectVisible" class="text-button" type="button">${tr('Select visible')}</button><button id="qcClearSelected" class="text-button" type="button">${tr('Clear')}</button></div></div>
    <label>${tr('Search')}<input id="qcSearch" type="search" placeholder="${tr('Name, project, resolution')}"/></label>
    <label>${tr('Project')}<select id="qcProject"><option value="">${tr('All projects')}</option></select></label>
    <div id="qcLibrary" class="qc-library-list"></div>
   </aside></details>
   <main class="qc-gallery-panel">
    <div id="qcEmpty" class="qc-empty"><b>${tr('Choose one or more benchmarks')}</b><p>${tr('The gallery groups existing images by prompt/example. Nothing is copied or regenerated.')}</p></div>
    <div id="qcGallery" hidden>
     <div class="qc-filterbar">
      <label>${tr('Example')}<select id="qcPromptFilter"></select></label>
      <label>${tr('Resolution')}<select id="qcResolutionFilter"></select></label>
      <label>${tr('Transformer')}<select id="qcTransformerFilter"></select></label>
      <label>${tr('Encoder')}<select id="qcEncoderFilter"></select></label>
      <label>${tr('VAE')}<select id="qcVaeFilter"></select></label>
      <label>${tr('CFG')}<select id="qcCfgFilter"></select></label>
      <label>${tr('Negative prompt')}<select id="qcNegativeFilter"></select></label>
      <label>${tr('Steps')}<select id="qcStepsFilter"></select></label>
      <label>${tr('Seed')}<select id="qcSeedFilter"></select></label>
      <label>${tr('Status')}<select id="qcStatusFilter"></select></label>
      <label class="qc-checkline"><input id="qcImagesOnly" type="checkbox" checked/>${tr('Images only')}</label>
      <label class="qc-checkline"><input id="qcCollapseDuplicates" type="checkbox" checked/>${tr('Collapse identical configurations')}</label>
      <label class="qc-checkline"><input id="qcShowHidden" type="checkbox"/>${tr('Show hidden results')}</label>
     </div>
     <p class="qc-hint">${tr('Hiding keeps the PNG, job, metrics and latent. Duplicate grouping uses matching prompt, components, seed and settings, not image similarity. Nothing is deleted.')}</p>
     <div class="qc-summary" id="qcSummary"></div>
     <div id="qcGroups" class="qc-groups"></div>
    </div>
   </main>
  </div>
  <div id="qcTray" class="qc-tray" hidden><div><b id="qcTrayCount"></b><small>${tr('Select up to four results')}</small></div><div class="inline-actions"><button id="qcCompareGrid" class="primary" type="button">${tr('Compare side by side')}</button><button id="qcCompareAB" class="ghost" type="button">${tr('A/B slider')}</button><button id="qcTrayClear" class="text-button" type="button">${tr('Clear')}</button></div></div>
  <div id="qcViewer" class="qc-viewer" hidden role="dialog" aria-modal="true"><div class="qc-viewer-shell"><div class="qc-viewer-head"><div><b id="qcViewerTitle">${tr('Visual comparison')}</b><small id="qcViewerMeta"></small></div><div class="inline-actions"><label>${tr('Zoom')} <input id="qcZoom" type="range" min="1" max="5" step="0.1" value="1"/></label><button id="qcZoomOut" type="button" class="ghost qc-zoom-tool" aria-label="${tr('Zoom out')}">−</button><button id="qcZoomIn" type="button" class="ghost qc-zoom-tool" aria-label="${tr('Zoom in')}">+</button><button id="qcZoomReset" type="button" class="ghost qc-zoom-tool">${tr('Fit image')}</button><button id="qcViewerClose" class="ghost" type="button">${tr('Close')}</button></div></div><div id="qcViewerBody" class="qc-viewer-body"></div></div></div>`;
 view.append(tabs,workspace,compare);
 // Keep the shared comparison viewer visible even when the user is browsing
 // the normal Library: its benchmark tab is hidden outside Benchmark Lab.
 const viewer=by('qcViewer');if(viewer)document.body.appendChild(viewer);
 bind();restoreTab();loadLibrary().catch(showError);
}
function restoreTab(){try{setTab(sessionStorage.getItem('qwen21-comparison-tab')||'batch')}catch(_){setTab('batch')}}
function setTab(name){state.tab=name==='compare'?'compare':'batch';by('qbBatchWorkspace').hidden=state.tab!=='batch';by('qcComparison').hidden=state.tab!=='compare';document.querySelectorAll('[data-qc-tab]').forEach(b=>b.classList.toggle('active',b.dataset.qcTab===state.tab));try{sessionStorage.setItem('qwen21-comparison-tab',state.tab)}catch(_){}if(state.tab==='compare')loadLibrary().catch(showError)}
function showError(e){const box=by('qcEmpty');if(box){box.innerHTML=`<b>${tr('Comparison gallery error')}</b><p>${esc(e.message||e)}</p>`;box.hidden=false}console.error(e)}
async function loadLibrary(){const d=await apiReq('/benchmarks/library');state.library=d.benchmarks||[];renderProjectFilter();renderLibrary()}
function renderProjectFilter(){const el=by('qcProject');if(!el)return;const old=el.value;const projects=unique(state.library.map(x=>x.project_id)).map(id=>({id,name:state.library.find(x=>x.project_id===id)?.project_name||id}));el.replaceChildren(new Option(tr('All projects'),''),...projects.map(x=>new Option(x.name,x.id)));if([...el.options].some(x=>x.value===old))el.value=old;else el.value=''}
function visibleLibrary(){const q=(by('qcSearch')?.value||'').trim().toLowerCase(),pid=by('qcProject')?.value||'';return state.library.filter(x=>(!pid||x.project_id===pid)&&(!q||[x.name,x.project_name,(x.resolutions||[]).join(' ')].join(' ').toLowerCase().includes(q)))}
function statusLine(x){const c=x.status_counts||{};const bits=[];if(c.completed)bits.push(c.completed+' '+tr('completed'));if(c.failed)bits.push(c.failed+' '+tr('failed'));if(x.active_count)bits.push(x.active_count+' '+tr('active'));return bits.join(' · ')||tr('planned')}
function renderLibrary(){const list=visibleLibrary(),box=by('qcLibrary');if(!box)return;by('qcLibraryCount').textContent=list.length+' / '+state.library.length;const archiveCount=by('qcArchiveSummary');if(archiveCount)archiveCount.textContent=state.selected.size+' '+tr('selected')+' / '+state.library.length;box.innerHTML=list.map(x=>`<article class="qc-benchmark ${state.selected.has(x.id)?'selected':''}" data-qc-benchmark="${esc(x.id)}"><label class="qc-benchmark-main"><input type="checkbox" data-qc-select="${esc(x.id)}" ${state.selected.has(x.id)?'checked':''}/><span><b data-no-i18n>${esc(x.name)}</b><small data-no-i18n>${esc(x.project_name)} · ${new Date(Number(x.created_at||0)*1000).toLocaleString()}</small><span class="qc-chips"><i>${x.image_count} ${tr('images')}</i><i>${x.prompt_count} ${tr('examples')}</i>${(x.resolutions||[]).slice(0,3).map(r=>`<i>${esc(r)}</i>`).join('')}</span><small>${esc(statusLine(x))}</small></span></label><div class="qc-benchmark-actions"><button type="button" class="text-button" data-qc-open="${esc(x.id)}">${tr('Open report')}</button><button type="button" class="text-button" data-qc-rename="${esc(x.id)}">${tr('Rename')}</button><button type="button" class="text-button danger" data-qc-delete-report="${esc(x.id)}">${tr('Delete report')}</button><button type="button" class="text-button danger" data-qc-delete-all="${esc(x.id)}">${tr('Delete all')}</button></div></article>`).join('')||`<div class="qc-empty-small">${tr('No benchmarks match these filters.')}</div>`}
async function loadComparison(){const ids=[...state.selected];if(!ids.length)throw Error(tr('Select at least one benchmark.'));const d=await apiReq('/benchmarks/compare','POST',{ids});state.comparison=d.comparison;state.picked=[];setupFilters();renderGallery();updateTray()}
function selectOptions(el,values,label){const old=el.value;el.replaceChildren(new Option(tr(label),''),...values.map(v=>new Option(short(v),v)));if([...el.options].some(o=>o.value===old))el.value=old}
function setupFilters(){const items=state.comparison?.items||[];const p=by('qcPromptFilter');p.replaceChildren(new Option(tr('All examples'),''),...unique(items.map(x=>x.prompt_key)).map(k=>new Option(items.find(x=>x.prompt_key===k)?.prompt_title||k,k)));
 selectOptions(by('qcResolutionFilter'),unique(items.map(x=>`${x.settings?.width||''}×${x.settings?.height||''}`)),'All resolutions');selectOptions(by('qcTransformerFilter'),unique(items.map(x=>x.transformer)),'All transformers');selectOptions(by('qcEncoderFilter'),unique(items.map(x=>x.encoder)),'All encoders');selectOptions(by('qcVaeFilter'),unique(items.map(x=>x.vae)),'All VAEs');selectOptions(by('qcCfgFilter'),unique(items.map(cfgValue)),'All CFG values');const neg=by('qcNegativeFilter');neg.replaceChildren(new Option(tr('All negative prompt states'),''),new Option(tr('Negative OFF'),'off'),new Option(tr('Negative ON'),'on'));selectOptions(by('qcStepsFilter'),unique(items.map(x=>String(x.settings?.steps??''))),'All step counts');selectOptions(by('qcSeedFilter'),unique(items.map(x=>String(x.settings?.seed??''))),'All seeds');selectOptions(by('qcStatusFilter'),unique(items.map(x=>x.status)),'All statuses');by('qcStatusFilter').value=unique(items.map(x=>x.status)).includes('completed')?'completed':'';by('qcEmpty').hidden=true;by('qcGallery').hidden=false}
function filteredItems(){if(!state.comparison)return[];const val=id=>by(id)?.value||'';return state.comparison.items.filter(x=>{
 const resolution=`${x.settings?.width||''}×${x.settings?.height||''}`;if(val('qcPromptFilter')&&x.prompt_key!==val('qcPromptFilter'))return false;if(val('qcResolutionFilter')&&resolution!==val('qcResolutionFilter'))return false;if(val('qcTransformerFilter')&&x.transformer!==val('qcTransformerFilter'))return false;if(val('qcEncoderFilter')&&x.encoder!==val('qcEncoderFilter'))return false;if(val('qcVaeFilter')&&x.vae!==val('qcVaeFilter'))return false;if(val('qcCfgFilter')&&cfgValue(x)!==val('qcCfgFilter'))return false;if(val('qcNegativeFilter')&&negativeValue(x)!==val('qcNegativeFilter'))return false;if(val('qcStepsFilter')&&String(x.settings?.steps??'')!==val('qcStepsFilter'))return false;if(val('qcSeedFilter')&&String(x.settings?.seed??'')!==val('qcSeedFilter'))return false;if(val('qcStatusFilter')&&x.status!==val('qcStatusFilter'))return false;if(by('qcImagesOnly')?.checked&&!x.url)return false;if(x.hidden&&!by('qcShowHidden')?.checked)return false;return true})}
function cardTitle(x){
 // A short title alone is not an identity: cards always show the full component trio.
 if(x.transformer&&x.transformer!=='Baseline transformer')return x.transformer;
 if(x.encoder&&x.encoder!=='Baseline encoder')return x.encoder;
 if(x.vae&&x.vae!=='Baseline VAE')return x.vae;
 return tr('Baseline');
}
function execution(x){const r=x.result||{};return String(r.execution_backend||r.pipeline||r.execution_mode||x.precision_policy||'Image Lab worker')}
function itemDetails(x,viewer=false){
 const s=x.settings||{},r=x.result||{},c=x.configuration||{},gpu=x.metrics?.peak_gpu_process_bytes??x.metrics?.peak_cuda_reserved_bytes;
 const negative=String(s.negative_prompt||'').trim();
 const sampler=[r.sampler_name,r.scheduler_name].filter(Boolean).join(' / ');
 const rows=[
  ['Transformer',x.transformer],['Text Encoder',x.encoder],['VAE',x.vae],['LoRA',x.lora_summary||'No LoRA'],
  ...(c.baseline_repository&&c.baseline_repository!=='—'?[[tr('Base model'),c.baseline_repository]]:[]),
  [tr('Backend'),execution(x)],
  ...(c.dtype?[[tr('Dtype'),c.dtype]]:[]),
  ...(c.memory_mode?[[tr('Memory mode'),c.memory_mode]]:[]),
  [tr('Precision policy'),x.precision_policy||'—'],
  ...(sampler?[[tr('Sampler / Scheduler'),sampler]]:[]),
  [tr('Resolution'),`${s.width??'—'}×${s.height??'—'}`],
  [tr('Steps'),s.steps??'—'],['CFG',cfgValue(x)||'—'],
  [tr('Negative prompt'),negative?tr('ON'):tr('OFF')],
  ['Seed',s.seed??'—'],
  [tr('Time'),seconds(x.duration_seconds)],['VRAM',memory(gpu)],[tr('Peak RAM (process)'),memory(x.metrics?.peak_rss_bytes)],
  [tr('Job ID'),x.job_id||'—']
 ];
 const content=rows.map(([name,value])=>`<dt>${esc(name)}</dt><dd data-no-i18n title="${esc(value)}">${esc(value)}</dd>`).join('');
 return `<dl class="qc-card-details${viewer?' qc-viewer-details':''}">${content}</dl>`+
  (negative?`<details class="qc-negative-details"><summary>${tr('Negative prompt')} · ${tr('Show text')}</summary><p data-no-i18n>${esc(negative)}</p></details>`:'');
}
function resultCard(x){
 const s=x.settings||{},negative=negativeValue(x)==='on',cfg=cfgValue(x);
 const picked=state.picked.includes(x.job_id);
 const repeatCount=state.duplicateCounts.get(x.job_id)||1;
 return `<article class="qc-result ${picked?'picked':''}" data-qc-job="${esc(x.job_id)}"><div class="qc-result-head"><div><b data-no-i18n title="${esc(cardTitle(x))}">${esc(cardTitle(x))}</b><small data-no-i18n>${esc(x.benchmark_name)} · ${esc(x.project_name)}</small><small class="qc-result-example" data-no-i18n>${esc(x.prompt_title||'')}</small></div><div class="qc-card-actions"><button type="button" class="qc-hide" data-qc-hide="${esc(x.job_id)}" title="${tr(x.hidden?'Show result':'Hide result')}" aria-label="${tr(x.hidden?'Show result':'Hide result')}">${x.hidden?'◉':'⊘'}</button><button type="button" class="qc-pick ${picked?'active':''}" data-qc-pick="${esc(x.job_id)}" aria-label="${tr('Select for comparison')}">${picked?'✓':'+'}</button></div></div>
 ${x.hidden?`<span class="qc-hidden-badge">${tr('Hidden result')}</span>`:''}${repeatCount>1?`<span class="qc-duplicate-badge">${repeatCount} ${tr('same-settings results')}</span>`:''}
 ${x.url?`<button type="button" class="qc-image-button" data-qc-view="${esc(x.job_id)}" aria-label="${tr('Open image')} · ${esc(cardTitle(x))}"><img src="${esc(x.url)}" alt="${esc(x.label)}" loading="lazy"/></button>`:`<div class="qc-no-image"><span>${esc(tr(x.status||'No image'))}</span>${x.compatibility_reason?`<small>${esc(x.compatibility_reason)}</small>`:''}</div>`}
 <div class="qc-chips"><i>${esc(`${s.width??'—'}×${s.height??'—'}`)}</i><i>${esc((s.steps??'—')+' '+tr('steps'))}</i><i>CFG ${esc(cfg||'—')}</i><i class="${negative?'on':''}">${negative?tr('Negative ON'):tr('Negative OFF')}</i></div>
 ${itemDetails(x)}${x.error_present?`<small class="qc-error-tag">${tr('Error recorded')}</small>`:''}</article>`;
}
function renderGallery(){
 const matching=filteredItems(),items=[],counts=new Map(),firstBySignature=new Map();
 const collapse=Boolean(by('qcCollapseDuplicates')?.checked);
 for(const x of matching){
  const key=x.fingerprint||x.job_id;
  if(!counts.has(key))counts.set(key,0);
  counts.set(key,counts.get(key)+1);
  if(collapse&&firstBySignature.has(key))continue;
  firstBySignature.set(key,x.job_id);items.push(x);
 }
 state.duplicateCounts=new Map();
 for(const x of items)state.duplicateCounts.set(x.job_id,counts.get(x.fingerprint||x.job_id)||1);
 const groups=new Map();for(const x of items){if(!groups.has(x.prompt_key))groups.set(x.prompt_key,[]);groups.get(x.prompt_key).push(x)}
 const collapsed=matching.length-items.length;
 by('qcSummary').innerHTML=`<b>${items.length} ${tr('results')}</b><span>${groups.size} ${tr('examples')}</span><span>${state.comparison?.benchmarks?.length||0} ${tr('benchmarks')}</span>${collapsed?`<span>${collapsed} ${tr('matching settings collapsed')}</span>`:''}`;
 by('qcGroups').innerHTML=[...groups.values()].map(group=>{const first=group[0],cards=group.map(x=>resultCard(x)).join('');return `<section class="qc-group"><div class="qc-group-head"><div><span class="eyebrow">${tr('Example')}</span><h4 data-no-i18n>${esc(first.prompt_title)}</h4><details><summary>${tr('Prompt')}</summary><p data-no-i18n>${esc(first.prompt)}</p></details></div><span>${group.length} ${tr('results')}</span></div><div class="qc-result-grid">${cards}</div></section>`}).join('')||`<div class="qc-empty"><b>${tr('No results match these filters.')}</b></div>`;
}
async function toggleHidden(jobId){
 const x=state.comparison?.items?.find(x=>x.job_id===jobId);if(!x)return;
 const desired=!Boolean(x.hidden);
 await apiReq('/benchmarks/'+x.benchmark_id+'/visibility','PATCH',{job_id:jobId,hidden:desired});
 x.hidden=desired;
 // Keep a selected result in the comparison tray, even when hidden from the gallery.
 renderGallery();updateTray();
}
function togglePick(job){const i=state.picked.indexOf(job);if(i>=0)state.picked.splice(i,1);else{if(state.picked.length>=4)state.picked.shift();state.picked.push(job)}renderGallery();updateTray()}
function pickedItems(){const all=state.comparison?.items||[];return state.picked.map(id=>all.find(x=>x.job_id===id)).filter(Boolean)}
function updateTray(){const n=state.picked.length;by('qcTray').hidden=!n;by('qcTrayCount').textContent=n+' '+tr('selected');by('qcCompareGrid').disabled=n<2;by('qcCompareAB').disabled=n!==2}
function openSingle(jobId){const x=state.comparison?.items?.find(x=>x.job_id===jobId);if(!x?.url)return;state.viewerMode='grid';renderViewer([x])}
function viewerLabel(x){const s=x.settings||{};return [cardTitle(x),`${s.width||'—'}×${s.height||'—'}`,`CFG ${cfgValue(x)||'—'}`,negativeValue(x)==='on'?tr('Negative ON'):tr('Negative OFF')].join(' · ')}
// Viewer camera: default is fit-to-container, not cropped or upscaled.
// When zooming, pan is bounded by the true rendered image dimensions rather
// than the size of its scroll container, so all four edges remain reachable.
function paneZoomLimits(pane){
 if(!pane)return {x:0,y:0};
 const sw=pane.clientWidth,sh=pane.clientHeight;
 if(!sw||!sh)return {x:0,y:0};
 let x=0,y=0;
 // A/B may compare different aspect ratios: allow the full extent of both.
 pane.querySelectorAll('img').forEach(img=>{
  const nw=img.naturalWidth||sw,nh=img.naturalHeight||sh;
  if(!nw||!nh)return;
  const fit=Math.min(state.viewerMode==='ab'?Infinity:1,sw/nw,sh/nh);
  x=Math.max(x,(nw*fit*state.zoom-sw)/2);
  y=Math.max(y,(nh*fit*state.zoom-sh)/2);
 });
 return {x:Math.max(0,x),y:Math.max(0,y)};
}
function applyZoom(){
 const z=Math.max(1,Math.min(5,Number(by('qcZoom')?.value||state.zoom||1)));
 state.zoom=z;
 const panes=[...document.querySelectorAll('#qcViewerBody .qc-zoom-pane, #qcViewerBody .qc-ab-stage')];
 const bounds=panes.map(pane=>({pane,lim:paneZoomLimits(pane)}));
 const maxX=Math.max(0,...bounds.map(x=>x.lim.x)),maxY=Math.max(0,...bounds.map(x=>x.lim.y));
 state.panX=Math.min(maxX,Math.max(-maxX,state.panX));
 state.panY=Math.min(maxY,Math.max(-maxY,state.panY));
 bounds.forEach(({pane,lim})=>{
  const x=Math.min(lim.x,Math.max(-lim.x,state.panX)),y=Math.min(lim.y,Math.max(-lim.y,state.panY));
  const transform=`translate3d(${x}px,${y}px,0) scale(${z})`;
  pane.querySelectorAll('img').forEach(img=>img.style.transform=transform);
  pane.classList.toggle('qc-can-pan',z>1);
 });
 if(by('qcZoom'))by('qcZoom').value=String(z);
}
function setZoom(value,resetPan=false){
 if(resetPan){state.panX=0;state.panY=0}
 state.zoom=Math.max(1,Math.min(5,Math.round(Number(value)*10)/10));
 if(by('qcZoom'))by('qcZoom').value=String(state.zoom);
 applyZoom();
}
function centerOfPointers(){const list=[...state.pointers.values()];return {x:list.reduce((s,p)=>s+p.x,0)/list.length,y:list.reduce((s,p)=>s+p.y,0)/list.length}}
function pointerDistance(){const a=[...state.pointers.values()];return a.length<2?0:Math.hypot(a[0].x-a[1].x,a[0].y-a[1].y)}
function installPan(){
 const panes=[...document.querySelectorAll('#qcViewerBody .qc-zoom-pane, #qcViewerBody .qc-ab-stage')];
 state.pointers.clear();state.gesture=null;
 panes.forEach(pane=>{
  pane.addEventListener('pointerdown',e=>{
   if(e.target.closest('input,button,.qc-ab-label'))return;
   if(state.zoom<=1&&e.pointerType==='mouse')return;
   e.preventDefault();pane.setPointerCapture(e.pointerId);
   state.pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});
   state.gesture={center:centerOfPointers(),distance:pointerDistance(),zoom:state.zoom};
  });
  pane.addEventListener('pointermove',e=>{
   if(!state.pointers.has(e.pointerId))return;
   state.pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});
   const c=centerOfPointers(),g=state.gesture;
   if(!g)return;
   // At fit (1x), a one-finger vertical gesture scrolls the viewer details;
   // at magnification it pans the image instead.
   if(state.zoom===1&&state.pointers.size===1&&e.pointerType==='touch'){
    by('qcViewer').scrollTop-=c.y-g.center.y;
    state.gesture={center:c,distance:0,zoom:1};
    return;
   }
   if(state.pointers.size>=2&&g.distance){
    const z=Math.max(1,Math.min(5,g.zoom*pointerDistance()/g.distance));
    state.zoom=z;
    if(by('qcZoom'))by('qcZoom').value=String(z);
   }
   state.panX+=c.x-g.center.x;state.panY+=c.y-g.center.y;
   state.gesture={center:c,distance:pointerDistance(),zoom:state.zoom};
   applyZoom();
  });
  const stop=e=>{if(!state.pointers.has(e.pointerId))return;state.pointers.delete(e.pointerId);if(pane.hasPointerCapture?.(e.pointerId))pane.releasePointerCapture(e.pointerId);state.gesture=state.pointers.size?{center:centerOfPointers(),distance:pointerDistance(),zoom:state.zoom}:null};
  pane.addEventListener('pointerup',stop);pane.addEventListener('pointercancel',stop);
  pane.addEventListener('dblclick',e=>{if(e.target.closest('input'))return;setZoom(state.zoom>1?1:2,true)});
  pane.addEventListener('wheel',e=>{if(!e.ctrlKey&&!e.metaKey)return;e.preventDefault();setZoom(state.zoom+(e.deltaY<0?.2:-.2))},{passive:false});
 });
}
function renderViewer(items,mode=state.viewerMode){
 state.viewerMode=mode;by('qcViewer').hidden=false;document.body.classList.add('qc-modal-open');
 by('qcViewerMeta').textContent=items.map(viewerLabel).join('  |  ');
 const body=by('qcViewerBody');
 if(mode==='ab'&&items.length===2){
  body.className='qc-viewer-body qc-ab-wrap';
  body.innerHTML=`<div class="qc-ab-stage"><img src="${esc(items[0].url)}" alt="A"/><div class="qc-ab-top" id="qcAbTop"><img src="${esc(items[1].url)}" alt="B"/></div><span class="qc-ab-label a">A · ${esc(viewerLabel(items[0]))}</span><span class="qc-ab-label b">B · ${esc(viewerLabel(items[1]))}</span><input id="qcAbRange" type="range" min="0" max="100" value="50" aria-label="A/B"/></div><div class="qc-ab-captions">${items.map((x,i)=>`<section><b>${i===0?'A':'B'} · ${esc(cardTitle(x))}</b>${itemDetails(x,true)}</section>`).join('')}</div>`;
  by('qcAbRange').oninput=e=>by('qcAbTop').style.clipPath=`inset(0 ${100-Number(e.target.value)}% 0 0)`;
 }else{
  body.className='qc-viewer-body qc-viewer-grid';
  body.innerHTML=items.map(x=>`<figure data-qc-job="${esc(x.job_id)}"><div class="qc-zoom-pane"><img src="${esc(x.url)}" alt="${esc(x.label)}"/></div><figcaption><b data-no-i18n>${esc(cardTitle(x))}</b><small data-no-i18n>${esc(x.label)}</small>${itemDetails(x,true)}</figcaption></figure>`).join('');
 }
 state.zoom=1;state.panX=0;state.panY=0;
 setZoom(1,true);
 document.querySelectorAll('#qcViewerBody img').forEach(img=>img.addEventListener('load',applyZoom,{once:true}));
 installPan();
}
function closeViewer(){by('qcViewer').hidden=true;document.body.classList.remove('qc-modal-open');state.pointers.clear();state.gesture=null}
async function renameBenchmark(id){const item=state.library.find(x=>x.id===id);if(!item)return;const name=prompt(tr('Benchmark name'),item.name);if(!name?.trim())return;await apiReq('/benchmarks/'+id,'PATCH',{name:name.trim()});await loadLibrary();const option=[...(by('qbHistory')?.options||[])].find(x=>x.value===id);if(option)option.textContent=name.trim();if(by('qbResultsTitle')&&window.QwenStudio?.state?.report?.id===id)by('qbResultsTitle').textContent=name.trim()}
async function deleteBenchmark(id,all){const item=state.library.find(x=>x.id===id);if(!item)return;const message=all?tr('Delete this benchmark, all its generated images, jobs and saved latents? This cannot be undone.'):tr('Delete only the benchmark report? Generated images and jobs will be kept.');if(!confirm(message+'\n\n'+item.name))return;await apiReq('/benchmarks/'+id+'/delete','POST',{delete_contents:all});state.selected.delete(id);if(state.comparison?.benchmarks?.some(x=>x.id===id))state.comparison=null;await loadLibrary();by('qcGallery').hidden=true;by('qcEmpty').hidden=false;by('qcEmpty').innerHTML=`<b>${tr('Benchmark deleted')}</b><p>${all?tr('Images, jobs and linked latents were removed.'):tr('Images and jobs were kept.')}</p>`;window.dispatchEvent(new CustomEvent('qwen-benchmark-library-updated'))}
function openReport(id){const select=by('qbHistory');if(select){if([...select.options].some(x=>x.value===id)){select.value=id;by('qbOpen')?.click();setTab('batch')}else{const item=state.library.find(x=>x.id===id);if(item?.project_id!==currentProject())alert(tr('Open the benchmark project first to use the detailed report view.'))}}}
function bind(){document.querySelectorAll('[data-qc-tab]').forEach(b=>b.onclick=()=>setTab(b.dataset.qcTab));by('qcRefresh').onclick=()=>loadLibrary().catch(showError);by('qcLoad').onclick=()=>loadComparison().catch(showError);by('qcSearch').oninput=renderLibrary;by('qcProject').onchange=renderLibrary;by('qcSelectVisible').onclick=()=>{visibleLibrary().slice(0,100).forEach(x=>state.selected.add(x.id));renderLibrary()};by('qcClearSelected').onclick=()=>{state.selected.clear();renderLibrary()};
 by('qcLibrary').onclick=e=>{const t=e.target.closest('button');if(!t)return;const id=t.dataset.qcOpen||t.dataset.qcRename||t.dataset.qcDeleteReport||t.dataset.qcDeleteAll;if(t.dataset.qcOpen)openReport(id);if(t.dataset.qcRename)renameBenchmark(id).catch(showError);if(t.dataset.qcDeleteReport)deleteBenchmark(id,false).catch(showError);if(t.dataset.qcDeleteAll)deleteBenchmark(id,true).catch(showError)};by('qcLibrary').onchange=e=>{const c=e.target.closest('[data-qc-select]');if(!c)return;c.checked?state.selected.add(c.dataset.qcSelect):state.selected.delete(c.dataset.qcSelect);renderLibrary()};
 for(const id of ['qcPromptFilter','qcResolutionFilter','qcTransformerFilter','qcEncoderFilter','qcVaeFilter','qcCfgFilter','qcNegativeFilter','qcStepsFilter','qcSeedFilter','qcStatusFilter','qcImagesOnly','qcCollapseDuplicates','qcShowHidden'])by(id).onchange=renderGallery;by('qcGroups').onclick=e=>{const h=e.target.closest('[data-qc-hide]');if(h){h.disabled=true;toggleHidden(h.dataset.qcHide).catch(showError).finally(()=>{h.disabled=false});return}const p=e.target.closest('[data-qc-pick]');if(p){togglePick(p.dataset.qcPick);return}const v=e.target.closest('[data-qc-view]');if(v)openSingle(v.dataset.qcView)};by('qcTrayClear').onclick=()=>{state.picked=[];renderGallery();updateTray()};by('qcCompareGrid').onclick=()=>renderViewer(pickedItems(),'grid');by('qcCompareAB').onclick=()=>renderViewer(pickedItems(),'ab');by('qcViewerClose').onclick=closeViewer;by('qcViewer').onclick=e=>{if(e.target===by('qcViewer'))closeViewer()};by('qcZoom').oninput=e=>setZoom(Number(e.target.value));by('qcZoomIn').onclick=()=>setZoom(state.zoom+.2);by('qcZoomOut').onclick=()=>setZoom(state.zoom-.2);by('qcZoomReset').onclick=()=>setZoom(1,true);window.addEventListener('resize',()=>{if(!by('qcViewer')?.hidden)applyZoom()});document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!by('qcViewer').hidden)closeViewer()});window.addEventListener('studio-project-changed',()=>{if(state.tab==='compare')loadLibrary().catch(showError)});window.addEventListener('qwen-benchmark-library-updated',()=>{if(state.tab==='compare')loadLibrary().catch(showError)})}
function compareLibrary(assets,mode='grid'){
 if(!by('qcViewer'))inject();
 if(!Array.isArray(assets)||assets.length<2||assets.length>4)throw Error('Select 2–4 images.');
 const items=assets.map(a=>{
  if(a?.type!=='image'||!a.url)throw Error('Choose valid image assets.');
  const meta=a.metadata||{},p=meta.params||{},result=meta.runtime_result||{};
  const cfg=p._model_config_snapshot||meta.model_config||{};
  const adapters=p.loras||meta.loras||[];
  return {job_id:a.id,artifact_id:a.id,label:meta.original_name||a.id.slice(0,12),url:a.url,
   prompt:p.user_prompt||meta.prompt||'',prompt_title:meta.original_name||'Library image',
   transformer:cfg.model_path?.split(/[\\/]/).pop()||'Baseline transformer',
   encoder:cfg.text_encoder_path?.split(/[\\/]/).pop()||'Baseline encoder',
   vae:cfg.vae_path?.split(/[\\/]/).pop()||'Baseline VAE',
   lora_summary:adapters.map(l=>(l.name||l.path||'LoRA').split(/[\\/]/).pop()+' @'+(l.strength??1)).join(' + ')||'No LoRA',
   settings:{width:p.width||meta.width,height:p.height||meta.height,steps:p.steps,cfg:p.cfg,cfg_enabled:p.cfg_enabled,negative_prompt:p.negative_prompt,seed:result.seed??p.seed},
   result,configuration:{baseline_repository:cfg.model_dir?.split(/[\\/]/).pop()||'',dtype:cfg.dtype||'',memory_mode:cfg.memory_mode||''},metrics:result.metrics||{},duration_seconds:result.duration_seconds,
   precision_policy:cfg.quantized_loading||'',benchmark_name:'Library',project_name:'',status:'completed'};
 });
 // Identical overlay, zoom/pan controls and A/B slider used by the Benchmark Lab.
 renderViewer(items,mode==='ab'&&items.length===2?'ab':'grid');
}
window.QwenComparison={compareLibrary};
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',inject,{once:true});else inject();
})();
