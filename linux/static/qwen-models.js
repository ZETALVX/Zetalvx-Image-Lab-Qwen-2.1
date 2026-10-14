/* Modified for Zetalvx Image Lab Qwen 1.0.27: catalog-first model configuration; Apache-2.0, retained NOTICE. */
/* Qwen-only models, authenticated downloads and per-user account tokens. */
let qwenHub=null,qwenHubLoading=null,qwenModelDirty=false,qwenDownloadPreview=null,qwenActiveUpload=null,qwenUploadRunning=false,qwenUploadCancelRequested=false;
const QWEN_MODEL_FIELDS={model_dir:'qwenModelDir',model_path:'qwenModelPath',device:'qwenModelDevice',dtype:'qwenModelDtype',memory_mode:'qwenModelMemory',gpu_memory:'qwenGpuMemory',cpu_memory:'qwenCpuMemory',default_steps:'qwenDefaultSteps',default_cfg:'qwenDefaultCfg',vae_path:'qwenVaePath',vae_config_path:'qwenVaeConfig',text_encoder_dir:'qwenTextEncoderDir',text_encoder_path:'qwenTextEncoderPath',processor_dir:'qwenProcessorDir',scheduler_dir:'qwenSchedulerDir',text_encoder_dtype:'qwenTextEncoderDtype',scheduler_shift:'qwenSchedulerShift',decode_device:'qwenModelDecodeDevice',vae_dtype:'qwenModelDecodeDtype',vae_decode_mode:'qwenModelDecodeMode',tile_height:'qwenModelTileHeight',tile_width:'qwenModelTileWidth',tile_stride_height:'qwenModelTileStrideHeight',tile_stride_width:'qwenModelTileStrideWidth'};
async function hubApi(path,options={}){if(options.method&&options.method!=='GET'){options={...options,headers:{'Content-Type':'application/json',...(options.headers||{})},body:options.body??'{}'}}return api('/api/model-hub/'+path,options)}
function qwenModelTab(name){$$('[data-hub-panel]').forEach(x=>x.hidden=x.dataset.hubPanel!==name);$$('[data-hub-tab]').forEach(x=>{const on=x.dataset.hubTab===name;x.classList.toggle('active',on);x.setAttribute('aria-pressed',String(on))});if(name==='downloads')window.ZetalvxDownloadManager?.refresh(true);}
function renderQwenValidation(v){
 const box=$('#qwenValidation'),fields=v?.fields||[],present=fields.filter(x=>x.exists).length;
 const prior=box.querySelector('details');
 const open=prior?prior.open:!!(v&&!v.ready);
 box.className='validation-box '+(v?.ready?'ready':'bad');
 box.innerHTML=`<div class="qwen-validation-summary"><b>${tr(v?.ready?'READY':'MISSING FILES')}</b><span>${present} / ${fields.length} ${tr('checks passed')}</span></div>
 <details class="qwen-validation-details" ${open?'open':''}><summary>${tr('Files and compatibility details')}</summary>
 ${fields.map(x=>`<div class="path-row"><div data-no-i18n><b>${esc(x.field)}</b><code>${esc(x.value||'—')}</code>${x.detail?`<small>${esc(tr(x.detail))}</small>`:''}</div><span>${x.exists?'✓':'×'} ${tr(x.exists?'Present':'Missing')}</span></div>`).join('')}
 </details>`;
}
function renderQwenAccounts(data){for(const [provider,id] of [['huggingface','hubHfState'],['civitai','hubCivitaiState']]){const a=data?.[provider];$('#'+id).textContent=a?.stored?(a.account||tr('Token saved')):tr('Not connected')}}
function qwenFormatState(){const wrap=$('#qwenModelPathWrap');if(wrap)wrap.hidden=true;$('#qwenRepoWithoutTransformer').disabled=$('#hubKind').value!=='repository';}
const QWEN_COMPONENT_PICKERS={
 repository:{picker:'qwenRepositoryPicker',target:'qwenModelDir',preview:'qwenRepositoryPathPreview',section:'qwenBaseModelSection'},
 transformer:{picker:'qwenTransformerPicker',target:'qwenModelPath',preview:'qwenTransformerPathPreview',section:'qwenTransformerSection'},
 vae:{picker:'qwenVaePicker',target:'qwenVaePath',preview:'qwenVaePathPreview',section:'qwenVaeSection'},
 text_encoder:{picker:'qwenTextEncoderPicker',target:'qwenTextEncoderDir',preview:'qwenTextEncoderPathPreview',section:'qwenTextEncoderSection'},
 processor:{picker:'qwenProcessorPicker',target:'qwenProcessorDir',preview:'qwenProcessorPathPreview',section:'qwenProcessorSection'},
 scheduler:{picker:'qwenSchedulerPicker',target:'qwenSchedulerDir',preview:'qwenSchedulerPathPreview',section:'qwenSchedulerSection'}
};
function qwenOriginalLabel(kind){return tr(({repository:'Use server path below',transformer:'Original transformer from base repository',vae:'Official VAE from base repository',text_encoder:'Text encoder from base repository',processor:'Processor from base repository',scheduler:'Scheduler from base repository'})[kind]||'Original component');}
function qwenOriginalPath(kind){const base=$('#qwenModelDir')?.value?.trim()||'';if(kind==='repository')return base||tr('No repository selected');const part={transformer:'transformer',vae:'vae',text_encoder:'text_encoder',processor:'processor',scheduler:'scheduler'}[kind];return base?base.replace(/\/$/,'')+'/'+part:tr('Base repository component');}
function qwenComponentPath(kind){if(kind==='text_encoder'&&$('#qwenTextEncoderPath')?.value)return $('#qwenTextEncoderPath').value.trim();const spec=QWEN_COMPONENT_PICKERS[kind];return spec?($('#'+spec.target)?.value||'').trim():'';}
function qwenSetComponentPath(kind,path){const spec=QWEN_COMPONENT_PICKERS[kind];if(!spec)return;const target=$('#'+spec.target);if(target)target.value=path||'';if(kind==='text_encoder'){const file=/\.safetensors$/i.test(path||'');$('#qwenTextEncoderPath').value=file?path:'';target.value=file?'':(path||'');}if(kind==='transformer')$('#qwenModelFormat').value=!path?'original':path.toLowerCase().endsWith('.gguf')?'gguf':'safetensors';if(kind==='vae'&&path&& !/\.safetensors$/i.test(path))$('#qwenVaeConfig').value='';qwenModelDirty=true;qwenFormatState();qwenRenderComponentPickers();qwenRenderProfileComponentSummary();}
function qwenCatalogKindMatches(entry,kind){return entry.kind===kind||(kind==='text_encoder'&&entry.kind==='text_encoder_file');}
function qwenRenderComponentPickers(){
 const items=qwenHub?.local_catalog||[];
 for(const [kind,spec] of Object.entries(QWEN_COMPONENT_PICKERS)){
  const sel=$('#'+spec.picker);if(!sel)continue;const path=qwenComponentPath(kind);const options=[new Option(qwenOriginalLabel(kind),'')];
  for(const e of items.filter(x=>qwenCatalogKindMatches(x,kind)&&x.present&&x.selectable!==false))options.push(new Option((e.name||e.path.split('/').pop())+(kind==='text_encoder'?' · '+tr(e.kind==='text_encoder_file'?'Encoder weights':'Text encoder'):''),e.id));
  const signature=JSON.stringify(options.map(o=>[o.value,o.text]));
  if(sel.dataset.qwenChoices!==signature){sel.replaceChildren(...options);sel.dataset.qwenChoices=signature;}
  const match=items.find(x=>qwenCatalogKindMatches(x,kind)&&x.present&&x.selectable!==false&&String(x.path)===path);sel.value=match?.id||'';
  const preview=$('#'+spec.preview);if(preview)preview.textContent=path||qwenOriginalPath(kind);
 }
 const vae=qwenComponentPath('vae'),details=$('#qwenVaeConfigDetails');if(details)details.open=!!(vae&&/\.safetensors$/i.test(vae));
}
function qwenRenderProfileComponentSummary(){
 const box=$('#qwenProfileComponentSummary');if(!box)return;const row=(label,value)=>`<div><span>${esc(tr(label))}</span><b data-no-i18n>${esc(value)}</b></div>`;const short=(path,fallback)=>path?(path.split('/').filter(Boolean).pop()||path):tr(fallback);
 box.innerHTML=row('Base model',short($('#qwenModelDir').value,'Not selected'))+row('Transformer',short(qwenComponentPath('transformer'),'Original'))+row('VAE / Decoder',short(qwenComponentPath('vae'),'Official'))+row('Text encoder',short(qwenComponentPath('text_encoder'),'Base repository'))+row('Processor',short(qwenComponentPath('processor'),'Base repository'))+row('Scheduler',short(qwenComponentPath('scheduler'),'Base repository'))+row('LoRA stack',tr('{count} adapters',{count:(S.loras||[]).length}));
}
function qwenPickCatalogEntry(entry){if(!entry||!entry.present)return;if(entry.kind.startsWith('prompt_enhancer_')){window.QwenPE?.selectEntry(entry);return}if(entry.kind==='lora'){const l=$('#qwenLoraSelect');if(l)l.value=entry.path;$('#qwenAdvanced').open=true;showView('image');return}const kind=entry.kind==='text_encoder_file'?'text_encoder':entry.kind;const spec=QWEN_COMPONENT_PICKERS[kind];if(!spec)return;if(kind==='repository'){const target=$('#qwenModelDir');target.value=entry.path;qwenModelDirty=true;qwenRenderComponentPickers();qwenRenderProfileComponentSummary();}else qwenSetComponentPath(kind,entry.path);const section=$('#'+spec.section);if(section)section.open=true;section?.scrollIntoView({behavior:'smooth',block:'start'});}
function qwenOpenAddForKind(kind){qwenModelTab('add');const local=$('#qwenLocalKind');if(local&&[...local.options].some(o=>o.value===kind))local.value=kind;const remoteMap={repository:'repository',transformer:'transformer',lora:'lora',vae:'vae_repository',text_encoder:'text_encoder_repository',processor:'processor_repository',scheduler:'scheduler_repository'},hub=$('#hubKind');if(hub&&remoteMap[kind]&&[...hub.options].some(o=>o.value===remoteMap[kind]))hub.value=remoteMap[kind];const up=$('#qwenUploadKind');if(up&&['transformer','lora','vae','text_encoder','text_encoder_file','prompt_enhancer_t2i','prompt_enhancer_i2i'].includes(kind))up.value=kind==='text_encoder'?'text_encoder_file':kind;up?.dispatchEvent(new Event('change'));qwenFormatState();setStatus('qwenModelStatus','Add or download the component. When it is ready, it will appear in the My models selectors.');window.scrollTo({top:0,behavior:'smooth'});}
function renderQwenHub(d){
 qwenHub=d;const model=d.models?.[0],cfg=model?.config||{};if(S.boot)S.boot.models=d.models||[];
 if(!qwenModelDirty){$('#qwenModelRetain').checked=cfg.retain_latents!==false;for(const [k,id] of Object.entries(QWEN_MODEL_FIELDS)){$('#'+id).value=cfg[k]??({gpu_memory:'21GiB',cpu_memory:'44GiB',default_steps:40,default_cfg:1,text_encoder_dtype:'auto',decode_device:'configured',vae_dtype:'auto',vae_decode_mode:'auto',tile_height:1024,tile_width:1024,tile_stride_height:768,tile_stride_width:768}[k]??'')}const path=cfg.model_path||'';$('#qwenModelFormat').value=path?(path.toLowerCase().endsWith('.gguf')?'gguf':'safetensors'):'original';}
 qwenFormatState();renderQwenValidation(model?.validation);renderQwenAccounts(d.accounts);
 const cap=d.runtime?.capabilities||{};$('#runtimeState').textContent=d.runtime?.ok?'Qwen runtime · Online':'Qwen runtime · Offline';$('#qwenRuntimeStatus').textContent=cap.qwen_pipeline?'Qwen pipeline available':'Qwen pipeline not available';$('#qwenModelState').textContent='Qwen Image 2.1 · '+tr(model?.validation?.ready?'Ready':'Configure files');
 $('#qwenLoraSupport').textContent=cap.lora?tr('This runtime exposes Qwen LoRA loading. Use only verified compatible adapters.'):tr('This runtime does not expose Qwen LoRA loading. Adapters cannot be applied.');$('#qwenLoraAdd').disabled=!cap.lora;
 const items=d.local_catalog||[];
 $('#qwenLocalCatalog').innerHTML=items.map(e=>`<article class="qwen-file-row${e.selectable===false?' qwen-file-invalid':''}"><b data-no-i18n>${esc(e.name||e.path.split('/').pop())}</b> <span class="status-pill">${esc(tr(({repository:'Repository',transformer:'Transformer',vae:'VAE / Decoder',text_encoder:'Text encoder',text_encoder_file:'Encoder weights',processor:'Processor',scheduler:'Scheduler',lora:'LoRA',prompt_enhancer_t2i:'PE-T2I · Prompt Enhancer',prompt_enhancer_i2i:'PE-I2I · Prompt Enhancer'})[e.kind]||e.kind))} · ${tr(e.present?'Present':'Missing')}${e.selectable===false?' · '+tr('Not selectable'):''}${e.active?' · '+tr('Active'):''}</span><code data-no-i18n>${esc(e.path)}</code>${e.issue?`<small class="qwen-field-help qwen-invalid-help">${esc(tr(e.issue))}</small>`:''}<div class="inline-actions">${['repository','transformer','vae','text_encoder','text_encoder_file','processor','scheduler','prompt_enhancer_t2i','prompt_enhancer_i2i'].includes(e.kind)&&e.present&&e.selectable!==false?`<button class="primary" data-qwen-select="${esc(e.id)}">${tr('Select in configuration')}</button>`:''}${e.kind==='text_encoder_file'&&e.present&&e.selectable!==false?`<button class="ghost" data-studio-encoder="${esc(e.id)}">${tr('Choose in Create / Edit')}</button>`:''}${e.kind==='lora'&&e.present?`<button class="ghost" data-qwen-use-lora="${esc(e.id)}">${tr('Choose in Create / Edit')}</button>`:''}${e.registered?`<button class="ghost danger" data-qwen-unlink="${esc(e.id)}">${tr('Unlink (keep file)')}</button>`:''}</div></article>`).join('')||`<p class="muted">${tr('No Qwen files linked yet. Open Add to link or download them.')}</p>`;
 for(const [id,kind,label] of [['qwenTransformerCatalog','transformer','Choose a catalog file…'],['qwenLoraSelect','lora','Choose a catalog file…'],['qwenJobTextEncoder','text_encoder_file','Use model configuration'],['qwenJobVae','vae','Use model configuration']]){const sel=$('#'+id);if(!sel)continue;const old=sel.value;sel.replaceChildren(new Option(tr(label),''),...items.filter(x=>x.kind===kind&&x.present&&x.selectable!==false).map(x=>new Option(x.name||x.path.split('/').pop(),x.path)));if([...sel.options].some(x=>x.value===old))sel.value=old;}
 qwenRenderComponentPickers();qwenRenderProfileComponentSummary();
}
async function refreshModelsHub(){if(qwenHubLoading)return qwenHubLoading;qwenHubLoading=hubApi('status').then(d=>{renderQwenHub(d);window.dispatchEvent(new CustomEvent('qwen-hub-updated',{detail:d}));return d}).finally(()=>qwenHubLoading=null);return qwenHubLoading;}
function readQwenModelForm(){
 const config={};for(const [key,id] of Object.entries(QWEN_MODEL_FIELDS)){const value=$('#'+id).value.trim();config[key]=key==='scheduler_shift'?(value===''?null:Number(value)):(key.startsWith('default_')||key.startsWith('tile_')?Number(value):value)}
 config.retain_latents=$('#qwenModelRetain').checked;
 return config;
}
async function saveQwenModel(){
 const config=readQwenModelForm();
 if($('#qwenModelFormat').value==='original')config.model_path='';else{const ext=$('#qwenModelFormat').value==='gguf'?'.gguf':'.safetensors';if(!config.model_path.toLowerCase().endsWith(ext))throw Error(tr('Select the matching transformer file extension: {extension}',{extension:ext}))}
 const d=await api('/api/models/qwen_image_21',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:true,config})});qwenModelDirty=false;renderQwenValidation(d.validation);await refreshModelsHub();if(typeof q22ApplyModelDecode==='function')q22ApplyModelDecode(d.model.config);setStatus('qwenModelStatus','Configuration saved. Changes apply to the next generation.');
}
function qwenSafeLink(value){try{const u=new URL(value);return u.protocol==='https:'?u.href:'#'}catch(_){return '#'}}
async function inspectQwenDownload(){qwenDownloadPreview=null;$('#hubPreview').hidden=true;setStatus('qwenModelStatus','Inspecting model source…');const d=await hubApi('inspect',{method:'POST',body:JSON.stringify({url:$('#hubUrl').value.trim(),kind:$('#hubKind').value,filename:$('#hubFilename').value.trim(),exclude_transformer:$('#qwenRepoWithoutTransformer').checked})});const p=d.preview;qwenDownloadPreview=p;$('#hubPreviewName').textContent=p.filename;$('#hubPreviewMeta').textContent=`${formatBytes(p.size||0)} · ${tr(({repository:'Repository',vae_repository:'VAE / Decoder',text_encoder_repository:'Text encoder',processor_repository:'Processor',scheduler_repository:'Scheduler',transformer:'Transformer',vae:'VAE / Decoder',lora:'LoRA'})[p.kind]||p.kind)} · ${p.license||tr('Check the model licence')}`;$('#hubPreviewSource').href=qwenSafeLink(p.license_url||p.source);$('#qwenDownloadFiles').textContent=JSON.stringify({revision:p.revision,files:p.file_count||1,sha256:p.sha256||null,skip_transformer_weights:!!p.exclude_transformer},null,2);$('#hubAcceptTerms').checked=false;$('#hubPreview').hidden=false;setStatus('qwenModelStatus','Review this exact source before starting.');}
async function startQwenDownload(){if(!qwenDownloadPreview)throw Error(tr('Inspect the source first.'));if(!$('#hubAcceptTerms').checked)throw Error(tr('Review and accept the model licence before downloading.'));await hubApi('downloads',{method:'POST',body:JSON.stringify({preview_id:qwenDownloadPreview.id,accept_terms:true})});qwenDownloadPreview=null;$('#hubPreview').hidden=true;qwenModelTab('downloads');setStatus('qwenModelStatus','Download queued on the server.');window.ZetalvxDownloadManager?.refresh(true);}
async function finalizeQwenUpload(opts={}){
 const u=qwenActiveUpload;if(!u)return false;
 let d=await hubApi(`uploads/${u.id}/finalize`,{method:'POST',body:JSON.stringify({accept_unknown:$('#qwenUploadConfirm').checked})});
 if(d.needs_confirmation){
   if(!confirm(tr('The file header cannot prove Qwen Image 2.1 compatibility. Have you verified the source and licence?'))){
      setStatus('qwenUploadStatus','Upload complete but not installed. Confirm compatibility or cancel.');return false;
   }
   d=await hubApi(`uploads/${u.id}/finalize`,{method:'POST',body:JSON.stringify({accept_unknown:true})});
 }
 if(d.needs_confirmation)throw Error('Compatibility confirmation still required.');
 qwenActiveUpload=null;
 if(!opts.keepInput)$('#qwenModelUpload').value='';
 if(!opts.deferRefresh)await refreshModelsHub();
 if(!opts.quiet)setStatus('qwenUploadStatus',tr('Upload complete: {path}',{path:d.path}));
 return true;
}
async function uploadQwenOneFile(file, index, total){
 const d=await hubApi('uploads',{method:'POST',body:JSON.stringify({filename:file.name,size:file.size,kind:$('#qwenUploadKind').value})});
 qwenActiveUpload={...d.upload,complete:false};const size=Math.min(d.chunk_size||8388608,8388608);
 for(let offset=0;offset<file.size;offset+=size){
   if(qwenUploadCancelRequested)throw Error('Upload cancelled.');
   const chunk=file.slice(offset,Math.min(file.size,offset+size));const fd=new FormData();
   fd.append('offset',String(offset));fd.append('chunk',chunk,'chunk');
   const r=await api(`/api/model-hub/uploads/${qwenActiveUpload.id}/chunk`,{method:'POST',body:fd});
   const fraction=r.upload.received/file.size;
   $('#qwenUploadProgress').value=(index+fraction)/total*100;
   setStatus('qwenUploadStatus',`LoRA / file ${index+1} of ${total}: ${file.name} · ${formatBytes(r.upload.received)} / ${formatBytes(file.size)}`);
 }
 if(qwenUploadCancelRequested)throw Error('Upload cancelled.');
 qwenActiveUpload.complete=true;
 if(!await finalizeQwenUpload({deferRefresh:true,keepInput:true,quiet:true}))throw Error('File compatibility not confirmed.');
 $('#qwenUploadProgress').value=(index+1)/total*100;
}
async function cancelCurrentUploadStage(){
 if(!qwenActiveUpload)return;
 const id=qwenActiveUpload.id;
 try{await hubApi('uploads/'+id,{method:'DELETE'});}finally{qwenActiveUpload=null;}
}
async function startQwenUpload(){
 if(qwenUploadRunning)return;
 qwenUploadRunning=true;qwenUploadCancelRequested=false;
 $('#qwenUploadStart').disabled=true;$('#qwenUploadCancel').disabled=false;
 try{
   const folderMode=$('#qwenUploadMode').value==='folder';
   if(folderMode&&$('#qwenUploadKind').value!=='lora')throw Error('Folder upload is available only for LoRA files.');
   if(qwenActiveUpload?.complete&&!folderMode){await finalizeQwenUpload();return;}
   if(qwenActiveUpload)throw Error('Cancel the interrupted upload before starting another.');
   let files;
   if(folderMode){
      const selected=Array.from($('#qwenUploadFolder').files||[]);
      if(!selected.length)throw Error('Choose a LoRA folder first.');
      files=selected.filter(f=>/\.safetensors$/i.test(f.name));
      if(!files.length)throw Error('No .safetensors files found in this folder.');
   }else{
      const file=$('#qwenModelUpload').files[0];if(!file)throw Error(tr('Choose a model file first.'));files=[file];
   }
   if(files.length>250)throw Error('Upload at most 250 LoRA files per folder.');
   if(!$('#qwenUploadConfirm').checked)throw Error('Confirm source compatibility and licence before uploading.');
   let ok=0,errors=[];
   for(let i=0;i<files.length;i++){
      if(qwenUploadCancelRequested)break;
      try{await uploadQwenOneFile(files[i],i,files.length);ok++;}
      catch(err){
         errors.push({file:files[i].webkitRelativePath||files[i].name,error:String(err.message||err)});
         // For multi-file imports, discard incomplete staging to allow the next file.
         // For a single-file retry, retain an interrupted upload as before.
         if(folderMode||qwenUploadCancelRequested)try{await cancelCurrentUploadStage();}catch(_){}
         if(!folderMode)throw err;
      }
   }
   if(ok)await refreshModelsHub();
   if(folderMode){
      const msg=`LoRA folder: ${ok} uploaded, ${errors.length} failed${qwenUploadCancelRequested?' (cancelled)':''}. Each successful file is listed separately in Available components.`;
      setStatus('qwenUploadStatus',msg,errors.length>0);
      if(errors.length)showDetails('LoRA folder upload errors',errors);
      if(!qwenUploadCancelRequested)$('#qwenUploadFolder').value='';
   }else{
      $('#qwenModelUpload').value='';setStatus('qwenUploadStatus',`Upload complete: ${files[0].name}`);
   }
 }catch(e){setStatus('qwenUploadStatus',e.message,true)}
 finally{qwenUploadRunning=false;$('#qwenUploadStart').disabled=false;$('#qwenUploadCancel').disabled=!qwenActiveUpload;}
}
async function cancelQwenUpload(){
 if(qwenUploadRunning){qwenUploadCancelRequested=true;setStatus('qwenUploadStatus','Cancelling after the current chunk…');return;}
 if(!qwenActiveUpload)return;
 await cancelCurrentUploadStage();$('#qwenUploadCancel').disabled=true;
 setStatus('qwenUploadStatus','Upload cancelled.');
}
let qwenBrowserTarget='',qwenBrowserKind='any',qwenBrowserParent='';
async function qwenBrowse(path=''){const d=await api('/api/setup/browse?path='+encodeURIComponent(path));$('#browserPath').value=d.path;qwenBrowserParent=d.parent;const encoderMode=qwenBrowserKind==='text_encoder';const completeEncoder=encoderMode&&d.entries.some(x=>!x.directory&&String(x.name).toLowerCase()==='config.json');$('#browserSelectDir').hidden=qwenBrowserKind==='file'||(encoderMode&&!completeEncoder);const visible=d.entries.filter(x=>x.directory||qwenBrowserKind!=='dir').filter(x=>!encoderMode||x.directory||/\.safetensors$/i.test(x.name));$('#browserEntries').innerHTML=visible.map(x=>`<button class="ghost" data-browser-path="${esc(x.path)}" data-no-i18n data-browser-dir="${x.directory?'1':'0'}">${x.directory?'📁':'▧'} ${esc(x.name)}</button>`).join('');$('#fileBrowser').classList.remove('hidden');}
function qwenBrowserChoose(path){const el=$('#'+qwenBrowserTarget);if(el){el.value=path;el.dispatchEvent(new Event('change',{bubbles:true}));}$('#fileBrowser').classList.add('hidden');}
let qwenFolderScanResult=null;
function renderQwenFolderPreview(scan){
  qwenFolderScanResult=scan;
  const container=$('#qwenFolderPreview');container.hidden=false;
  $('#qwenFolderCount').textContent=`${scan.total} models found${scan.truncated?' (scan limit reached)':''}`;
  const fileKinds=[['','Choose type'],['transformer','Transformer / checkpoint'],['lora','LoRA'],['vae','VAE'],['text_encoder_file','Text encoder'],['prompt_enhancer_t2i','PE-T2I'],['prompt_enhancer_i2i','PE-I2I']];
  const folderKinds=[['','Choose type'],['repository','Diffusers repository'],['vae','VAE directory'],['text_encoder','Text encoder folder'],['processor','Processor'],['scheduler','Scheduler']];
  const preferred=$('#qwenLocalKind')?.value;
  $('#qwenFolderEntries').innerHTML=scan.candidates.map((r,i)=>{
     const kinds=r.folder?folderKinds:fileKinds;
     const inferred=(!r.folder&&preferred==='lora'&&/\.safetensors$/i.test(r.name))?'lora':r.kind;
     const selected=!!r.selected||inferred==='lora'&&preferred==='lora';
     return `<label class="hub-folder-item"><input type="checkbox" data-folder-index="${i}" ${selected?'checked':''}/><span><b data-no-i18n>${esc(r.name)}</b><code data-no-i18n>${esc(r.relative)}</code></span><select data-folder-kind="${i}" aria-label="Component type">${kinds.map(([v,n])=>`<option value="${v}" ${v===inferred?'selected':''}>${n}</option>`).join('')}</select></label>`;
  }).join('')||'<p class="muted">No supported model files found in this folder.</p>';
  $('#qwenFolderStatus').textContent='Review the component types and select the models to link.';
}

function bindQwenModels(){
 const dirtyIds=new Set([...Object.values(QWEN_MODEL_FIELDS),'qwenModelFormat','qwenModelRetain']);document.addEventListener('input',e=>{if(dirtyIds.has(e.target.id)){qwenModelDirty=true;if(e.target.id==='qwenModelDir'||e.target.id==='qwenVaeConfig'){qwenRenderComponentPickers();qwenRenderProfileComponentSummary();}}});document.addEventListener('change',e=>{if(dirtyIds.has(e.target.id)){qwenModelDirty=true;qwenRenderProfileComponentSummary();}});
 $('#qwenModelFormat').onchange=qwenFormatState;$('#hubKind').onchange=qwenFormatState;
 for(const [kind,spec] of Object.entries(QWEN_COMPONENT_PICKERS)){const sel=$('#'+spec.picker);if(sel)sel.onchange=()=>{if(kind==='repository'){const entry=qwenHub?.local_catalog?.find(x=>x.id===sel.value);if(entry){$('#qwenModelDir').value=entry.path;qwenModelDirty=true;qwenRenderComponentPickers();qwenRenderProfileComponentSummary();}}else{const entry=qwenHub?.local_catalog?.find(x=>x.id===sel.value);qwenSetComponentPath(kind,entry?.path||'');}};}
 $('#qwenLocalPath').addEventListener('input',()=>{qwenFolderScanResult=null;$('#qwenFolderPreview').hidden=true;});
 $('#qwenModelsRefresh').onclick=e=>uiAction(e.currentTarget,refreshModelsHub);$('#qwenModelSave').onclick=e=>uiAction(e.currentTarget,saveQwenModel);
 $('#qwenModelCheck').onclick=e=>uiAction(e.currentTarget,async()=>{if(qwenModelDirty){setStatus('qwenModelStatus','Save your configuration before checking the new paths.',true);return}const d=await api('/api/models/qwen_image_21/validate',{method:'POST'});renderQwenValidation(d.validation);setStatus('qwenModelStatus',d.validation.ready?'Qwen files are ready.':'Some Qwen components are missing.');});
 $('#qwenModelProbe').onclick=e=>uiAction(e.currentTarget,async()=>{const d=await api('/api/models/qwen_image_21/probe',{method:'POST'});showDetails('Qwen runtime diagnostics',d.probe)});
 $('#qwenLocalAdd').onclick=e=>uiAction(e.currentTarget,async()=>{
   const input=$('#qwenLocalPath'),raw=input.value.trim();if(!raw)throw Error(tr('Choose a model file or folder first.'));
   setStatus('qwenLocalStatus','Checking server files…');
   let result;
   try{result=await hubApi('local',{method:'POST',body:JSON.stringify({path:raw,kind:$('#qwenLocalKind').value,accept_unknown:$('#qwenLocalConfirm').checked})});}
   catch(error){setStatus('qwenLocalStatus',error.message,true);throw error;}
   const linked=result?.entry?.path||raw;
   input.dataset.browseDirectory=result.browse_directory||qwenBrowseStartDirectory(linked);
   // Keep the folder path for another scan/import; clear only individual files.
   if(!result.bulk)input.value='';
   await refreshModelsHub();
   if(result.bulk){
      const msg=`LoRA folder: ${result.linked} file(s) linked, ${result.failed} rejected. Each LoRA appears separately in Available components.`;
      setStatus('qwenLocalStatus',msg,!!result.failed);
      setStatus('qwenModelStatus',msg,!!result.failed);
      if(result.errors?.length)showDetails('LoRA files not linked',result.errors);
   }else{
      setStatus('qwenLocalStatus','File or component linked without copying. It is now available in My models.');
      setStatus('qwenModelStatus','Linked without copying. It is now available in the My models selectors.');
   }
 });
 $('#qwenFolderScan').onclick=e=>uiAction(e.currentTarget,async()=>{
   const raw=$('#qwenLocalPath').value.trim();if(!raw)throw Error('Choose a server folder first.');
   const scan=await hubApi('local/scan-folder',{method:'POST',body:JSON.stringify({path:raw})});
   renderQwenFolderPreview(scan);
 });
 $('#qwenFolderToggle').onclick=()=>{
   const checks=[...$('#qwenFolderEntries').querySelectorAll('[data-folder-index]')];
   const allChecked=checks.length>0&&checks.every(x=>x.checked);
   checks.forEach(x=>x.checked=!allChecked);
 };
 $('#qwenFolderImport').onclick=e=>uiAction(e.currentTarget,async()=>{
   if(!qwenFolderScanResult)throw Error('Scan a folder first.');
   const entries=qwenFolderScanResult.candidates.flatMap((r,i)=>{
     const selected=$(`[data-folder-index="${i}"]`)?.checked;
     const kind=$(`[data-folder-kind="${i}"]`)?.value;
     return selected&&kind?[{path:r.path,kind}]:[];
   });
   if(!entries.length)throw Error('Select at least one model and a component type.');
   if(entries.length>250)throw Error('Select at most 250 models per import.');
   if(!confirm(`Link ${entries.length} selected model(s) without copying?`))return;
   const result=await hubApi('local/link-folder',{method:'POST',body:JSON.stringify({root:qwenFolderScanResult.root,entries,accept_unknown:$('#qwenLocalConfirm').checked})});
   await refreshModelsHub();
   $('#qwenFolderStatus').textContent=`Linked ${result.linked} model(s), ${result.failed} failed. Each LoRA remains an individual catalog entry.`;
   if(result.errors?.length)showDetails('Models not linked',result.errors);
 });
 const updateUploadSelection=()=>{
   const kind=$('#qwenUploadKind').value,mode=$('#qwenUploadMode');
   const folderOption=[...mode.options].find(x=>x.value==='folder');folderOption.disabled=kind!=='lora';
   if(kind!=='lora'&&mode.value==='folder')mode.value='file';
   $('#qwenUploadFileLabel').hidden=mode.value==='folder';
   $('#qwenUploadFolderLabel').hidden=mode.value!=='folder';
   $('#qwenModelUpload').accept=kind==='transformer'?'.gguf,.safetensors':'.safetensors';
 };
 $('#qwenUploadKind').addEventListener('change',updateUploadSelection);
 $('#qwenUploadMode').addEventListener('change',updateUploadSelection);
 updateUploadSelection();
 $('#hubInspect').onclick=e=>uiAction(e.currentTarget,inspectQwenDownload);$('#hubStartDownload').onclick=e=>uiAction(e.currentTarget,startQwenDownload);$('#qwenUploadStart').onclick=startQwenUpload;$('#qwenUploadCancel').onclick=e=>uiAction(e.currentTarget,cancelQwenUpload,'qwenUploadStatus');
 function qwenBrowseStartDirectory(raw){
 const path=String(raw||'').trim();if(!path)return '';
 const normalized=path.replace(/\\/g,'/').replace(/\/+$/,'');
 const last=normalized.split('/').pop()||'';
 // File extensions used by this hub. A server browser must always start from a folder.
 if(/\.(?:safetensors|gguf|json|bin|pt|pth|ckpt)$/i.test(last)){const parent=normalized.slice(0,Math.max(1,normalized.lastIndexOf('/')))||'/';return /^[A-Za-z]:$/.test(parent)?parent+'/':parent;}
 return normalized;
}
document.addEventListener('click',e=>{
  const tab=e.target.closest('[data-hub-tab]');if(tab){qwenModelTab(tab.dataset.hubTab);return}
  const select=e.target.closest('[data-qwen-select]');if(select){const entry=qwenHub?.local_catalog?.find(x=>x.id===select.dataset.qwenSelect);if(entry){qwenPickCatalogEntry(entry);if(!entry.kind.startsWith('prompt_enhancer_'))setStatus('qwenModelStatus','Component selected in the form. Save the configuration or save it as a model profile.');}return}
  const unlink=e.target.closest('[data-qwen-unlink]');if(unlink){uiAction(unlink,async()=>{if(!confirm(tr('Remove this catalog entry? The original file is kept.')))return;await hubApi('local/'+encodeURIComponent(unlink.dataset.qwenUnlink),{method:'DELETE'});await refreshModelsHub();});return}
  const lora=e.target.closest('[data-qwen-use-lora]');if(lora){const entry=qwenHub?.local_catalog?.find(x=>x.id===lora.dataset.qwenUseLora);if(entry){$('#qwenLoraSelect').value=entry.path;$('#qwenAdvanced').open=true;showView('image');}return}
  const add=e.target.closest('[data-qwen-add-kind]');if(add){qwenOpenAddForKind(add.dataset.qwenAddKind);return}
  const tabTarget=e.target.closest('[data-hub-tab-target]');if(tabTarget){qwenModelTab(tabTarget.dataset.hubTabTarget);return}
  const browse=e.target.closest('[data-qwen-browse]');if(browse){qwenBrowserTarget=browse.dataset.qwenBrowse;qwenBrowserKind=browse.dataset.browseKind||'any';if(qwenBrowserTarget==='qwenLocalPath'&&$('#qwenLocalKind')?.value==='text_encoder')qwenBrowserKind='text_encoder';const field=$('#'+qwenBrowserTarget);let path=field.value||field.dataset.browseDirectory||qwenHub?.paths?.repository||'';path=qwenBrowseStartDirectory(path);uiAction(browse,()=>qwenBrowse(path));return}
  const item=e.target.closest('[data-browser-path]');if(item){if(item.dataset.browserDir==='1')qwenBrowse(item.dataset.browserPath).catch(err=>alert(err.message));else qwenBrowserChoose(item.dataset.browserPath)}
 });
 $('#browserClose').onclick=()=>$('#fileBrowser').classList.add('hidden');$('#browserGo').onclick=()=>qwenBrowse($('#browserPath').value).catch(e=>alert(e.message));$('#browserUp').onclick=()=>qwenBrowse(qwenBrowserParent).catch(e=>alert(e.message));$('#browserSelectDir').onclick=()=>qwenBrowserChoose($('#browserPath').value);
 $$('[data-account-save]').forEach(b=>b.onclick=()=>uiAction(b,async()=>{const provider=b.dataset.accountSave,input=$(provider==='huggingface'?'#hubHfToken':'#hubCivitaiToken');try{const d=await hubApi('accounts/'+provider,{method:'PUT',body:JSON.stringify({token:input.value.trim(),store_without_test:$('#hubSkipTokenTest').checked})});renderQwenAccounts(d.accounts);setStatus('qwenModelStatus','Token saved on the server. No platform password is requested.')}finally{input.value=''}}));
 $$('[data-account-delete]').forEach(b=>b.onclick=()=>uiAction(b,async()=>{if(!confirm(tr('Remove this token from the app?')))return;const d=await hubApi('accounts/'+b.dataset.accountDelete,{method:'DELETE'});renderQwenAccounts(d.accounts);setStatus('qwenModelStatus','Token removed from the app. Revoke it separately on the platform if needed.');}));
}
bindQwenModels();
