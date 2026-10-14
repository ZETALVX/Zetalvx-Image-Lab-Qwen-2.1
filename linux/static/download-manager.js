/* Modified for Zetalvx Image Lab Qwen 1.0.24: interface/localization only; Apache-2.0, retained NOTICE. */
/* Zetalvx Image Lab — Qwen 2.1 Edition 0.1.0.41 — persistent model download manager. */
(()=>{'use strict';
 const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
 const t=s=>window.ZI18n?.t(s)||s;
 const ACTIVE=new Set(['queued','downloading','validating','installing','cancelling','running','needs_confirmation']);
 let jobs=[],timer=null,busy=false,filter='all',managerSettings={parallel_downloads:1,running_slots:0,max_parallel_downloads:3};
 const bytes=n=>{n=Number(n)||0;const u=['B','KiB','MiB','GiB'];let i=0;while(n>=1024&&i<3){n/=1024;i++}return `${n.toFixed(i>1?1:0)} ${u[i]}`};
 const duration=s=>{s=Math.max(0,Number(s)||0);if(!s)return '';if(s<60)return `${Math.ceil(s)}s`;const m=Math.floor(s/60),r=Math.ceil(s%60);return `${m}m ${r}s`};
 const status=s=>t(({queued:'Queued',downloading:'Downloading',validating:'Validating',installing:'Installing',cancelling:'Cancelling…',needs_confirmation:'Confirmation required',complete:'Complete',failed:'Failed',cancelled:'Cancelled',interrupted:'Interrupted'})[s]||s);
 const esc=s=>String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
 function logText(j){
  const h=j.history||[];
  let out=h.map(x=>`${new Date((x.at||0)*1000).toLocaleString()}  ${status(x.status)}${x.message?' · '+x.message:''}`).join('\n');
  if(j.error)out+=(out?'\n':'')+t('Error')+': '+j.error;
  if(j.warning)out+=(out?'\n':'')+t('Note')+': '+j.warning;
  if(j.path)out+=(out?'\n':'')+t('Path')+': '+j.path;
  if(j.sha256)out+=(out?'\n':'')+'SHA-256: '+j.sha256;
  return out||t('No events recorded.');
 }
 function card(j,compact=false){
  const total=Number(j.total_bytes)||0,done=Number(j.downloaded_bytes)||0;
  const pct=total?Math.max(0,Math.min(100,done/total*100)):0,running=ACTIVE.has(j.status);
  const speed=Number(j.speed_bytes_sec)||0,eta=Number(j.eta_seconds)||0;
  const fileInfo=j.file_count>1?`${j.completed_files||0}/${j.file_count} ${t('files')}`:'';const queueInfo=j.status==='queued'&&j.queue_position?`${t('Queue position')} ${j.queue_position}`:'';
  const progress=(running||total)?`<progress class="download-manager-progress" max="100" ${total?`value="${pct}"`:''}></progress>`:'';
  const cancelBtn=running?`<button class="ghost danger" data-download-cancel type="button">${esc(t('Cancel'))}</button>`:'';
  const detailsBtn=!compact?`<button class="ghost" data-download-details type="button">${esc(t('Details'))}</button>`:'';
  const details=!compact?`<details class="download-manager-log"><summary>${esc(t('Log and verification'))}</summary><pre data-no-i18n>${esc(logText(j))}</pre></details>`:'';
  return `<article class="download-manager-card" data-download-source="${esc(j.source)}" data-download-id="${esc(j.id)}">
   <div class="download-manager-head"><div class="download-manager-title"><b data-no-i18n>${esc(j.name||j.label)}</b><small>${esc(j.label)} · #${esc((j.id||'').slice(0,8))}</small></div><span class="download-manager-state">${esc(status(j.status))}</span></div>
   ${progress}
   <div class="download-manager-meta"><span>${total?`${bytes(done)} / ${bytes(total)} · ${pct.toFixed(0)}%`:done?bytes(done):''}</span>${speed&&running?`<span>${bytes(speed)}/s</span>`:''}${eta&&running?`<span>~${duration(eta)} ${esc(t('remaining'))}</span>`:''}${fileInfo?`<span>${esc(fileInfo)}</span>`:''}${queueInfo?`<span>${esc(queueInfo)}</span>`:''}${j.current_file?`<span data-no-i18n>${esc(j.current_file)}</span>`:''}</div>
   <div class="download-manager-actions">${cancelBtn}${detailsBtn}${j.status==='needs_confirmation'?`<button class="primary" data-download-confirm type="button">${esc(t('Confirm Qwen compatibility and install'))}</button>`:''}</div>${details}
  </article>`;
 }
 async function cancel(j,b){
  if(!confirm(t('Cancel this download? The incomplete temporary file will be removed.')))return;
  b.disabled=true;
  try{
   await api(`/api/model-hub/downloads/${encodeURIComponent(j.id)}/cancel`,{method:'POST'});
   await refresh(true);
  }catch(e){alert(e.message)}finally{b.disabled=false}
 }
 function bind(root){
  root.querySelectorAll('[data-download-confirm]').forEach(b=>b.onclick=async()=>{const a=b.closest('[data-download-id]'),j=jobs.find(x=>x.id===a.dataset.downloadId);if(!j)return;if(!confirm(t('The header cannot prove Qwen Image 2.1 compatibility. Confirm only after checking the source and licence.')))return;b.disabled=true;try{await api('/api/model-hub/downloads/'+encodeURIComponent(j.id)+'/install',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:['transformer','lora','vae'].includes(j.kind)?j.kind:'transformer',accept_unknown:true})});await refresh(true);await refreshModelsHub();}catch(e){alert(e.message)}finally{b.disabled=false}});

  root.querySelectorAll('[data-download-cancel]').forEach(b=>b.onclick=()=>{const a=b.closest('[data-download-id]'),j=jobs.find(x=>x.id===a.dataset.downloadId&&x.source===a.dataset.downloadSource);if(j)cancel(j,b)});
  root.querySelectorAll('[data-download-details]').forEach(b=>b.onclick=()=>{const d=b.closest('.download-manager-card').querySelector('.download-manager-log');d.open=!d.open});
 }
 function visible(){return jobs.filter(j=>filter==='active'?ACTIVE.has(j.status):filter==='finished'?!ACTIVE.has(j.status):true)}
 function render(){
  const count=jobs.filter(j=>ACTIVE.has(j.status)).length,queued=jobs.filter(j=>j.status==='queued').length,running=Number(managerSettings.running_slots)||jobs.filter(j=>['downloading','validating','installing'].includes(j.status)).length,badge=$('#downloadTabCount');if(badge)badge.textContent=count?String(count):'';const summary=$('#downloadQueueSummary');if(summary)summary.textContent=`${running} ${t('running')} · ${queued} ${t('queued')} · ${managerSettings.parallel_downloads||1} ${t('simultaneous')}`;const sel=$('#downloadConcurrency');if(sel&&!sel.matches(':focus'))sel.value=String(managerSettings.parallel_downloads||1);
  const main=$('#downloadManagerList');if(main){const v=visible();main.innerHTML=v.map(j=>card(j)).join('')||`<div class="download-manager-empty">${esc(t('No downloads recorded.'))}</div>`;bind(main)}
  const jp=$('#jobsDownloadPanel'),jl=$('#jobsDownloadList');if(jp&&jl){const v=jobs.filter(j=>ACTIVE.has(j.status)||['failed','interrupted'].includes(j.status)).slice(0,6);jp.hidden=!v.length;jl.innerHTML=v.map(j=>card(j,true)).join('');bind(jl)}
  const active=count>0;if(active&&!timer)timer=setInterval(()=>refresh(),1200);if(!active&&timer){clearInterval(timer);timer=null}
 }
 async function refresh(force=false){if(busy)return;busy=true;try{const d=await api('/api/download-manager');jobs=d.jobs||[];managerSettings=d.settings||managerSettings;render()}catch(e){if(force)console.warn('Download manager:',e)}finally{busy=false}}
 document.addEventListener('click',e=>{const f=e.target.closest('[data-download-filter]');if(f){filter=f.dataset.downloadFilter;$$('[data-download-filter]').forEach(x=>x.classList.toggle('active',x===f));render();return}if(e.target.closest('[data-hub-tab="downloads"]'))setTimeout(()=>refresh(true),0)});
 $('#downloadManagerRefresh')?.addEventListener('click',()=>refresh(true));
 $('#downloadConcurrency')?.addEventListener('change',async e=>{const select=e.currentTarget,old=managerSettings.parallel_downloads||1;select.disabled=true;try{const d=await api('/api/download-manager/settings',{method:'POST',body:JSON.stringify({parallel_downloads:Number(select.value)})});managerSettings={...managerSettings,...(d.settings||{})};await refresh(true)}catch(err){select.value=String(old);alert(err.message)}finally{select.disabled=false}});
 window.addEventListener('creator-view-change',e=>{if(['models','jobs'].includes(e.detail?.view))refresh(true)});
 window.addEventListener('focus',()=>refresh());document.addEventListener('visibilitychange',()=>{if(!document.hidden)refresh()});
 window.ZetalvxDownloadManager={refresh};setTimeout(()=>refresh(),800);
})();
