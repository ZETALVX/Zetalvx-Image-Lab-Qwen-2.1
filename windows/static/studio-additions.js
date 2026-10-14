/* 0.1.0.27 — one video/frame workflow. The player is a local object URL.
   FFmpeg produces the PNG preview; Save uses that same server preview ID.
   No new backend routes, conversion jobs, libraries or external services. */
(()=>{
 'use strict';
 const $=s=>document.querySelector(s), t=(s,v)=>window.ZI18n?.t(s,v)||s;
 const fileInput=$('#pickerFile'), player=$('#pickerVideo'), image=$('#pickerPreview');
 if(!fileInput||!player)return;
 let video=null,blobUrl='',epoch=0,revision=0,preview=null,desired=null,draining=false;
 let loading=false,saving=false,seekByCode=false,suppressPause=false,previewTimer=null;
 const setStatus=(s,bad=false)=>{const p=$('#pickerStatus');p.textContent=t(s);p.classList.toggle('is-error',bad)};
 function seconds(value){
  const raw=String(value).trim().replace(',','.');
  const chunks=raw.split(':'),parts=chunks.map(Number);
  const valid=chunks.length<=3&&chunks.every((x,i)=>i===chunks.length-1?/^(?:\d+(?:\.\d+)?|\.\d+)$/.test(x):/^\d+$/.test(x));
  if(!valid||parts.some(x=>!Number.isFinite(x)||x<0)||parts.slice(1).some(x=>x>=60))throw Error(t('Usa secondi oppure HH:MM:SS.mmm.'));
  return parts.reduce((n,x)=>n*60+x,0);
 }
 function validProject(){return !!video&&S.project?.id===video.project_id&&!pendingProjectId;}
 function invalidate(){
  revision++;preview=null;$('#pickerSave').disabled=true;
  image.hidden=true;$('#pickerPreviewEmpty').hidden=false;$('#pickerFrameInfo').textContent='';
 }
 function setPosition(n){
  $('#pickerRange').value=String(n);$('#pickerSeconds').value=String(Math.round(n*1000)/1000);
 }
 function pauseForSelection(){if(!player.paused)suppressPause=true;player.pause();}
 function seekPlayer(n){
  if(!Number.isFinite(player.duration))return;
  const target=Math.min(n,Math.max(0,player.duration-.001));
  if(Math.abs(player.currentTime-target)<.0001)return;
  seekByCode=true;player.currentTime=target;
 }
 async function discardServer(v){if(v)await api('/api/image-tools/video-picker/'+v.id,{method:'DELETE'}).catch(()=>{});}
 async function discard(){
  const old=video;epoch++;video=null;desired=null;clearTimeout(previewTimer);invalidate();
  pauseForSelection();player.removeAttribute('src');player.load();
  if(blobUrl){URL.revokeObjectURL(blobUrl);blobUrl='';}
  $('#pickerWorkspace').hidden=true;$('#pickerPlaybackWarning').hidden=true;
  fileInput.value='';$('#pickerFilename').textContent=t('Nessun file');$('#pickerLoad').disabled=true;
  setStatus('');await discardServer(old);
 }
 function queueFrame(mode='timestamp'){
  if(!video||saving||loading||!validProject())return;
  clearTimeout(previewTimer);
  try{
   const time=mode==='first'?0:mode==='last'?Math.max(0,video.duration-.001):seconds($('#pickerSeconds').value);
   if(time>=video.duration)throw Error(t('L’istante scelto è oltre la fine del video.'));
   invalidate();
   desired={video,epoch,revision,mode,time};
   setStatus('Preparazione anteprima…');drain();
  }catch(e){invalidate();setStatus(e.message,true);}
 }
 async function drain(){
  if(draining)return;draining=true;
  try{
   while(desired){
    const q=desired;desired=null;
    try{
     const d=await api('/api/image-tools/video-picker/'+q.video.id+'/preview',{
      method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:q.mode,timestamp:q.time})});
     if(epoch!==q.epoch||revision!==q.revision||!validProject())continue;
     const src='/api/image-tools/video-picker/'+q.video.id+'/preview/'+d.preview.id;
     // Only enable Save after the matching PNG has actually loaded in the browser.
     const loaded=await new Promise(resolve=>{
      const probe=new Image();const timer=setTimeout(()=>resolve(false),15000);
      probe.onload=()=>{clearTimeout(timer);resolve(true)};probe.onerror=()=>{clearTimeout(timer);resolve(false)};probe.src=src;
     });
     if(epoch!==q.epoch||revision!==q.revision||!validProject())continue;
     if(!loaded)throw Error(t('Anteprima non caricata. Riprova.'));
     preview=d.preview;image.src=src;image.hidden=false;$('#pickerPreviewEmpty').hidden=true;
     const label=q.mode==='last'?t('Ultimo frame esatto'):q.mode==='first'?t('Primo frame'):`${q.time.toFixed(3)} s`;
     const info=preview.info||{};$('#pickerFrameInfo').textContent=`${label} · ${info.width||video.width} × ${info.height||video.height}`;
     $('#pickerSave').disabled=false;setStatus('Anteprima pronta. Verrà salvato questo PNG.');
    }catch(e){if(epoch===q.epoch&&revision===q.revision){invalidate();setStatus(e.message,true)}}
   }
  }finally{draining=false;}
 }
 function schedulePreview(){
  clearTimeout(previewTimer);if(!video||!player.paused||saving||loading)return;
  previewTimer=setTimeout(()=>queueFrame(player.currentTime>=video.duration-.001?'last':'timestamp'),220);
 }
 fileInput.addEventListener('change',()=>{
  const f=fileInput.files[0];$('#pickerFilename').textContent=f?f.name:t('Nessun file');$('#pickerLoad').disabled=!f||loading;
 });
 $('#pickerLoad').addEventListener('click',async()=>{
  const file=fileInput.files[0];if(!file||loading||saving)return;
  if(!S.project){setStatus('Apri prima un progetto.',true);return;}
  if(file.size>120*1024**2){setStatus('Video oltre 120 MiB.',true);return;}
  const pid=S.project.id,old=video,token=++epoch;video=null;desired=null;invalidate();clearTimeout(previewTimer);
  pauseForSelection();player.removeAttribute('src');player.load();if(blobUrl){URL.revokeObjectURL(blobUrl);blobUrl='';}
  $('#pickerWorkspace').hidden=true;loading=true;fileInput.disabled=true;$('#pickerLoad').disabled=true;
  $('#pickerLoad').textContent=t('Caricamento…');setStatus('Caricamento video sul server…');
  try{
   await discardServer(old);
   const fd=new FormData();fd.append('file',file);fd.append('project_id',pid);
   const d=await api('/api/image-tools/video-picker',{method:'POST',body:fd});
   if(epoch!==token||S.project?.id!==pid||pendingProjectId){await discardServer(d.video);return;}
   video=d.video;blobUrl=URL.createObjectURL(file);$('#pickerPlaybackWarning').hidden=true;
   player.src=blobUrl;player.load();$('#pickerWorkspace').hidden=false;
   $('#pickerRange').max=String(Math.max(0,video.duration-.001));setPosition(0);loading=false;queueFrame('first');
  }catch(e){if(epoch===token)setStatus(e.message,true)}
  finally{loading=false;fileInput.disabled=false;$('#pickerLoad').disabled=!fileInput.files[0];$('#pickerLoad').textContent=t('Carica video');}
 });
 player.addEventListener('error',()=>{if(video)$('#pickerPlaybackWarning').hidden=false;});
 player.addEventListener('play',()=>{if(!video)return;clearTimeout(previewTimer);invalidate();setStatus('Metti in pausa per scegliere il fotogramma.');});
 player.addEventListener('timeupdate',()=>{if(video&&!seekByCode)setPosition(player.currentTime);});
 player.addEventListener('seeking',()=>{if(!seekByCode&&video){clearTimeout(previewTimer);invalidate();}});
 player.addEventListener('seeked',()=>{
  if(seekByCode){seekByCode=false;return;}
  if(video){setPosition(player.currentTime);schedulePreview();}
 });
 player.addEventListener('pause',()=>{if(suppressPause){suppressPause=false;return;}if(video&&!seekByCode){setPosition(player.currentTime);schedulePreview();}});
 $('#pickerRange').addEventListener('input',()=>{
  if(!video)return;clearTimeout(previewTimer);invalidate();pauseForSelection();setPosition(Number($('#pickerRange').value));setStatus('Rilascia il cursore per aggiornare l’anteprima.');
 });
 $('#pickerRange').addEventListener('change',()=>{if(!video)return;const n=Number($('#pickerRange').value);pauseForSelection();seekPlayer(n);queueFrame();});
 $('#pickerSeconds').addEventListener('input',()=>{clearTimeout(previewTimer);pauseForSelection();invalidate();setStatus('Aggiorna l’anteprima prima di salvare.');});
 function manualPreview(){if(!video)return;pauseForSelection();try{const n=seconds($('#pickerSeconds').value);seekPlayer(n);$('#pickerRange').value=String(n);queueFrame();}catch(e){invalidate();setStatus(e.message,true)}}
 $('#pickerSeconds').addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();manualPreview();}});
 $('#pickerRefresh').addEventListener('click',manualPreview);
 $('#pickerFirst').addEventListener('click',()=>{if(video){pauseForSelection();setPosition(0);seekPlayer(0);queueFrame('first');}});
 $('#pickerLast').addEventListener('click',()=>{if(video){pauseForSelection();const n=Math.max(0,video.duration-.001);setPosition(n);seekPlayer(n);queueFrame('last');}});
 $('#pickerSave').addEventListener('click',async()=>{
  if(!preview||!validProject()||saving||desired||draining)return;
  const v=video,p=preview,token=epoch;saving=true;$('#pickerSave').disabled=true;setStatus('Salvataggio PNG…');
  try{
   await api('/api/image-tools/video-picker/'+v.id+'/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({preview_id:p.id})});
   if(epoch===token&&validProject()){await openProjectNoPoll(v.project_id);setStatus('Fotogramma salvato nella libreria del progetto.');}
  }catch(e){if(epoch===token)setStatus(e.message,true)}
  finally{saving=false;if(epoch===token)$('#pickerSave').disabled=!preview||!validProject();}
 });
 $('#pickerRemove').addEventListener('click',()=>{if(!saving)discard();});
 window.addEventListener('studio-project-changed',()=>{if(video&&S.project?.id!==video.project_id)discard();});
 // Release browser memory without issuing unload-time network writes.
 window.addEventListener('pagehide',()=>{pauseForSelection();});
})();
