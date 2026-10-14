/* 1.0.2: authenticated export handling. No external URLs or tokens in links.
   <=128 MiB: same-session fetch -> bounded browser Blob (Safari-friendly).
   Larger exports: native streaming, Range-enabled endpoint; no giant JS Blob.
*/
(()=>{
 'use strict';
 const LIMIT=128*1024*1024, active=new Set();
 const tr=s=>window.ZI18n?.t(s)||s;
 function filename(header,fallback){
  const utf=/filename\*=UTF-8''([^;]+)/i.exec(header||'');
  const plain=/filename="([^"]*)"|filename=([^;]+)/i.exec(header||'');
  let n=fallback||'download';try{if(utf)n=decodeURIComponent(utf[1]);else if(plain)n=plain[1]||plain[2]}catch(_){}
  return n.replace(/[\\/\x00-\x1f\x7f]/g,'_').trim()||'download';
 }
 function save(url,name){
  const a=document.createElement('a');a.href=url;a.download=name;a.rel='noopener';
  document.body.append(a);a.click();a.remove();
 }
 function report(anchor,text,bad=false){
  let e=anchor.parentElement?.querySelector('[data-file-download-status]');
  if(!e){e=document.createElement('small');e.setAttribute('data-file-download-status','');e.setAttribute('role','status');anchor.parentElement?.append(e)}
  e.textContent=tr(text);e.classList.toggle('is-error',bad);
 }
 async function responseError(r){
  if(r.status===401)return tr('Session expired. Sign in again before downloading.');
  try{if((r.headers.get('Content-Type')||'').includes('json')){const d=await r.json();return d.error||`HTTP ${r.status}`}}catch(_){}
  return `Download: HTTP ${r.status}`;
 }
 async function start(anchor){
  const url=new URL(anchor.href,location.href);
  if(url.origin!==location.origin||!/^https?:$/.test(url.protocol))throw Error(tr('Only same-origin downloads are allowed.'));
  if(active.has(url.href))return;
  active.add(url.href);anchor.setAttribute('aria-busy','true');report(anchor,'Preparing download…');
  try{
   const options={credentials:'same-origin',cache:'no-store',redirect:'error'};
   const h=await fetch(url.href,{...options,method:'HEAD'});
   if(!h.ok)throw Error(await responseError(h));
   const disposition=h.headers.get('Content-Disposition')||'';
   if(!/^attachment\b/i.test(disposition))throw Error(tr('The server did not return a downloadable file.'));
   const name=filename(disposition,anchor.dataset.downloadName||'download');
   const length=Number(h.headers.get('Content-Length')||-1);
   if(!Number.isSafeInteger(length)||length<0||length>LIMIT){
    save(url.href,name);report(anchor,'Download handed to the browser.');return;
   }
   const r=await fetch(url.href,options);
   if(!r.ok)throw Error(await responseError(r));
   if(!/^attachment\b/i.test(r.headers.get('Content-Disposition')||''))throw Error(tr('The server did not return a downloadable file.'));
   const declared=Number(r.headers.get('Content-Length')||-1);
   if(!Number.isSafeInteger(declared)||declared<0||declared>LIMIT){await r.body?.cancel();throw Error(tr('File size changed. Retry the download.'));}
   if(!r.body?.getReader){await r.body?.cancel();save(url.href,name);report(anchor,'Download handed to the browser.');return;}
   const reader=r.body.getReader(),parts=[];let received=0;
   try{
    for(;;){const {done,value}=await reader.read();if(done)break;received+=value.byteLength;
     if(received>LIMIT||received>declared)throw Error(tr('Unexpected download size.'));
     parts.push(value);report(anchor,`${tr('Downloading…')} ${Math.round(received/Math.max(1,declared)*100)}%`);
    }
   }catch(e){await reader.cancel().catch(()=>{});throw e}finally{reader.releaseLock()}
   if(received!==declared)throw Error(tr('Incomplete download. Retry.'));
   const blob=new Blob(parts,{type:r.headers.get('Content-Type')||'application/octet-stream'});
   const object=URL.createObjectURL(blob),finalName=filename(r.headers.get('Content-Disposition'),name);
   // A visible second link preserves a real user gesture if a mobile browser
   // ignores the automatic click after an asynchronous fetch.
   anchor.parentElement?.querySelector('[data-ready-download]')?.remove();
   const ready=document.createElement('a');ready.href=object;ready.download=finalName;
   ready.className='button-link ghost small';ready.textContent=tr('Save file');ready.setAttribute('data-ready-download','');
   anchor.parentElement?.append(ready);save(object,finalName);
   // Bounded lifetime for the in-browser copy, not immediate revoke on Safari.
   setTimeout(()=>{URL.revokeObjectURL(object);ready.remove()},120000);
   report(anchor,'File ready. Check your browser downloads, or press Save file.');
  }catch(e){report(anchor,e.message||'Download failed.',true)}
  finally{active.delete(url.href);anchor.removeAttribute('aria-busy')}
 }
 document.addEventListener('click',event=>{
  const anchor=event.target.closest?.('a[data-safe-download]');
  if(!anchor||event.defaultPrevented||event.button>0||event.metaKey||event.ctrlKey||event.shiftKey||event.altKey)return;
  event.preventDefault();start(anchor).catch(e=>report(anchor,e.message,true));
 });
})();
