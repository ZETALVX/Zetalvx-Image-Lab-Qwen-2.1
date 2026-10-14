/* Windows 1.0.7: keep the host-local first-setup code live without persisting plaintext. */
(()=>{'use strict';
 function start(){
  const box=document.querySelector('[data-local-setup-code]');
  if(!box)return;
  const code=box.querySelector('code[data-no-i18n]');
  let timer=0,stopped=false;
  const schedule=ms=>{clearTimeout(timer);if(!stopped)timer=setTimeout(sync,Math.max(1000,Math.min(ms,15000)))};
  async function sync(){
   try{
    const r=await fetch('/api/first-run/local-code',{method:'GET',credentials:'same-origin',cache:'no-store',headers:{Accept:'application/json'}});
    if(r.status===409)return;
    if(!r.ok)throw Error('setup-code');
    const data=await r.json();
    if(data&&typeof data.code==='string'&&data.code&&code)code.textContent=data.code;
    const left=Math.max(1,Number(data&&data.expires_in)||5);
    box.dataset.expiresIn=String(Math.floor(left));
    schedule(Math.min(15000,(left+1)*1000));
   }catch(_){schedule(5000)}
  }
  window.addEventListener('pagehide',()=>{stopped=true;clearTimeout(timer)},{once:true});
  schedule(1000);
 }
 if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',start,{once:true});else start();
})();
