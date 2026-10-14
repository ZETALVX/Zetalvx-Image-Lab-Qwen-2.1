/* 1.0.4 — bounded generation/status requests. Never replay a POST/DELETE. */
(function(root){
 'use strict';
 function policy(url,opt){
  const p=new URL(url,root.document?.baseURI||'https://imagelab.invalid/').pathname;
  const method=(opt.method||'GET').toUpperCase(),read=method==='GET';
  const status=/^\/api\/(?:bootstrap|session\/tokens|projects\/[^/]+|jobs\/[^/]+)$/.test(p);
  const submit=p==='/api/jobs';
  return {method,read,timeout:status&&read?45000:submit&&method==='POST'?30000:0};
 }
 async function fetchText(url,opt={}){
  const p=policy(url,opt);
  if(!p.timeout){const response=await root.fetch(url,opt);return {response,text:await response.text()};}
  for(let attempt=0;attempt<(p.read?2:1);attempt++){
   const controller=new AbortController();let timer,timedOut=false;
   const onAbort=()=>controller.abort();
   if(opt.signal?.aborted){const e=new Error('Request cancelled');e.name='AbortError';throw e;}
   opt.signal?.addEventListener('abort',onAbort,{once:true});
   try{
    // The deadline also covers body reads. A stalled body must not leave a
    // button/poll permanently pending after the response headers arrive.
    const deadline=new Promise((_,reject)=>{timer=setTimeout(()=>{
     timedOut=true;controller.abort();const e=new Error('Server response timed out');e.name='TimeoutError';reject(e);
    },p.timeout);});
    const request=root.fetch(url,{...opt,signal:controller.signal,credentials:opt.credentials||'same-origin',...(p.read?{cache:'no-store'}:{})})
      .then(async response=>({response,text:await response.text()}));
    const result=await Promise.race([request,deadline]);
    if(p.read&&attempt===0&&[502,503,504].includes(result.response.status)){await new Promise(resolve=>setTimeout(resolve,800));continue;}
    return result;
   }catch(cause){
    if(opt.signal?.aborted)throw cause;
    const transient=timedOut||cause?.name==='TimeoutError'||cause instanceof TypeError||cause?.name==='TypeError';
    if(p.read&&attempt===0&&transient){await new Promise(resolve=>setTimeout(resolve,800));continue;}
    if(!transient)throw cause;
    const e=new Error(p.read?'Connection interrupted while updating the queue. Reconnecting automatically.':'Submission response not confirmed. The job may already be queued: check Queue & history before sending it again.');
    e.name=timedOut?'TimeoutError':'NetworkError';e.code=timedOut?'request_timeout':'network_error';
    e.submissionUncertain=!p.read;e.cause=cause;throw e;
   }finally{
    clearTimeout(timer);opt.signal?.removeEventListener('abort',onAbort);
   }
  }
 }
 root.ZetalvxRequestRecovery={fetchText,policy};
})(typeof window!=='undefined'?window:globalThis);
