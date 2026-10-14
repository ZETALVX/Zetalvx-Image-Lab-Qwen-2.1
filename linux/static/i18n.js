/* Modified for Zetalvx Image Lab Qwen 1.0.24: interface/localization only; Apache-2.0, retained NOTICE. */
/* Offline UI catalogue. No model, prompt, caption or user file is sent for translation. */
(()=>{'use strict';
 const C=window.ZETALVX_LOCALES||{languages:[],entries:[]};
 const storageKey='zetalvx.ui.language';const languages=new Map(C.languages.map(l=>[l.code,l]));
 const normal=s=>String(s).replace(/\s+/g,' ').trim();const table=new Map(),reverse=new Map(),folded=new Map(),patterns=[];
 for(const e of C.entries){
  for(const s of e.sources){table.set(normal(s),e);folded.set(normal(s).toLocaleLowerCase(),e);if(s.includes('{')){const names=[];const re=s.split(/(\{\w+\})/).map(x=>/^\{\w+\}$/.test(x)?(names.push(x.slice(1,-1)),'(.+?)'):x.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('');patterns.push({re:new RegExp('^'+re+'$'),names,e});}}
  for(const v of Object.values(e.text))if(!reverse.has(normal(v)))reverse.set(normal(v),e);
 }
 let saved='';try{saved=localStorage.getItem(storageKey)||''}catch(_){}
 let lang=languages.has(saved)?saved:(languages.has(navigator.language.split('-')[0])?navigator.language.split('-')[0]:'en');
 const unresolved=new Set();
 function t(s,vars={}){
  const raw=String(s??''),n=normal(raw);let e=table.get(n)||reverse.get(n)||folded.get(n.toLocaleLowerCase()),v={...vars};
  if(!e){for(const p of patterns){const m=n.match(p.re);if(m){e=p.e;p.names.forEach((k,i)=>v[k]=m[i+1]);break;}}}
  let value=e?(e.text[lang]||e.text.en||e.sources[0]):raw;
  if(!e && /[a-zA-ZÀ-ž\u4e00-\u9fff]/.test(n) && n.length<800)unresolved.add(n);
  return value.replace(/\{(\w+)\}/g,(all,k)=>Object.prototype.hasOwnProperty.call(v,k)?String(v[k]):all);
 }
 const skip='script,style,pre,code,textarea,input,[data-no-i18n],[data-language-picker],.hub-path,.hub-file-list,.details-json,.asset-info>b,.asset-info>small,[data-user-preset-id] b,.project-card b';
 const records=new WeakMap(),attrs=new WeakMap();let scheduled=false,observer;
 function translateNode(node){
  if(node.parentElement?.closest(skip))return;const raw=node.nodeValue;if(!raw?.trim())return;
  let record=records.get(node);if(!record||raw!==record.last){record={source:raw,last:raw};records.set(node,record)}
  const n=normal(record.source);let translated=t(n);
  if(translated===n){
   // Composed summaries consist of independent labelled values; preserve user content.
   if(n.includes(' · '))translated=n.split(' · ').map(x=>t(x)).join(' · ');
  }
  const value=record.source.match(/^\s*/)[0]+translated+record.source.match(/\s*$/)[0];record.last=value;
  if(node.nodeValue!==value)node.nodeValue=value;
 }
 function paint(){
  if(!document.body)return;
  observer?.disconnect();document.documentElement.lang=lang;document.documentElement.dir=languages.get(lang)?.rtl?'rtl':'ltr';
  if(document.body.dataset.authTitle)document.title=t(document.body.dataset.authTitle)+' · Zetalvx Image Lab — Qwen 2.1 Edition';
  const walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);while(walker.nextNode())translateNode(walker.currentNode);
  document.querySelectorAll('[placeholder],[title],[aria-label],[alt]').forEach(el=>{
   if(el.closest('[data-no-i18n]'))return;let m=attrs.get(el);if(!m){m={};attrs.set(el,m)}
   for(const k of ['placeholder','title','aria-label','alt']){
    const val=el.getAttribute(k);if(val===null)continue;let r=m[k];if(!r||val!==r.last)r=m[k]={source:val,last:val};r.last=t(r.source);if(val!==r.last)el.setAttribute(k,r.last);
   }
  });
  document.querySelectorAll('[data-language-picker]').forEach(sel=>{if(!sel.options.length){C.languages.forEach(l=>sel.add(new Option(l.name,l.code)));sel.addEventListener('change',()=>setLanguage(sel.value))}sel.value=lang});
  observer?.observe(document.body,{subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:['placeholder','title','aria-label','alt']});
 }
 function setLanguage(code){if(!languages.has(code))return;lang=code;try{localStorage.setItem(storageKey,code)}catch(_){};paint();window.dispatchEvent(new Event('languagechange'))}
 function schedule(){if(scheduled)return;scheduled=true;requestAnimationFrame(()=>{scheduled=false;paint()})}
 window.ZI18n={t,setLanguage,get language(){return lang},languages:C.languages,missing:()=>Array.from(unresolved)};
 const oldAlert=window.alert.bind(window),oldConfirm=window.confirm.bind(window),oldPrompt=window.prompt.bind(window);
 window.alert=s=>oldAlert(t(s));window.confirm=s=>oldConfirm(t(s));window.prompt=(s,value)=>oldPrompt(t(s),value);
 function init(){observer=new MutationObserver(schedule);paint()}
 if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init,{once:true});else init();
})();
