/* Modified for Zetalvx Image Lab Qwen 1.0.24: interface/localization only; Apache-2.0, retained NOTICE. */
/* 1.0.24: offline single-article guide, complete locale data. No user data is searched. */
(()=>{'use strict';
 const by=id=>document.getElementById(id),root=by('view-guide');if(!root)return;
 const data=window.QWEN_GUIDE_DATA,topics=data.topics,search=by('guideSearch');
 const t=(s,v={})=>window.ZI18n?.t(s,v)||s;
 const norm=s=>String(s).normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLocaleLowerCase().replace(/\s+/g,' ').trim();
 let language=window.ZI18n?.language||'en',selected=topics[0].id;
 const title=topic=>topic.title[language];
 const content=topic=>topic.body[language];
 const plain=html=>{const node=document.createElement('div');node.innerHTML=html;return node.textContent};
 function filter(){
  const words=norm(search.value).split(' ').filter(Boolean);let count=0;
  for(const topic of topics){
   const haystack=norm(title(topic)+' '+plain(content(topic))+' '+(topic.keywords||''));
   const visible=words.every(w=>haystack.includes(w));
   by('guideIndex').querySelector('[data-guide-topic="'+topic.id+'"]').hidden=!visible;
   if(visible)count++;
  }
  by('guideNoResults').hidden=count!==0;
  by('guideMatchCount').textContent=t('{count} of {total}',{count,total:topics.length});
  by('guideClear').disabled=!search.value;
 }
 function renderArticle(){
  const index=topics.findIndex(x=>x.id===selected),topic=topics[index];
  by('guideArticleTitle').textContent=title(topic);
  // Only audited bundled HTML is used; prompts/server responses never enter this view.
  by('guideArticleBody').innerHTML=content(topic);
  by('guideArticlePosition').textContent=t('Topic {index} of {total}',{index:index+1,total:topics.length});
  by('guideOpenPage').dataset.guideGo=topic.view;
  by('guideOpenPage').dataset.guideAnchor=topic.anchor||'';
  by('guidePrev').disabled=index===0;by('guideNext').disabled=index===topics.length-1;
  root.querySelectorAll('[data-guide-topic]').forEach(b=>{const on=b.dataset.guideTopic===selected;b.classList.toggle('active',on);b.setAttribute('aria-pressed',String(on));});
 }
 function localize(){
  language=window.ZI18n?.language||'en';
  by('guideIndex').replaceChildren(...topics.map((topic,i)=>{
   const b=document.createElement('button');b.type='button';b.className='guide-topic-link';b.dataset.guideTopic=topic.id;
   b.setAttribute('aria-controls','guideReader');b.title=title(topic);
   const n=document.createElement('span');n.className='guide-topic-number';n.textContent=String(i+1).padStart(2,'0');n.setAttribute('aria-hidden','true');
   const label=document.createElement('span');label.textContent=(topic.short_title||topic.title)[language];
   b.append(n,label);return b;
  }));
  renderArticle();filter();
 }
 function open(key){
  if(!topics.some(x=>x.id===key))return;
  selected=key;search.value='';filter();showView('guide');renderArticle();
  requestAnimationFrame(()=>{by('guideReader').scrollIntoView({block:'start',behavior:'auto'});by('guideArticleTitle').focus({preventScroll:true});});
 }
 search.addEventListener('input',filter);
 by('guideClear').onclick=()=>{search.value='';filter();search.focus()};
 search.addEventListener('keydown',e=>{
  if(e.key==='Escape'){search.value='';filter();}
  if(e.key==='Enter'){const first=[...by('guideIndex').querySelectorAll('button')].find(b=>!b.hidden);if(first)open(first.dataset.guideTopic);}
 });
 by('guideAllTopics').onclick=()=>{search.value='';filter();by('guideNavigator').scrollIntoView({block:'start',behavior:'auto'});by('guideIndex').querySelector('[data-guide-topic="'+selected+'"]').focus({preventScroll:true});};
 by('guidePrev').onclick=()=>{const i=topics.findIndex(x=>x.id===selected);if(i>0)open(topics[i-1].id);};
 by('guideNext').onclick=()=>{const i=topics.findIndex(x=>x.id===selected);if(i<topics.length-1)open(topics[i+1].id);};
 root.addEventListener('click',e=>{
  const b=e.target.closest('[data-guide-go]');if(!b)return;
  showView(b.dataset.guideGo);const anchor=b.dataset.guideAnchor&&by(b.dataset.guideAnchor);
  if(anchor)requestAnimationFrame(()=>{for(let node=anchor;node;node=node.parentElement)if(node.tagName==='DETAILS')node.open=true;anchor.style.scrollMarginTop='calc(var(--qwen-header-height) + 16px)';anchor.scrollIntoView({block:'start',behavior:'auto'});});
 });
 window.addEventListener('languagechange',localize);
 window.QwenGuide={open,filter,get selected(){return selected}};localize();
})();
