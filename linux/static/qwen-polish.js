/* Presentation only: current navigation, mobile sheet, keyboard, tabs and focus. */
(()=>{'use strict';
 const by=id=>document.getElementById(id),menu=by('mobileMoreMenu'),toggle=by('mobileMoreToggle');
 const mobile=matchMedia('(max-width:900px)');
 const inertNodes=[document.querySelector('.topbar'),document.querySelector('.sidebar'),document.querySelector('.mobile-dock'),by('mainContent')].filter(Boolean);
 let inertBefore=[],oldOverflow='',moreOpen=false,lastView=null;
 const visible=el=>!!el&&el.getClientRects().length>0&&!el.closest('[hidden],.hidden');
 const focusables=()=>[...menu.querySelectorAll('button,a[href],input,select,[tabindex="0"]')].filter(x=>!x.disabled&&visible(x));
 function updateNavigation(){
  const current=document.querySelector('section.view.active')?.id.replace(/^view-/,'')||'home';
  document.querySelectorAll('.nav[data-view]').forEach(b=>{if(b.dataset.view===current)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current')});
  const secondary=!['home','image','library'].includes(current);
  toggle.classList.toggle('active',secondary||moreOpen);
  if(secondary)toggle.setAttribute('aria-current','page');else toggle.removeAttribute('aria-current');
 }
 function closeMore(restore=true){
  menu.classList.add('hidden');toggle.setAttribute('aria-expanded','false');
  if(moreOpen){inertBefore.forEach(([el,value])=>{el.inert=value});inertBefore=[];document.body.style.overflow=oldOverflow;moreOpen=false;if(restore&&visible(toggle))toggle.focus({preventScroll:true});}
  updateNavigation();
 }
 function toggleMore(){
  if(moreOpen){closeMore();return}if(!mobile.matches)return;
  oldOverflow=document.body.style.overflow;document.body.style.overflow='hidden';
  menu.classList.remove('hidden');toggle.setAttribute('aria-expanded','true');moreOpen=true;
  inertBefore=inertNodes.map(el=>[el,el.inert]);inertNodes.forEach(el=>el.inert=true);
  updateNavigation();by('mobileMoreClose').focus({preventScroll:true});
 }
 menu.addEventListener('click',e=>{if(e.target===menu)closeMore()});
 document.addEventListener('keydown',e=>{
  if(!moreOpen)return;
  if(e.key==='Escape'){e.preventDefault();e.stopPropagation();closeMore();return}
  if(e.key==='Tab'){
   const list=focusables(),first=list[0],last=list[list.length-1];
   if(!first){e.preventDefault();menu.querySelector('.mobile-more-sheet').focus();return}
   if(e.shiftKey&&(document.activeElement===first||!menu.contains(document.activeElement))){e.preventDefault();last.focus()}
   else if(!e.shiftKey&&(document.activeElement===last||!menu.contains(document.activeElement))){e.preventDefault();first.focus()}
  }
 });
 function updateViewport(){
  if(!mobile.matches)closeMore(false);
  const vv=window.visualViewport,active=document.activeElement;
  const editing=active?.matches('textarea,input:not([type=checkbox]):not([type=radio]):not([type=file]):not([type=button]):not([type=submit]),[contenteditable=true]');
  const gap=vv?Math.max(window.innerHeight,document.documentElement.clientHeight)-vv.height:0;
  document.body.classList.toggle('qwen-keyboard-open',!!(mobile.matches&&editing&&vv&&Math.abs(vv.scale-1)<.05&&gap>150));
 }
 window.visualViewport?.addEventListener('resize',updateViewport);window.addEventListener('resize',updateViewport);
 document.addEventListener('focusin',updateViewport);document.addEventListener('focusout',()=>setTimeout(updateViewport,0));
 mobile.addEventListener?.('change',updateViewport);
 const header=document.querySelector('.topbar');
 if(window.ResizeObserver&&header)new ResizeObserver(()=>document.body.style.setProperty('--qwen-header-height',Math.ceil(header.getBoundingClientRect().height)+'px')).observe(header);
 window.addEventListener('creator-view-change',e=>{
  const name=e.detail?.view;updateNavigation();
  if(lastView&&name!==lastView){const workspace=document.querySelector('.workspace');workspace.scrollTop=0;if(mobile.matches)window.scrollTo({top:0,behavior:'auto'});const title=document.querySelector('#view-'+name+' h2,#view-'+name+' h1');if(title){title.tabIndex=-1;title.focus({preventScroll:true});}}
  lastView=name;updateViewport();
 });
 function tabState(){
  const tabs=[...document.querySelectorAll('[data-hub-tab]')];
  tabs[0]?.parentElement.setAttribute('role','tablist');
  tabs[0]?.parentElement.setAttribute('aria-label',window.ZI18n?.t('Qwen model sections')||'Qwen model sections');
  tabs.forEach(b=>{const key=b.dataset.hubTab,on=b.classList.contains('active');b.id=b.id||'qwen-tab-'+key;b.setAttribute('role','tab');b.setAttribute('aria-selected',String(on));b.tabIndex=on?0:-1;const panel=document.querySelector('[data-hub-panel="'+key+'"]');if(panel){panel.id=panel.id||'qwen-panel-'+key;panel.setAttribute('role','tabpanel');panel.setAttribute('aria-labelledby',b.id);b.setAttribute('aria-controls',panel.id)}});
 }
 const originalTab=qwenModelTab;
 qwenModelTab=function(name){originalTab(name);tabState()};
 document.querySelectorAll('[data-hub-tab]').forEach(tab=>tab.addEventListener('keydown',e=>{
  if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;
  const list=[...document.querySelectorAll('[data-hub-tab]')],i=list.indexOf(tab),rtl=document.documentElement.dir==='rtl';
  let n=e.key==='Home'?0:e.key==='End'?list.length-1:(i+((e.key==='ArrowRight')!==rtl?1:-1)+list.length)%list.length;
  e.preventDefault();list[n].click();list[n].focus();
 }));
 window.addEventListener('languagechange',()=>{tabState();updateNavigation()});
 window.QwenUI={closeMore,toggleMore,updateNavigation,updateViewport};
 tabState();updateNavigation();updateViewport();
})();
