/* Zetalvx Image Lab 1.0.27: local end-user licence page, no network requests. */
(()=>{'use strict';
 const data=window.ZETALVX_LEGAL;
 const paths=['license-texts/LICENSE.txt','license-texts/NOTICE.txt','https://huggingface.co/Qwen/Qwen-Image-2.1#license','https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1','https://ffmpeg.org/legal.html'];
 function render(){
  const lang=window.ZI18n?.language||'en',d=data[lang];
  if(!d)throw Error('Missing bundled legal language: '+lang);
  document.title=d.title+' · Zetalvx Image Lab';
  document.getElementById('legalTitle').textContent=d.title;
  document.getElementById('legalIntro').textContent=d.intro;
  document.getElementById('legalBack').textContent=d.back;
  document.getElementById('legalLanguageLabel').textContent=d.language;
  const summary=document.getElementById('legalSummary');summary.replaceChildren();
  for(const [name,value] of d.summary){const box=document.createElement('div');box.className='legal-key';const small=document.createElement('span'),strong=document.createElement('strong');small.textContent=name;strong.textContent=value;box.append(small,strong);summary.append(box);}
  const container=document.getElementById('legalSections');
  const opened=new Set([...container.querySelectorAll('details[open]')].map(x=>x.id));const fresh=container.children.length===0;container.replaceChildren();
  for(const s of d.sections){const box=document.createElement('details');box.id='legal-'+s.id;box.className='legal-card';box.open=opened.has(box.id)||(fresh&&(s.id==='1'||s.id==='2'));
   const title=document.createElement('summary');title.textContent=s.title;const content=document.createElement('div');content.className='legal-body';const p=document.createElement('p');p.textContent=s.body;content.append(p);
   if(s.id==='8'){const links=document.createElement('nav');links.className='legal-links';for(let i=0;i<paths.length;i++){const a=document.createElement('a');a.href=paths[i];a.textContent=d.resources[i];if(/^https:/.test(paths[i])){a.target='_blank';a.rel='noopener noreferrer';}links.append(a);}content.append(links);}
   box.append(title,content);container.append(box);
  }
 }
 function init(){render();window.addEventListener('languagechange',render);}
 if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init,{once:true});else init();
})();
