/* Modified for Zetalvx Image Lab Qwen 1.0.24: interface/localization only; Apache-2.0, retained NOTICE. */
/* .33 UI-only: local theme, image lightbox. Does not select assets or queue jobs. */
(()=>{
 'use strict';const by=id=>document.getElementById(id),t=s=>window.ZI18n?.t(s)||s;
 const themes=[['forest','Forest','#8acbb2'],['blue','Blue','#80b9ff'],['violet','Violet','#c4a5ff'],['amber','Amber','#e9ba6a'],['rose','Rose','#efadc0'],['graphite','Graphite','#c2cad6']];
 function drawThemes(){const box=by('themeOptions');if(!box)return;box.replaceChildren();for(const [id,label,color] of themes){const b=document.createElement('button');b.type='button';b.className='theme-choice';b.dataset.theme=id;b.setAttribute('aria-pressed',String(document.documentElement.dataset.uiTheme===id));const dot=document.createElement('i');dot.setAttribute('aria-hidden','true');dot.style.setProperty('--swatch',color);const text=document.createElement('span');text.textContent=t(label);b.append(dot,text);b.onclick=()=>{document.documentElement.dataset.uiTheme=id;try{localStorage.setItem('zetalvx.ui.theme',id)}catch(_){}drawThemes()};box.append(b)}}
 const viewer=by('imageViewer'),image=by('imageViewerImage'),close=by('imageViewerClose');let opener=null,overflow='';
 function show(img,trigger){if(!img)return;const src=img.currentSrc||img.src;if(!src)return;opener=trigger||document.activeElement;overflow=document.body.style.overflow;image.src=src;image.alt=img.alt||'';const host=img.closest('.asset-card'),title=host?.querySelector('.asset-info b')?.textContent;by('imageViewerCaption').textContent=title||img.alt||'';viewer.classList.remove('hidden');document.body.style.overflow='hidden';close.focus()}
 function hide(){if(viewer.classList.contains('hidden'))return;viewer.classList.add('hidden');image.removeAttribute('src');document.body.style.overflow=overflow;if(opener?.isConnected)opener.focus();opener=null}
 close?.addEventListener('click',hide);viewer?.addEventListener('click',e=>{if(e.target!==image&&!close.contains(e.target))hide()});
 document.addEventListener('keydown',e=>{if(viewer.classList.contains('hidden'))return;if(e.key==='Escape'){e.preventDefault();hide()}if(e.key==='Tab'){e.preventDefault();close.focus()}});
 const selector='.asset-card,.media-tool-preview,.frame-preview-stage';
 const icon='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M9 3H3v6M15 3h6v6M3 15v6h6M21 15v6h-6M3 3l6 6M21 3l-6 6M3 21l6-6M21 21l-6-6"/></svg>';
 function decorate(){document.querySelectorAll(selector).forEach(host=>{const img=host.querySelector('img');let b=host.querySelector(':scope > .image-zoom');if(!img||(img.getAttribute('src')||'')===''){b?.remove();return}host.classList.add('zoom-host');if(!b){b=document.createElement('button');b.type='button';b.className='image-zoom';b.innerHTML=icon;host.append(b)}b._image=img;b.title=t('Enlarge image');b.setAttribute('aria-label',t('Enlarge image'))})}
 document.addEventListener('click',e=>{const b=e.target.closest('.image-zoom');if(b){e.preventDefault();e.stopImmediatePropagation();show(b._image,b)}},true);
 let queued=false;const observer=new MutationObserver(()=>{if(queued)return;queued=true;requestAnimationFrame(()=>{queued=false;decorate()})});observer.observe(document.querySelector('.workspace'),{childList:true,subtree:true,attributes:true,attributeFilter:['src']});
 window.addEventListener('languagechange',()=>{drawThemes();decorate();close.title=t('Close image');close.setAttribute('aria-label',t('Close image'));viewer.setAttribute('aria-label',t('Image preview'))});
 drawThemes();decorate();window.CreatorImageViewer={show,hide};
})();
