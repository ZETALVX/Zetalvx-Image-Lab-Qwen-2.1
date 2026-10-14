/* Theme preference is browser-local and applied before rendering, without tinting images. */
(()=>{const allowed=['forest','blue','violet','amber','rose','graphite'];let theme='blue';try{const x=localStorage.getItem('zetalvx.ui.theme');if(allowed.includes(x))theme=x;}catch(_){}document.documentElement.dataset.uiTheme=theme;})();
