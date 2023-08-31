"use strict";(self.webpackChunkapp=self.webpackChunkapp||[]).push([[3287],{3287:(q,O,h)=>{h.r(O),h.d(O,{startInputShims:()=>X});var g=h(5861),P=h(8360),m=h(839),K=h(7484);h(4874),h(6225);const A=new WeakMap,I=(t,e,s,n=0,r=!1)=>{A.has(t)!==s&&(s?k(t,e,n,r):H(t,e))},k=(t,e,s,n=!1)=>{const r=e.parentNode,o=e.cloneNode(!1);o.classList.add("cloned-input"),o.tabIndex=-1,n&&(o.disabled=!0),r.appendChild(o),A.set(t,o);const d="rtl"===t.ownerDocument.dir?9999:-9999;t.style.pointerEvents="none",e.style.transform=`translate3d(${d}px,${s}px,0) scale(0)`},H=(t,e)=>{const s=A.get(t);s&&(A.delete(t),s.remove()),t.style.pointerEvents="",e.style.transform=""},C="input, textarea, [no-blur], [contenteditable]",U="$ionPaddingTimer",T=(t,e,s)=>{const n=t[U];n&&clearTimeout(n),e>0?t.style.setProperty("--keyboard-offset",`${e}px`):t[U]=setTimeout(()=>{t.style.setProperty("--keyboard-offset","0px"),s&&s()},120)},N=(t,e,s)=>{t.addEventListener("focusout",()=>{e&&T(e,0,s)},{once:!0})};let y=0;const p="data-ionic-skip-scroll-assist",V=(t,e,s,n,r,o,a,d=!1)=>{const i=o&&(void 0===a||a.mode===K.a.None),S=function(){var u=(0,g.Z)(function*(){e.hasAttribute(p)?e.removeAttribute(p):J(t,e,s,n,r,i,d)});return function(){return u.apply(this,arguments)}}();return t.addEventListener("focusin",S,!0),()=>{t.removeEventListener("focusin",S,!0)}},w=t=>{document.activeElement!==t&&(t.setAttribute(p,"true"),t.focus())},J=function(){var t=(0,g.Z)(function*(e,s,n,r,o,a,d=!1){if(!n&&!r)return;const i=((t,e,s)=>{var n;return((t,e,s,n)=>{const r=t.top,o=t.bottom,a=e.top,i=a+15,u=Math.min(e.bottom,n-s)-50-o,v=i-r,l=Math.round(u<0?-u:v>0?-v:0),_=Math.min(l,r-a),M=Math.abs(_)/.3;return{scrollAmount:_,scrollDuration:Math.min(400,Math.max(150,M)),scrollPadding:s,inputSafeY:4-(r-i)}})((null!==(n=t.closest("ion-item,[ion-item]"))&&void 0!==n?n:t).getBoundingClientRect(),e.getBoundingClientRect(),s,t.ownerDocument.defaultView.innerHeight)})(e,n||r,o);if(n&&Math.abs(i.scrollAmount)<4)return w(s),void(a&&null!==n&&(T(n,y),N(s,n,()=>y=0)));if(I(e,s,!0,i.inputSafeY,d),w(s),(0,m.r)(()=>e.click()),a&&n&&(y=i.scrollPadding,T(n,y)),typeof window<"u"){let S;const u=function(){var l=(0,g.Z)(function*(){void 0!==S&&clearTimeout(S),window.removeEventListener("ionKeyboardDidShow",v),window.removeEventListener("ionKeyboardDidShow",u),n&&(yield(0,P.c)(n,0,i.scrollAmount,i.scrollDuration)),I(e,s,!1,i.inputSafeY),w(s),a&&N(s,n,()=>y=0)});return function(){return l.apply(this,arguments)}}(),v=()=>{window.removeEventListener("ionKeyboardDidShow",v),window.addEventListener("ionKeyboardDidShow",u)};if(n){const l=yield(0,P.g)(n);if(i.scrollAmount>l.scrollHeight-l.clientHeight-l.scrollTop)return"password"===s.type?(i.scrollAmount+=50,window.addEventListener("ionKeyboardDidShow",v)):window.addEventListener("ionKeyboardDidShow",u),void(S=setTimeout(u,1e3))}u()}});return function(s,n,r,o,a,d){return t.apply(this,arguments)}}(),X=function(){var t=(0,g.Z)(function*(e,s){const n=document,r="ios"===s,o="android"===s,a=e.getNumber("keyboardHeight",290),d=e.getBoolean("scrollAssist",!0),i=e.getBoolean("hideCaretOnScroll",r),S=e.getBoolean("inputBlurring",r),u=e.getBoolean("scrollPadding",!0),v=Array.from(n.querySelectorAll("ion-input, ion-textarea")),l=new WeakMap,_=new WeakMap,j=yield K.K.getResizeMode(),M=function(){var f=(0,g.Z)(function*(c){yield new Promise(b=>(0,m.c)(c,b));const x=c.shadowRoot||c,D=x.querySelector("input")||x.querySelector("textarea"),L=(0,P.f)(c),W=L?null:c.closest("ion-footer");if(D){if(L&&i&&!l.has(c)){const b=((t,e,s)=>{if(!s||!e)return()=>{};const n=d=>{(t=>t===t.getRootNode().activeElement)(e)&&I(t,e,d)},r=()=>I(t,e,!1),o=()=>n(!0),a=()=>n(!1);return(0,m.a)(s,"ionScrollStart",o),(0,m.a)(s,"ionScrollEnd",a),e.addEventListener("blur",r),()=>{(0,m.b)(s,"ionScrollStart",o),(0,m.b)(s,"ionScrollEnd",a),e.removeEventListener("blur",r)}})(c,D,L);l.set(c,b)}if("date"!==D.type&&"datetime-local"!==D.type&&(L||W)&&d&&!_.has(c)){const b=V(c,D,L,W,a,u,j,o);_.set(c,b)}}});return function(x){return f.apply(this,arguments)}}();S&&(()=>{let t=!0,e=!1;const s=document;(0,m.a)(s,"ionScrollStart",()=>{e=!0}),s.addEventListener("focusin",()=>{t=!0},!0),s.addEventListener("touchend",a=>{if(e)return void(e=!1);const d=s.activeElement;if(!d||d.matches(C))return;const i=a.target;i!==d&&(i.matches(C)||i.closest(C)||(t=!1,setTimeout(()=>{t||d.blur()},50)))},!1)})();for(const f of v)M(f);n.addEventListener("ionInputDidLoad",f=>{M(f.detail)}),n.addEventListener("ionInputDidUnload",f=>{(f=>{if(i){const c=l.get(f);c&&c(),l.delete(f)}if(d){const c=_.get(f);c&&c(),_.delete(f)}})(f.detail)})});return function(s,n){return t.apply(this,arguments)}}()}}]);