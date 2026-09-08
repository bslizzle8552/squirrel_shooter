/* Browser-only MJPEG AVI playback. Original file/frames stay unchanged on Pi. */
function cleanAviPlayer({url,fps}) {
 const canvas=document.getElementById('video'),context=canvas.getContext('2d');
 const play=document.getElementById('play'),pause=document.getElementById('pause'),message=document.getElementById('message');
 let bytes=null,frames=[],index=0,running=false,loading=false,epoch=0;
 function chunks(view,start,end,inMovie=false){
  const tag=p=>String.fromCharCode(...new Uint8Array(view.buffer,p,4));
  for(let p=start;p+8<=end;){const type=tag(p),size=view.getUint32(p+4,true),body=p+8,next=body+size;
   if(next>end)throw Error('Incomplete AVI chunk');
   if(type==='LIST'&&size>=4)chunks(view,body+4,next,inMovie||tag(body)==='movi');
   else if(inMovie&&/^\d\d[dD][bc]$/.test(type))frames.push([body,next]);
   p=next+(size%2);
  }
 }
 async function tick(run){
  if(!running||run!==epoch)return;
  const started=performance.now();
  try {const [start,end]=frames[index];const bitmap=await createImageBitmap(new Blob([bytes.subarray(start,end)],{type:'image/jpeg'}));if(!running||run!==epoch){bitmap.close();return;}context.drawImage(bitmap,0,0);bitmap.close();
   message.textContent=(index/fps).toFixed(1)+' / '+(frames.length/fps).toFixed(1)+' seconds';index++;
   if(index>=frames.length){running=false;index=0;play.disabled=false;pause.disabled=true;return;}
   if(running)setTimeout(()=>tick(run),Math.max(0,1000/fps-(performance.now()-started)));
  }catch(e){running=false;message.textContent=e.message+' — use Download original AVI.';play.disabled=false;pause.disabled=true;}
 }
 play.onclick=async()=>{if(loading||running)return;loading=true;play.disabled=true;
  try{if(!bytes){message.textContent='Loading original AVI…';const response=await fetch(url);if(!response.ok)throw Error('Clip unavailable');
    const data=await response.arrayBuffer();bytes=new Uint8Array(data);const view=new DataView(data);
    if(data.byteLength<12||String.fromCharCode(...bytes.subarray(0,4))!=='RIFF'||String.fromCharCode(...bytes.subarray(8,12))!=='AVI ')throw Error('Unsupported AVI');
    chunks(view,12,Math.min(data.byteLength,view.getUint32(4,true)+8));if(!frames.length)throw Error('No MJPEG frames');}
   running=true;pause.disabled=false;tick(++epoch);
  }catch(e){bytes=null;frames=[];message.textContent=e.message+' — use Download original AVI.';play.disabled=false;}finally{loading=false;}
 };
 pause.onclick=()=>{epoch++;running=false;pause.disabled=true;play.disabled=false;play.textContent='Resume / replay';};
 document.addEventListener('visibilitychange',()=>{if(document.hidden)pause.onclick();});
}
