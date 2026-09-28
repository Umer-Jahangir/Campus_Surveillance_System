const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync('src/templates/dashboard.html', 'utf8');
const code = html.slice(html.indexOf('let detailAnimation = null;'), html.indexOf('function closeDetail()'));
for (const [width, height] of [[1440,900],[900,1440]]) {
  const image = { naturalWidth:320, naturalHeight:256 };
  let drawn;
  const canvas = { width:0, height:0, getBoundingClientRect:()=>({width,height}), getContext:()=>({fillRect(){},drawImage(...args){drawn=args;}}) };
  const context = { detailStreamId:0, cancelAnimationFrame(){}, requestAnimationFrame(){return 1;},
    document:{getElementById(id){return id==='mjpeg-0'?image:id==='detail-box-canvas'?canvas:{classList:{add(){}}};}} };
  vm.createContext(context); vm.runInContext(code+'\npaintDetailFrame();',context);
  const scale = Math.min(width/320,height/256);
  assert.equal(drawn[0],image,'Must copy already-decoded grid pixels, not an independent video');
  assert.deepEqual(drawn.slice(1),[(width-320*scale)/2,(height-256*scale)/2,320*scale,256*scale]);
  assert.equal(canvas.width,width); assert.equal(canvas.height,height);
}
console.log('2 detail-rendering regression cases passed');
