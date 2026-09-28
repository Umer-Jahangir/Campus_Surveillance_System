// Replay recorded real offline metadata; this is UI plumbing, not an accuracy test.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync('src/templates/dashboard.html','utf8');
const data=JSON.parse(fs.readFileSync('audit/validation-20260918/focused/offline-scene-only.json','utf8')).last_frame;
let handler,timer;const text={};
const sandbox={S:{streams:[{}],running:true,lastFrameAt:{},streamStats:{},lastBhvr:{}},
  socket:{on:(name,fn)=>{handler=fn}},setInterval:fn=>{timer=fn},
  setText:(id,value)=>text[id]=value,detailStreamId:null,
  document:{getElementById:()=>null},Date,
  updateStreamStatsBar:()=>{},updateSidebarStats:()=>{},renderBehaviorStrip:()=>{}};
vm.createContext(sandbox);
vm.runInContext(html.slice(html.indexOf('socket.on("frame_meta"'),html.indexOf('function showSceneEvent(')),sandbox);
vm.runInContext(html.slice(html.indexOf('setInterval(() => {\n  let freshStreams'),html.indexOf('}, 1000);',html.indexOf('setInterval(() => {\n  let freshStreams'))+9),sandbox);
handler(data);sandbox.S.lastFrameAt[0]=1;timer();
assert.match(text['scene-0'],/Scene only · person inference disabled/);
assert.match(text['scene-0'],/Offline 189\/189 frames.*avg processed FPS.*source speed/);
assert.equal(text['cstate-0'],'Offline complete · last analyzed frame');
assert.equal(text['status-text'],'Offline analysis complete · showing last results');
handler({...data,source_kind:'live'});sandbox.S.lastFrameAt[0]=1;timer();
assert.match(text['scene-0'],/Stale \/ disconnected/);
assert.match(text['status-text'],/streams stale or stopped/);
console.log('Recorded offline completion/progress and stale live status remain distinct');
