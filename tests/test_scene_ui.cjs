// Render an actual recorded backend event through the dashboard handler.
// This tests UI plumbing, not model accuracy.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync('src/templates/dashboard.html','utf8');
for(const match of html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)) new Function(match[1]);
const start=html.indexOf('function showSceneEvent('),end=html.indexOf("socket.on('scene_alert'",start);
const rows=[],values={};
const history={prepend(row){rows.unshift(row)},get children(){return rows}};
const sandbox={S:{totalDets:0},setText:(id,value)=>values[id]=value,toast:()=>{},
 document:{getElementById:()=>history,createElement:()=>({textContent:''})}};
vm.createContext(sandbox);vm.runInContext(html.slice(start,end),sandbox);
const record=JSON.parse(fs.readFileSync('audit/scene_demo/dashboard_actual.json','utf8'));
record.events.forEach(event=>sandbox.showSceneEvent(event,false));
assert.equal(values['sys-dets'],1);
assert.match(rows[1].textContent,/start at 5\.97s/);
assert.match(rows[0].textContent,/observation lost.*outcome unknown at 12\.53s/);
sandbox.showSceneEvent({stream_id:0,phase:'end',source_time:15,reason:'fight no longer detected in analyzed window'},false);
assert.match(rows[0].textContent,/fight no longer detected at 15\.00s/);
assert.equal(values['sys-dets'],1);
console.log('Real start/observation loss and fixture negative decision rendered distinctly; inline JavaScript parses');
