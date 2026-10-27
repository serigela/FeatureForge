"""Self-contained local operations dashboard."""

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FeatureForge · Operations</title>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;color:#e9f2f2;background:#0b1518}
*{box-sizing:border-box}body{margin:0}main{max-width:1100px;margin:auto;padding:48px 24px 80px}
.eyebrow{color:#66d6c4;font-size:12px;font-weight:800;letter-spacing:.16em;text-transform:uppercase}
h1{font-size:clamp(36px,6vw,64px);letter-spacing:-.06em;margin:14px 0 8px}p{color:#9eb3b4;line-height:1.5}
.lead{max-width:670px;font-size:17px;margin-bottom:36px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
.card,.panel{border:1px solid #264044;background:#112126;border-radius:18px;padding:22px}
.label{color:#91aaac;font-size:13px}.value{font-size:30px;font-weight:750;letter-spacing:-.04em;margin-top:14px}
.span2{grid-column:span 2}.panel{margin-top:16px}h2{font-size:19px;margin:0 0 16px}
.row{display:flex;justify-content:space-between;gap:20px;padding:12px 0;border-bottom:1px solid #294045;color:#afc5c5}
.row:last-child{border:0}.row strong{color:#e9f2f2}.bar{height:8px;background:#233b3f;border-radius:99px;overflow:hidden;margin-top:8px}
.fill{height:100%;background:#66d6c4;border-radius:99px}.alert{color:#ffb48e}.ok{color:#66d6c4}
footer{font-size:13px;color:#789294;margin-top:28px}@media(max-width:700px){.grid{grid-template-columns:1fr 1fr}.span2{grid-column:span 2}}
</style></head><body><main><div class="eyebrow">FeatureForge / Operations</div>
<h1>Model health at a glance.</h1><p class="lead">Live predictions, paired canary evaluation, processing time, feature drift, and delayed outcomes from the local transaction-risk pipeline.</p>
<section class="grid"><div class="card"><div class="label">Model version</div><div class="value" id="version">—</div></div>
<div class="card"><div class="label">Predictions</div><div class="value" id="predictions">—</div></div>
<div class="card"><div class="label">Flagged for review</div><div class="value" id="review">—</div></div>
<div class="card"><div class="label">Drift status</div><div class="value" id="status">—</div></div></section>
<section class="grid"><div class="panel span2"><h2>Feature stability</h2><div id="features">Waiting for predictions.</div></div>
<div class="panel span2"><h2>Observed outcomes</h2><div id="outcomes">Waiting for feedback.</div></div></section>
<section class="grid"><div class="panel span2"><h2>Event pipeline</h2><div id="queue">Waiting for events.</div></div>
<div class="panel span2"><h2>Online feature checks</h2><div id="parity">Waiting for snapshots.</div></div></section>
<section class="panel"><h2>Model rollout</h2><div id="rollout">Waiting for model status.</div></section>
<section class="panel"><h2>Prediction processing time</h2><div id="latency">Waiting for measurements.</div></section>
<footer>Demo data is synthetic. Paired results score both models on the same events; only one result is served. Timings exclude the storage write and network. Auto-refreshes every 10 seconds.</footer>
</main><script>
const escapeHTML=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function refresh(){try{const [health,metrics,monitor,parity,rollout,latency]=await Promise.all(['/health','/metrics','/monitor','/parity','/rollout','/latency'].map(p=>fetch(p).then(r=>r.json())));
document.getElementById('version').textContent=health.model_version;
const counts=metrics.predictions||{};document.getElementById('predictions').textContent=(counts.approve||0)+(counts.review||0);
document.getElementById('review').textContent=counts.review||0;
const s=document.getElementById('status');s.textContent=monitor.status.replace('_',' ');s.className='value '+(monitor.status==='alert'?'alert':'ok');
const fs=Object.entries(monitor.feature_psi||{}).sort((a,b)=>b[1]-a[1]);
document.getElementById('features').innerHTML=fs.length?fs.map(([name,psi])=>`<div class="row"><div>${escapeHTML(name)}<div class="bar"><div class="fill" style="width:${Math.min(100,psi*250)}%"></div></div></div><strong>${psi.toFixed(3)}</strong></div>`).join(''):'Waiting for predictions.';
const o=monitor.outcomes||{};document.getElementById('outcomes').innerHTML=`<div class="row"><span>Labeled predictions</span><strong>${o.labeled_count||0}</strong></div><div class="row"><span>Precision</span><strong>${o.precision==null?'—':(o.precision*100).toFixed(1)+'%'}</strong></div><div class="row"><span>Recall</span><strong>${o.recall==null?'—':(o.recall*100).toFixed(1)+'%'}</strong></div>`;
const q=metrics.queue||{};document.getElementById('queue').innerHTML=['pending','inflight','retry','dead','done'].map(name=>`<div class="row"><span>${name}</span><strong>${q[name]||0}</strong></div>`).join('');
document.getElementById('parity').innerHTML=`<div class="row"><span>Users checked</span><strong>${parity.checked_users}</strong></div><div class="row"><span>Parity mismatches</span><strong>${parity.mismatch_count}</strong></div><div class="row"><span>Stale snapshots</span><strong>${parity.stale_count}</strong></div>`;
const assignment=rollout.rollout;const versions=rollout.versions||{};
document.getElementById('rollout').innerHTML=`<div class="row"><span>Traffic split</span><strong>${assignment?`${assignment.percent}% canary`:'100% active'}</strong></div>`+
Object.entries(versions).map(([version,v])=>`<div class="row"><span>${escapeHTML(version)} ${assignment&&version===assignment.canary_version?'(canary)':'(active)'}</span><strong>${v.predictions} served · ${v.labeled} labeled</strong></div>`).join('')+
(rollout.paired?`<div class="row"><span>Paired events / labeled</span><strong>${rollout.paired.event_count} / ${rollout.paired.labeled_count}</strong></div><div class="row"><span>Decision disagreement</span><strong>${rollout.paired.disagreement_rate==null?'—':(rollout.paired.disagreement_rate*100).toFixed(1)+'%'}</strong></div>`+Object.entries(rollout.paired.versions).map(([version,v])=>`<div class="row"><span>${escapeHTML(version)} paired outcomes</span><strong>precision ${v.precision==null?'—':(v.precision*100).toFixed(1)+'%'} · recall ${v.recall==null?'—':(v.recall*100).toFixed(1)+'%'}</strong></div>`).join(''):'');
const timing=latency.all||{};const largest=timing.max_ms||1;
document.getElementById('latency').innerHTML=timing.count?['p50_ms','p95_ms','p99_ms'].map(name=>`<div class="row"><span>${name.replace('_ms','').toUpperCase()} · ${timing.count} measured<div class="bar"><div class="fill" style="width:${Math.min(100,100*timing[name]/largest)}%"></div></div></span><strong>${timing[name].toFixed(3)} ms</strong></div>`).join(''):'Waiting for predictions.';
}catch(e){document.getElementById('status').textContent='Unavailable'}}refresh();setInterval(refresh,10000);
</script></body></html>"""
