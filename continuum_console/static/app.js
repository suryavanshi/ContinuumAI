const state = { catalog: null, runs: [], run: null, trace: null };
const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const api = async (path, options = {}) => {
  const response = await fetch(path, {headers:{'Content-Type':'application/json'}, ...options});
  const body = await response.json();
  if (response.status === 401 && path !== '/api/auth/login') showAuth();
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  return body;
};

async function bootstrap() {
  const auth = await api('/api/auth/status');
  if (auth.required && !auth.authenticated) { showAuth(); return; }
  $('#auth-gate').hidden = true;
  [state.catalog, state.runs] = await Promise.all([api('/api/catalog'), api('/api/runs').then(x => x.runs)]);
  populateForm();
  renderRunSelect();
  if (state.runs.length) await selectRun(state.runs[0].id);
}

function renderRunSelect() {
  $('#run-select').innerHTML = state.runs.map(run => `<option value="${escapeHtml(run.id)}">${escapeHtml(run.name)}</option>`).join('');
}

async function selectRun(id) {
  state.run = await api(`/api/runs/${encodeURIComponent(id)}`);
  state.trace = state.run.traces?.[0] || null;
  $('#run-select').value = id;
  render();
}

function render() {
  const run = state.run, cfg = run.config, metrics = run.metrics || [], last = metrics.at(-1) || {};
  $('#run-header').innerHTML = `<div><div class="run-title"><h2>${escapeHtml(run.name)}</h2><span class="pill method">${escapeHtml(cfg.algorithm)}</span><span class="status ${escapeHtml(run.status)}">${escapeHtml(run.status)}</span></div><div class="run-meta"><span>${escapeHtml(cfg.model)}</span><span>·</span><span>${escapeHtml(cfg.dataset)}</span><span>·</span><span>${escapeHtml(cfg.steps)} steps</span></div></div>`;
  const rewardStart = metrics[0]?.reward ?? 0, rewardEnd = last.reward ?? 0;
  $('#stat-grid').innerHTML = [
    ['Distillation loss', last.distillation_loss?.toFixed(3) ?? '—', metrics.length ? '↓ optimized' : 'waiting'],
    ['Mean reward', last.reward?.toFixed(2) ?? '—', metrics.length ? `${(rewardEnd-rewardStart >= 0 ? '+' : '')}${(rewardEnd-rewardStart).toFixed(2)}` : 'waiting'],
    ['Teacher KL', last.teacher_kl?.toFixed(3) ?? '—', 'token-level'],
    ['Trajectories', run.traces?.length ?? 0, `${cfg.train_rows} configured`],
  ].map(([label,value,delta]) => `<div class="stat"><small>${label}</small><b>${value}</b><span class="delta">${delta}</span></div>`).join('');
  renderChart(metrics);
  renderTraces(run.traces || []);
  renderTraceDetail();
  renderAdvanced(metrics);
  $('#log-output').textContent = (run.logs || []).join('\n');
  $('#config-output').textContent = JSON.stringify({config:run.config, command:run.command}, null, 2);
  const launchable=['draft','failed'].includes(run.status);
  $('#launch-run').hidden=!launchable;
  $('#launch-run').textContent='Launch on Modal';
}

function showAuth(){ $('#auth-gate').hidden=false; }

async function refreshRunUntilSettled(id) {
  const active = new Set(['queued','running']);
  for (let attempt=0; attempt<720; attempt++) {
    await new Promise(resolve=>setTimeout(resolve,5000));
    const run=await api(`/api/runs/${encodeURIComponent(id)}`);
    state.run=run; render();
    if(!active.has(run.status)) return;
  }
}

function renderChart(metrics) {
  const target = $('#loss-chart');
  if (!metrics.length) { target.innerHTML = '<p class="muted">Metrics appear here after ingestion.</p>'; return; }
  const width=660,height=190,pad={l:44,r:16,t:12,b:28}, values=metrics.map(m=>Number(m.distillation_loss));
  const min=Math.min(...values)*.88,max=Math.max(...values)*1.08, x=i=>pad.l+i*(width-pad.l-pad.r)/Math.max(1,values.length-1), y=v=>pad.t+(max-v)*(height-pad.t-pad.b)/(max-min || 1);
  const points=values.map((v,i)=>`${x(i)},${y(v)}`).join(' '), area=`${pad.l},${height-pad.b} ${points} ${x(values.length-1)},${height-pad.b}`;
  const grid=[0,.25,.5,.75,1].map(t=>{const yy=pad.t+t*(height-pad.t-pad.b), val=max-t*(max-min);return `<line class="grid-line" x1="${pad.l}" y1="${yy}" x2="${width-pad.r}" y2="${yy}"/><text class="axis-label" x="4" y="${yy+3}">${val.toFixed(2)}</text>`}).join('');
  target.innerHTML=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Distillation loss by step"><defs><linearGradient id="loss-gradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#4f46e5" stop-opacity=".22"/><stop offset="1" stop-color="#4f46e5" stop-opacity="0"/></linearGradient></defs>${grid}<polygon class="loss-area" points="${area}"/><polyline class="loss-path" points="${points}"/>${values.map((v,i)=>`<circle class="chart-dot" cx="${x(i)}" cy="${y(v)}" r="3"/>`).join('')}<text class="axis-label" x="${pad.l}" y="${height-7}">step 1</text><text class="axis-label" text-anchor="end" x="${width-pad.r}" y="${height-7}">step ${values.length}</text></svg>`;
}

function renderTraces(traces) {
  $('#trace-count').textContent = `${traces.length} sampled traces`;
  $('#trace-rows').innerHTML = traces.map(trace => `<tr data-id="${escapeHtml(trace.id)}" class="${state.trace?.id===trace.id?'selected':''}"><td><b>${escapeHtml(trace.id)}</b></td><td>${Number(trace.reward).toFixed(2)}</td><td>${Number(trace.kl).toFixed(3)}</td><td>${escapeHtml(trace.judge_selected)}</td><td class="arrow">→</td></tr>`).join('');
  document.querySelectorAll('#trace-rows tr').forEach(row => row.addEventListener('click', () => { state.trace=traces.find(t=>t.id===row.dataset.id); renderTraces(traces); renderTraceDetail(); }));
}

function tokenStyle(delta) {
  const amount=Math.min(.42, .06+Math.abs(delta)*.75);
  return delta>=0 ? `background:rgba(8,127,91,${amount})` : `background:rgba(194,65,93,${amount})`;
}

function renderTraceDetail() {
  const trace=state.trace;
  if (!trace) { $('#trace-title').textContent='No trace data'; $('#trace-detail').innerHTML='<p class="muted">Ingest a trace to inspect token preferences.</p>'; return; }
  $('#trace-title').textContent=trace.id;
  $('#trace-reward').className=`status ${trace.reward>0?'completed':'failed'}`;
  $('#trace-reward').textContent=`reward ${Number(trace.reward).toFixed(2)}`;
  const tokens=(trace.tokens||[]).map((token,index)=>`<span class="token ${token.selected?'selected':''}" style="${tokenStyle(token.delta)}" title="token ${index} · Δ ${Number(token.delta).toFixed(3)} · student ${token.student_logprob} · teacher ${token.teacher_logprob}">${escapeHtml(token.text)}</span>`).join('');
  $('#trace-detail').innerHTML=`<div class="detail-block"><div class="detail-label">USER PROMPT</div><div class="detail-text">${escapeHtml(trace.prompt)}</div></div><div class="detail-block"><div class="detail-label">PRIVILEGED TEACHER HINT</div><div class="detail-text hint">${escapeHtml(trace.hint)}</div></div><div class="detail-block"><div class="detail-label">SAMPLED RESPONSE</div><div class="detail-text">${escapeHtml(trace.response)}</div></div><div class="detail-block"><div class="detail-label">TOKEN PREFERENCE HEATMAP</div><div class="token-stream">${tokens}</div><div class="token-key"><span>Teacher prefers sampled token</span><span>Student over-preference</span></div></div><details class="detail-block"><summary class="detail-label">TEACHER PROMPT CONTEXT</summary><div class="detail-text">${escapeHtml(trace.teacher_prompt)}</div></details>`;
}

function renderAdvanced(metrics) {
  $('#advanced-table').innerHTML = metrics.length ? `<div class="advanced-grid table-wrap"><table><thead><tr><th>Step</th><th>Loss</th><th>Reward</th><th>Teacher KL</th></tr></thead><tbody>${metrics.map(m=>`<tr><td>${m.step}</td><td>${Number(m.distillation_loss).toFixed(4)}</td><td>${Number(m.reward).toFixed(3)}</td><td>${Number(m.teacher_kl).toFixed(4)}</td></tr>`).join('')}</tbody></table></div>` : '<p class="muted">No metrics ingested.</p>';
}

function populateForm() {
  $('#algorithm').innerHTML=state.catalog.algorithms.map(a=>`<option value="${a.id}">${escapeHtml(a.name)}</option>`).join('');
  $('#hint-placement').innerHTML=state.catalog.hint_placements.map(h=>`<option value="${h.id}">${escapeHtml(h.name)}</option>`).join('');
  updateModels();
}
function updateModels(){const alg=state.catalog.algorithms.find(a=>a.id===$('#algorithm').value)||state.catalog.algorithms[0];$('#model').innerHTML=alg.models.map(m=>`<option>${escapeHtml(m)}</option>`).join('');}

document.addEventListener('click', event => {
  const tab=event.target.closest('.tab');
  if(tab){document.querySelectorAll('.tab,.tab-panel').forEach(el=>el.classList.remove('active'));tab.classList.add('active');$(`#${tab.dataset.tab}`).classList.add('active');}
});
$('#run-select').addEventListener('change', e=>selectRun(e.target.value));
$('#algorithm').addEventListener('change', updateModels);
$('#new-run').addEventListener('click', ()=>$('#run-dialog').showModal());
['#close-dialog','#cancel-dialog'].forEach(selector=>$(selector).addEventListener('click',()=>$('#run-dialog').close()));
$('#copy-command').addEventListener('click', async()=>{await navigator.clipboard.writeText((state.run.command||[]).join(' '));$('#copy-command').textContent='Copied';setTimeout(()=>$('#copy-command').textContent='Copy launch command',1200)});
$('#launch-run').addEventListener('click', async()=>{
  const run=state.run;
  $('#launch-summary').textContent=`Run: ${run.name}\nModel: ${run.config.model}\nSteps: ${run.config.steps}\nTrain rows: ${run.config.train_rows}\nValidation rows: ${run.config.val_rows}\nCompute: ${run.config.smoke_gpu?'H100 ×2 smoke':'H200 ×5 standard'}`;
  $('#launch-dialog').showModal();
});
['#close-launch','#cancel-launch'].forEach(selector=>$(selector).addEventListener('click',()=>$('#launch-dialog').close()));
$('#confirm-launch').addEventListener('click', async()=>{
  const run=state.run;
  $('#launch-dialog').close();
  $('#launch-run').classList.add('launching'); $('#launch-run').textContent='Submitting…';
  try { state.run=await api(`/api/runs/${encodeURIComponent(run.id)}/launch`,{method:'POST',body:JSON.stringify({confirm:run.id})}); render(); refreshRunUntilSettled(run.id); }
  catch(error){ alert(error.message); $('#launch-run').classList.remove('launching'); $('#launch-run').textContent='Launch on Modal'; }
});
$('#run-form').addEventListener('submit', async event => {
  event.preventDefault(); $('#form-error').textContent='';
  const payload=Object.fromEntries(new FormData(event.currentTarget)); ['steps','train_rows','val_rows'].forEach(key=>payload[key]=Number(payload[key]));
  payload.smoke_gpu=payload.smoke_gpu==='true';
  try { const run=await api('/api/runs',{method:'POST',body:JSON.stringify(payload)}); state.runs.unshift(run); renderRunSelect(); $('#run-dialog').close(); await selectRun(run.id); }
  catch(error){ $('#form-error').textContent=error.message; }
});

$('#login-form').addEventListener('submit',async event=>{
  event.preventDefault(); $('#login-error').textContent='';
  const payload=Object.fromEntries(new FormData(event.currentTarget));
  try { await api('/api/auth/login',{method:'POST',body:JSON.stringify(payload)}); $('#auth-gate').hidden=true; await bootstrap(); }
  catch(error){ $('#login-error').textContent=error.message; }
});
$('#logout').addEventListener('click',async()=>{await api('/api/auth/logout',{method:'POST',body:'{}'});state.run=null;state.runs=[];showAuth();});

bootstrap().catch(error => { $('#login-error').textContent=`Unable to start console: ${error.message}`; showAuth(); });
