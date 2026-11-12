import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity, ArrowDownUp, ChartNoAxesCombined, CircleHelp, FlaskConical, GitBranch,
  Hospital, LayoutDashboard, Network, Play, Radio, ShieldCheck, Sparkles,
} from 'lucide-react';
import '@fontsource-variable/inter';
import { algorithms, defaults, experimentPath, isActive, payload, readRuns, request, saveRun } from './api';
import { Compare, Monitor } from './results';
import './styles.css';

function Topology({ counts, status }) {
  return <section className="topology" aria-labelledby="topology-title">
    <div className="topology-intro">
      <div className="eyebrow light"><Network size={15} /> TRAINING TOPOLOGY</div>
      <h2 id="topology-title">Three hospitals.<br /> One shared model.</h2>
      <p>Each hospital trains locally. The coordinator receives model updates and aggregate counts.</p>
    </div>
    <div className="nodes">
      <div className="coordinator"><Network size={17} /> Model coordinator</div>
      <div className="connections"><span /><span /><span /></div>
      <div className="hospital-nodes">
        {['A', 'B', 'C'].map(id => <div className="hospital" key={id}>
          <Hospital size={19} />
          <div><strong>Hospital {id}</strong><small>{counts ? `${counts[id]} training records` : isActive(status) ? 'Participating' : 'Local training'}</small></div>
        </div>)}
      </div>
    </div>
  </section>;
}

function NumberField({ label, name, form, change, min, max, step = 1 }) {
  return <label>{label}<input type="number" required name={name} value={form[name]} onChange={change} min={min} max={max} step={step} /></label>;
}

function Setup({ form, setForm, start, busy, disabled, error }) {
  const change = event => setForm({ ...form, [event.target.name]: event.target.value });
  const numeric = (label, name, min, max, step) =>
    <NumberField key={name} {...{ label, name, form, change, min, max, step }} />;
  return <section className="panel setup" aria-labelledby="setup-title">
    <div className="panel-heading"><span className="icon-tile"><FlaskConical size={19} /></span>
      <div><h2 id="setup-title">Experiment setup</h2><p>Model · PyTorch logistic regression</p></div>
    </div>
    <form onSubmit={event => { event.preventDefault(); start(form); }}>
      <label>Algorithm<select name="algorithm" value={form.algorithm} onChange={change}>
        {['fedavg', 'fedprox', 'fedavgm'].map(key => <option key={key} value={key}>{algorithms[key]}</option>)}
      </select></label>
      <label>Data partition<select name="mode" value={form.mode} onChange={change}>
        <option value="iid">IID · balanced classes</option><option value="non-iid">Non-IID · class skew</option>
      </select></label>
      <div className="field-grid">
        {numeric('Rounds', 'rounds', 1, 100)}
        {numeric('Seed', 'seed', 0, 4294967295)}
      </div>
      <details><summary>Training parameters</summary>
        <div className="field-grid advanced">
          {numeric('Local epochs', 'epochs', 1, 10)}
          {numeric('Batch size', 'batchSize', 1, 512)}
          {numeric('Learning rate', 'learningRate', 0.000000001, 1, 'any')}
          {form.algorithm === 'fedprox' && numeric('Proximal strength (μ)', 'mu', 0, 10, 'any')}
          {form.algorithm === 'fedavgm' && numeric('Momentum (β)', 'beta', 0, 0.999999999, 'any')}
          {form.algorithm === 'fedavgm' && numeric('Server learning rate', 'serverLearningRate', 0.000000001, 10, 'any')}
        </div>
      </details>
      <button className="primary launch" type="submit" disabled={disabled}><Play size={16} />{busy ? 'Submitting…' : 'Start experiment'}</button>
      {error && <p className="notice error" role="alert">{error}</p>}
    </form>
  </section>;
}

function App() {
  const [runs, setRuns] = useState(readRuns);
  const [selected, setSelected] = useState(() => readRuns()[0] ?? null);
  const [page, setPage] = useState('overview');
  const [form, setForm] = useState(defaults);
  const [health, setHealth] = useState('checking');
  const [busy, setBusy] = useState(false);
  const [submitError, setSubmitError] = useState('');
  const [pollError, setPollError] = useState('');
  const [snapshot, setSnapshot] = useState(null);

  useEffect(() => {
    const controller = new AbortController();
    let timer;
    async function probe() {
      try {
        const data = await request('/api/health', { signal: controller.signal });
        if (!controller.signal.aborted) setHealth(Object.values(data).every(value => value === 'UP') ? 'up' : 'down');
      } catch {
        if (!controller.signal.aborted) setHealth('down');
      }
      if (!controller.signal.aborted) timer = setTimeout(probe, 12000);
    }
    probe();
    return () => { controller.abort(); clearTimeout(timer); };
  }, []);

  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();
    let timer;
    async function poll() {
      const url = experimentPath(selected);
      const options = { signal: controller.signal };
      let refresh = true;
      try {
        const status = await request(url, options);
        const report = await request(`${url}/metrics`, options);
        if (controller.signal.aborted) return;
        setSnapshot({ status, report });
        setPollError('');
        refresh = isActive(status.status);
      } catch (error) {
        if (!controller.signal.aborted) setPollError(error.message);
        refresh = error.status !== 404;
      }
      if (refresh && !controller.signal.aborted) timer = setTimeout(poll, 1200);
    }
    poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [selected]);

  function openRun(id) {
    if (id !== selected) {
      setSnapshot(null);
      setPollError('');
      setSelected(id);
    }
    setPage('overview');
  }

  async function start(settings) {
    setBusy(true);
    setSubmitError('');
    try {
      const data = await request('/api/experiments', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload(settings)),
      });
      setRuns(previous => saveRun(previous, data.runId));
      openRun(data.runId);
    } catch (error) {
      setSubmitError(error.message);
      setPage('overview');
    } finally {
      setBusy(false);
    }
  }

  const canStart = health === 'up' && !busy;
  return <div className="shell">
    <aside className="sidebar">
      <div className="brand"><span><GitBranch size={22} /></span><div>Federated<br /><strong>Hospitals</strong></div></div>
      <nav aria-label="Dashboard">
        <button aria-current={page === 'overview' ? 'page' : undefined} onClick={() => setPage('overview')}><LayoutDashboard size={18} /> Overview</button>
        <button aria-current={page === 'compare' ? 'page' : undefined} onClick={() => setPage('compare')}><ChartNoAxesCombined size={18} /> Compare results</button>
      </nav>
      <div className="sidebar-footer"><ShieldCheck size={17} /><span>Academic simulation<br /><strong>Not for clinical diagnosis</strong></span></div>
    </aside>
    <main>
      <header className="header">
        <div className="eyebrow"><Activity size={15} /> FEDERATED LEARNING LAB</div>
        <span className={`connection ${health}`} role="status"><Radio size={15} />
          {health === 'up' ? 'API online' : health === 'down' ? 'API offline' : 'Checking API'}
        </span>
      </header>
      <div className="content">
        <div className="page-heading">
          <div><h1>{page === 'overview' ? 'Research dashboard' : 'Compare results'}</h1>
            <p>{page === 'overview' ? 'Run and monitor a shared breast cancer classification model.' : 'Inspect completed runs and the measured Stage 3 benchmark.'}</p></div>
          <button className="secondary demo" disabled={!canStart} onClick={() => { setForm(defaults); start(defaults); }}><Sparkles size={17} /> Run quick demo</button>
        </div>
        {health === 'down' && <p className="notice error" role="alert">Start the Spring Boot API and MySQL, then wait for the dashboard to reconnect.</p>}
        {page === 'overview' ? <>
          <Topology counts={snapshot?.report.finalMetrics?.hospital_training_samples} status={snapshot?.status.status} />
          <div className="dataset-note"><ShieldCheck size={16} /> Simulation using a public research dataset <span>·</span> 569 records <span>·</span> 30 features <span>·</span> 114 held out</div>
          <div className="workspace">
            <Setup form={form} setForm={setForm} start={start} busy={busy} disabled={!canStart} error={submitError} />
            <Monitor snapshot={snapshot} selected={selected} pollError={pollError} />
          </div>
        </> : <Compare runs={runs} openRun={openRun} />}
        <footer><CircleHelp size={15} /> UCI Breast Cancer Wisconsin (Diagnostic) <span><ArrowDownUp size={14} /> Model-transfer estimates exclude transport overhead.</span></footer>
      </div>
    </main>
  </div>;
}

createRoot(document.getElementById('root')).render(<App />);
