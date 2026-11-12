import React, { useEffect, useState } from 'react';
import { Activity, ArrowUpRight, ChartNoAxesCombined, Download, FlaskConical } from 'lucide-react';
import { algorithms, bytes, experimentPath, isActive, modeLabel, percent, request } from './api';

function Status({ value }) {
  const labels = { QUEUED: 'Queued', RUNNING: 'Running', COMPLETED: 'Completed', FAILED: 'Failed' };
  return <span className={`status ${value?.toLowerCase() || 'idle'}`}>{labels[value] || 'Ready'}</span>;
}

function Curve({ history, metric, rounds }) {
  const values = history.map(entry => entry.validation[metric]);
  const max = metric === 'loss' ? Math.max(1, ...values) : 1;
  const left = 52, right = 610, top = 14, bottom = 173;
  const x = round => left + ((round - 1) / Math.max(1, rounds - 1)) * (right - left);
  const y = value => bottom - (value / max) * (bottom - top);
  const points = history.map((entry, index) => `${x(entry.round)},${y(values[index])}`).join(' ');
  const tickRounds = [...new Set([1, Math.ceil(rounds / 2), rounds])];
  return <svg className="curve" viewBox="0 0 630 204" role="img" aria-label={`Validation ${metric} by communication round`}>
    <title>Validation {metric} by communication round</title>
    {[0, 0.25, 0.5, 0.75, 1].map(fraction => <g key={fraction}>
      <line x1={left} x2={right} y1={y(max * fraction)} y2={y(max * fraction)} />
      <text x={left - 10} y={y(max * fraction) + 4} textAnchor="end">{metric === 'loss' ? (max * fraction).toFixed(2) : `${fraction * 100}%`}</text>
    </g>)}
    {tickRounds.map(round => <text key={round} x={x(round)} y="196" textAnchor="middle">{round}</text>)}
    {history.length > 1 && <polyline className="line" points={points} />}
    {history.map((entry, index) => <circle key={entry.round} cx={x(entry.round)} cy={y(values[index])} r="3.5">
      <title>Round {entry.round} · {metric === 'loss' ? values[index].toFixed(3) : percent(values[index])}</title>
    </circle>)}
  </svg>;
}

export function Monitor({ snapshot, selected, pollError }) {
  const [metric, setMetric] = useState('accuracy');
  const status = snapshot?.status;
  const report = snapshot?.report;
  const history = report?.history || [];
  const latest = history.at(-1)?.validation;
  const final = report?.finalMetrics;
  const rounds = status?.config.rounds || 0;
  const count = history.length;
  const roundBytes = history.reduce((total, entry) => total + entry.upload_model_bytes + entry.download_model_bytes, 0);
  const exports = selected && status?.status === 'COMPLETED' && final;
  return <section className="panel monitor" aria-labelledby="monitor-title">
    <div className="panel-heading"><span className="icon-tile"><Activity size={19} /></span>
      <div><h2 id="monitor-title">Training monitor</h2><p>{status ? `${algorithms[status.config.algorithm]} · ${modeLabel(status.config.mode)} · Seed ${status.config.seed}` : 'Track progress and evaluate the final model.'}</p></div>
      <Status value={status?.status} />
    </div>
    {pollError && <p className="notice error" role="alert">{pollError}</p>}
    {status?.status === 'FAILED' && <p className="notice error" role="alert">Training stopped ({status.errorCode}). Completed rounds remain visible.</p>}
    <div className="stats">
      <div><span>Completed rounds</span><strong>{rounds ? `${count} / ${rounds}` : '—'}</strong></div>
      <div><span>Validation accuracy</span><strong>{percent(latest?.accuracy)}</strong></div>
      <div><span>{final || !count ? 'Model transfer' : 'Round transfer'}</span>
        <strong>{bytes(final?.communication.total_model_bytes ?? (count ? roundBytes : undefined))}</strong></div>
    </div>
    {rounds > 0 && <progress value={count} max={rounds} aria-label="Completed training rounds" />}
    <div className="chart-box">
      <div className="chart-heading"><h3>Validation learning curve</h3>
        <label className="chart-select"><span className="sr-only">Validation metric</span>
          <select value={metric} onChange={event => setMetric(event.target.value)}>
            <option value="accuracy">Accuracy</option><option value="f1">F1 score</option><option value="loss">Loss</option>
          </select></label>
      </div>
      {count ? <Curve history={history} metric={metric} rounds={rounds} /> :
        <div className="empty-chart"><ChartNoAxesCombined size={27} /><strong>{selected ? 'Waiting for the first round' : 'No experiment selected'}</strong>
          <span>{selected ? 'Completed rounds will appear here.' : 'Start an experiment to see its validation metrics.'}</span></div>}
    </div>
    {final && <div className="final">
      <div className="chart-heading"><div><h3>Held-out test results</h3><p>{final.test.samples} records · Malignant is the positive class</p></div>
        {exports && <div className="export-actions">
          <a className="secondary" href={`${experimentPath(selected)}/export?format=csv`} download><Download size={15} /> CSV</a>
          <a className="secondary" href={`${experimentPath(selected)}/export?format=json`} download><Download size={15} /> JSON</a>
        </div>}
      </div>
      <div className="test-stats">{['accuracy', 'precision', 'recall', 'f1'].map(key => <div key={key}><span>{key === 'f1' ? 'F1 score' : key}</span><strong>{percent(final.test[key])}</strong></div>)}</div>
      <div className="confusion" aria-label="Confusion matrix">{Object.entries({ tp: 'True positive', tn: 'True negative', fp: 'False positive', fn: 'False negative' }).map(([key, label]) => <span key={key}>{label} <strong>{final.test.confusion_matrix[key]}</strong></span>)}</div>
    </div>}
  </section>;
}

export function Compare({ runs, openRun }) {
  const [items, setItems] = useState([]);
  const [study, setStudy] = useState(null);
  const [loading, setLoading] = useState(runs.length > 0);
  const [benchmarkError, setBenchmarkError] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    const options = { signal: controller.signal };
    let timer;
    async function refresh() {
      const result = await Promise.all(runs.map(async id => {
        try {
          const status = await request(experimentPath(id), options);
          const metrics = status.status === 'COMPLETED' ? await request(`${experimentPath(id)}/metrics`, options) : null;
          return { id, status, final: metrics?.finalMetrics };
        } catch (error) {
          return { id, missing: error.status === 404 };
        }
      }));
      if (!controller.signal.aborted) {
        setItems(result);
        setLoading(false);
        if (result.some(item => !item.missing && (!item.status || isActive(item.status.status)))) timer = setTimeout(refresh, 1800);
      }
    }
    refresh();
    request('/study-summary.json', options).then(setStudy).catch(() => {
      if (!controller.signal.aborted) setBenchmarkError(true);
    });
    return () => { controller.abort(); clearTimeout(timer); };
  }, [runs]);
  return <div className="comparison">
    <section className="panel">
      <div className="panel-heading"><span className="icon-tile"><FlaskConical size={19} /></span>
        <div><h2>Runs from this browser</h2><p>Final metrics use the holdout for the run’s seed.</p></div>
      </div>
      {loading ? <p className="notice">Loading runs…</p> : items.length ? <div className="table-wrap"><table>
        <thead><tr>{['Strategy', 'Partition', 'Seed', 'Status', 'Test accuracy', 'Test F1', 'Transfer', ''].map(label => <th key={label} scope="col">{label || <span className="sr-only">Open run</span>}</th>)}</tr></thead>
        <tbody>{items.map(({ id, status, final }) => <tr key={id}>
          <th scope="row">{status ? algorithms[status.config.algorithm] : id.slice(0, 8)}</th>
          <td>{status ? modeLabel(status.config.mode) : '—'}</td><td>{status?.config.seed ?? '—'}</td>
          <td>{status ? <Status value={status.status} /> : 'Unavailable'}</td>
          <td>{percent(final?.test.accuracy)}</td><td>{percent(final?.test.f1)}</td>
          <td>{bytes(final?.communication.total_model_bytes)}</td>
          <td><button className="secondary" onClick={() => openRun(id)} aria-label={`Open run ${id.slice(0, 8)}`}>Open <ArrowUpRight size={14} /></button></td>
        </tr>)}</tbody></table></div> : <p className="notice">Runs you start from this dashboard will appear here.</p>}
    </section>
    <section className="panel">
      <div className="panel-heading"><span className="icon-tile"><ChartNoAxesCombined size={19} /></span>
        <div><h2>Stage 3 benchmark</h2><p>{study ? `Mean ± sample SD · Seeds ${study.seeds.join(', ')} · Shared holdout of ${study.testSamples} records` : 'Measured IID and non-IID comparison.'}</p></div>
      </div>
      {study ? <div className="table-wrap"><table>
        <thead><tr>{['Strategy', 'IID accuracy', 'IID F1', 'Non-IID accuracy', 'Non-IID F1'].map(label => <th key={label} scope="col">{label}</th>)}</tr></thead>
        <tbody>{Object.entries(algorithms).map(([key, label]) => <tr key={key}><th scope="row">{label}</th>
          {['iid', 'non-iid'].flatMap(mode => {
            const stats = study.summary.find(row => row.mode === mode && row.algorithm === key).test;
            return ['accuracy', 'f1'].map(metric => <td key={`${mode}-${metric}`}>{percent(stats[metric].mean)} <small>± {(stats[metric].sd * 100).toFixed(1)} pp</small></td>);
          })}</tr>)}</tbody>
      </table></div> : <p className="notice">{benchmarkError ? 'The benchmark summary is unavailable.' : 'Loading benchmark…'}</p>}
    </section>
  </div>;
}
