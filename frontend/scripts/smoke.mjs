import assert from 'node:assert/strict';
import { setTimeout as delay } from 'node:timers/promises';

const base = process.argv[2] || 'http://127.0.0.1:5173';
async function get(path, options = {}) {
  const response = await fetch(new URL(path, base), { signal: AbortSignal.timeout(10000), ...options });
  assert.ok(response.ok, `${path} returned ${response.status}`);
  return response;
}
async function json(path, options) {
  return (await get(path, options)).json();
}

assert.match(await (await get('/')).text(), /<title>Federated Hospitals/);
assert.equal((await json('/study-summary.json')).summary.length, 8);
const health = await json('/api/health');
for (const component of ['api', 'database', 'mlEngine']) assert.equal(health[component], 'UP');
const accepted = await json('/api/experiments', {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    algorithm: 'fedprox', mode: 'non-iid', seed: 42, rounds: 2, epochs: 1,
    batchSize: 32, learningRate: 0.01, mu: 0.1,
  }),
});
assert.equal(accepted.status, 'QUEUED');
const run = `/api/experiments/${accepted.runId}`;
const deadline = Date.now() + 120000;
let progress;
do {
  progress = await json(run);
  assert.notEqual(progress.status, 'FAILED', progress.errorCode);
  if (progress.status === 'COMPLETED') break;
  await delay(500);
} while (Date.now() < deadline);
assert.equal(progress.status, 'COMPLETED');
assert.equal(progress.roundCount, 2);
const metrics = await json(`${run}/metrics`);
assert.deepEqual(metrics.history.map(entry => entry.round), [1, 2]);
assert.equal(metrics.finalMetrics.test.samples, 114);
assert.ok(metrics.finalMetrics.communication.total_model_bytes > 0);
const jsonExport = await get(`${run}/export?format=json`);
assert.match(jsonExport.headers.get('content-disposition'), /attachment/);
assert.equal((await jsonExport.json()).experiment.runId, accepted.runId);
const csv = await get(`${run}/export?format=csv`);
assert.match(csv.headers.get('content-type'), /text\/csv/);
assert.match(await csv.text(), /test,2,114,/);
console.log(`Dashboard proxy smoke passed: ${accepted.runId}, 2 rounds, 114 test records, CSV and JSON exports.`);
