export const algorithms = {
  fedavg: 'FedAvg',
  fedprox: 'FedProx',
  fedavgm: 'FedAvgM',
  centralized_logistic: 'Centralized logistic',
};

export const defaults = {
  algorithm: 'fedavgm', mode: 'iid', seed: 42, rounds: 20,
  epochs: 3, batchSize: 16, learningRate: 0.01, mu: 0.1, beta: 0.9, serverLearningRate: 1,
};

export function payload(form) {
  const { algorithm, mode, seed, rounds, epochs, batchSize, learningRate, mu, beta, serverLearningRate } = form;
  return {
    algorithm, mode, seed: Number(seed), rounds: Number(rounds),
    epochs: Number(epochs), batchSize: Number(batchSize), learningRate: Number(learningRate),
    ...(algorithm === 'fedprox' ? { mu: Number(mu) } : {}),
    ...(algorithm === 'fedavgm' ? { beta: Number(beta), serverLearningRate: Number(serverLearningRate) } : {}),
  };
}

export async function request(path, options) {
  const response = await fetch(path, { cache: 'no-store', ...options });
  const data = await response.json().catch(() => null);
  if (!response.ok || !data) {
    const error = new Error(data?.message || 'The local API is unavailable. Check that Spring Boot is running.');
    error.status = response.status;
    throw error;
  }
  return data;
}

export const experimentPath = id => `/api/experiments/${encodeURIComponent(id)}`;
export const isActive = status => ['QUEUED', 'RUNNING'].includes(status);
export const percent = value => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
export const bytes = value => Number.isFinite(value) ? `${value.toLocaleString()} B` : '—';
export const modeLabel = mode => mode === 'non-iid' ? 'Non-IID' : 'IID';

const storageKey = 'federated-hospitals-runs';
const uuid = /^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i;

export function readRuns() {
  try {
    const runs = JSON.parse(localStorage.getItem(storageKey));
    return Array.isArray(runs) ? runs.filter(id => typeof id === 'string' && uuid.test(id)).slice(0, 12) : [];
  } catch {
    return [];
  }
}

export function saveRun(ids, id) {
  const next = [id, ...ids.filter(value => value !== id)].slice(0, 12);
  try {
    localStorage.setItem(storageKey, JSON.stringify(next));
  } catch {
    // The dashboard can still track runs in this tab when browser storage is unavailable.
  }
  return next;
}
