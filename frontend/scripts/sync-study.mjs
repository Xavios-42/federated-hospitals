import { readFile, writeFile, mkdir } from 'node:fs/promises';

const source = process.argv[2] || new URL('../../results/stage3/study.json', import.meta.url);
const output = new URL('../public/', import.meta.url);
const report = JSON.parse(await readFile(source, 'utf8'));
const summary = {
  seeds: report.config.seeds,
  testSamples: report.test.samples,
  summary: report.summary.map(({ mode, algorithm, test }) => ({
    mode, algorithm,
    test: Object.fromEntries(['accuracy', 'f1'].map(metric => [
      metric, { mean: test[metric].mean, sd: test[metric].sd },
    ])),
  })),
};
await mkdir(output, { recursive: true });
await writeFile(new URL('study-summary.json', output), JSON.stringify(summary, null, 2) + '\n');
