import { defineConfig } from 'vite';

const proxy = { '/api': process.env.FL_API_TARGET || 'http://127.0.0.1:8080' };

export default defineConfig({
  server: { host: '127.0.0.1', port: 5173, strictPort: true, proxy },
  preview: { host: '127.0.0.1', port: 4173, strictPort: true, proxy },
});
