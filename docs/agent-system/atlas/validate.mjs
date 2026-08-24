import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const atlasPath = fileURLToPath(new URL('../atlas.html', import.meta.url));
const html = readFileSync(atlasPath, 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((match) => match[1]);

if (scripts.length === 0) {
  throw new Error('atlas.html contains no inline script');
}

for (const [index, script] of scripts.entries()) {
  new Function(script);
  console.log(`inline script ${index + 1}: syntax OK`);
}
