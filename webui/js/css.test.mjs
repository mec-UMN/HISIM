import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const css = fs.readFileSync(new URL('../css/app.css', import.meta.url), 'utf8');

test('app.css has balanced braces (one stray "{" silently disables every rule after it)', () => {
  const body = css.replace(/\/\*[\s\S]*?\*\//g, '');
  let depth = 0;
  let line = 1;
  for (const ch of body) {
    if (ch === '\n') line += 1;
    if (ch === '{') depth += 1;
    if (ch === '}') depth -= 1;
    assert.ok(depth >= 0, `unexpected "}" near line ${line}`);
  }
  assert.equal(depth, 0, 'a rule is never closed');
});

