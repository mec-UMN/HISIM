import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

// Source-pattern tests cannot see a script that fails to parse (e.g. a merge that duplicates a block);
// compile every classic script the page loads.
const html = fs.readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const scripts = [...html.matchAll(/<script src="js\/([a-z0-9-]+\.js)/g)].map((m) => m[1]);

test('every classic UI script the page loads compiles', () => {
  assert.ok(scripts.length > 8);
  for (const file of scripts) {
    const code = fs.readFileSync(new URL(`./${file}`, import.meta.url), 'utf8');
    assert.doesNotThrow(() => new vm.Script(code, { filename: file }), `${file} does not compile`);
  }
});
