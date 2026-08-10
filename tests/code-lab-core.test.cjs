const test = require('node:test');
const assert = require('node:assert/strict');
const core = require('../assets/code-lab-core.js');

test('standard library and NumPy snippets are browser-runnable', () => {
  assert.equal(core.analyzePython('print(2 + 3)').runnable, true);
  assert.equal(core.analyzePython('import numpy as np\nprint(np.arange(3))').runnable, true);
});

test('heavy framework snippets are routed to a cloud environment', () => {
  const torch = core.analyzePython('import torch\nprint(torch.ones(2))');
  assert.equal(torch.runnable, false);
  assert.equal(torch.kind, 'cloud');

  const inherited = core.analyzePython('return -F.logsigmoid(logits).mean()');
  assert.equal(inherited.runnable, false);
  assert.equal(inherited.kind, 'cloud');

  assert.equal(core.analyzePython('# compare with torch.nn\nprint(loss)  # still no dependency').runnable, true);
});

test('indented fragments and input calls fail closed with a reason', () => {
  assert.equal(core.analyzePython('    def backward(self):\n        pass').kind, 'fragment');
  assert.equal(core.analyzePython('name = input("name")').kind, 'input');
});

test('lab ids are stable and route-scoped', () => {
  assert.equal(core.makeLabId('/01-fundamentals/01-math', 2), '01-fundamentals/01-math::python-3');
  assert.equal(core.makeLabId('/', 0), 'home::python-1');
});

test('errors are compacted to the useful Python traceback', () => {
  const compact = core.compactError('PythonError: wrapper noise\nTraceback (most recent call last):\n  File "<exec>", line 2\nNameError: nope\n    at worker.js:2');
  assert.match(compact, /^Traceback/);
  assert.match(compact, /NameError: nope/);
  assert.doesNotMatch(compact, /worker\.js/);
});
