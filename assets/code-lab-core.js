(function (root, factory) {
  'use strict';

  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.AICodeLabCore = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  var HEAVY_DEPENDENCIES = [
    'torch',
    'tensorflow',
    'transformers',
    'peft',
    'bitsandbytes',
    'mcp',
    'langchain',
    'vllm',
    'openai'
  ];

  var HEAVY_IDENTIFIER = /\b(?:(?:torch|tensorflow|transformers|peft|bitsandbytes|mcp|langchain|vllm|openai)\s*\.|F\.(?:logsigmoid|softmax|cross_entropy))\b/;

  function importedModules(code) {
    var modules = [];
    String(code || '').split('\n').forEach(function (line) {
      var match = line.match(/^\s*(?:from|import)\s+([a-zA-Z_][\w.]*)/);
      if (match) modules.push(match[1].split('.')[0]);
    });
    return modules;
  }

  function analyzePython(code) {
    var source = String(code || '').replace(/\r\n?/g, '\n');
    var executableSource = source.split('\n').map(function (line) {
      return line.replace(/(^|\s+)#.*$/, '');
    }).join('\n');
    var firstCodeLine = source.split('\n').find(function (line) {
      return line.trim() && line.trim().charAt(0) !== '#';
    }) || '';
    var dependencies = importedModules(source);
    var heavy = dependencies.find(function (name) {
      return HEAVY_DEPENDENCIES.indexOf(name) !== -1;
    });

    if (!source.trim()) {
      return { runnable: false, kind: 'empty', reason: '这格还是空白页，先写点 Python。', dependencies: dependencies };
    }
    if (/^\s+/.test(firstCodeLine)) {
      return { runnable: false, kind: 'fragment', reason: '这是嵌套代码片段，需要和上方类定义组合后运行。', dependencies: dependencies };
    }
    if (heavy || HEAVY_IDENTIFIER.test(executableSource)) {
      return {
        runnable: false,
        kind: 'cloud',
        reason: '这段依赖 ' + (heavy || '深度学习框架') + '，请复制到本地或 GPU Notebook 运行。',
        dependencies: dependencies
      };
    }
    if (/\binput\s*\(/.test(source)) {
      return { runnable: false, kind: 'input', reason: '当前浏览器实验台暂不接管 input()，请改成变量后再运行。', dependencies: dependencies };
    }

    return { runnable: true, kind: 'browser', reason: '', dependencies: dependencies };
  }

  function makeLabId(path, index) {
    var safePath = String(path || '/').replace(/^#?\/?/, '').replace(/[^a-zA-Z0-9/_-]+/g, '-');
    return (safePath || 'home') + '::python-' + (Number(index) + 1);
  }

  function compactError(error) {
    var text = String(error && error.message ? error.message : error || '未知错误');
    var traceback = text.indexOf('Traceback (most recent call last):');
    if (traceback >= 0) text = text.slice(traceback);
    var lines = text.split('\n');
    var userFrame = lines.findIndex(function (line) { return line.indexOf('File "<exec>"') !== -1; });
    if (userFrame > 0) lines = [lines[0]].concat(lines.slice(userFrame));
    return lines.join('\n')
      .replace(/^PythonError:\s*/i, '')
      .replace(/\n\s*at\s+[^\n]+(?=\n|$)/g, '')
      .trim()
      .slice(0, 6000);
  }

  function statusText(stage) {
    var labels = {
      'loading-runtime': '正在把 Python 搬进浏览器…首次会慢一点',
      'loading-packages': '正在按需装载科学计算工具箱…',
      'running-setup': '正在悄悄补跑上方代码，避免变量失忆…',
      running: '代码正在冒烟，结果马上出炉…',
      rendering: '正在把图表贴回手帐…'
    };
    return labels[stage] || '实验进行中…';
  }

  return {
    analyzePython: analyzePython,
    compactError: compactError,
    importedModules: importedModules,
    makeLabId: makeLabId,
    statusText: statusText
  };
});
