(function () {
  'use strict';

  var core = window.AICodeLabCore;
  var pageLabs = [];
  var worker = null;
  var activeRun = null;
  var sequence = 0;
  var RUNTIME_TIMEOUT = 120000;
  var EXECUTION_TIMEOUT = 20000;

  function workerUrl() {
    return new URL('assets/python-worker.mjs?v=interactive-lab-4', document.baseURI).href;
  }

  function destroyWorker(message, cancelled) {
    if (worker) worker.terminate();
    worker = null;
    if (activeRun) {
      window.clearTimeout(activeRun.timer);
      var error = new Error(message || '实验已停止。');
      error.cancelled = Boolean(cancelled);
      activeRun.reject(error);
      activeRun = null;
    }
  }

  function ensureWorker() {
    if (worker) return worker;
    worker = new Worker(workerUrl(), { type: 'module' });
    worker.addEventListener('message', function (event) {
      var data = event.data || {};
      if (!activeRun || data.id !== activeRun.id) return;

      if (data.type === 'status') {
        activeRun.onStatus(data.stage);
        window.clearTimeout(activeRun.timer);
        var isExecuting = data.stage === 'running' || data.stage === 'running-setup';
        activeRun.timer = window.setTimeout(function () {
          destroyWorker(isExecuting
            ? '运行超过 20 秒，实验台已自动急停。检查一下是不是写出了永动机。'
            : 'Python 工具箱加载超时，请检查网络后重试。');
        }, isExecuting ? EXECUTION_TIMEOUT : RUNTIME_TIMEOUT);
        return;
      }

      window.clearTimeout(activeRun.timer);
      var pending = activeRun;
      activeRun = null;
      if (data.type === 'result') pending.resolve(data);
      else pending.reject(new Error(data.error || 'Python 实验没有成功返回。'));
    });
    worker.addEventListener('error', function () {
      destroyWorker('Python 引擎加载失败。请检查网络连接，或刷新页面后重试。');
    });
    return worker;
  }

  function runPython(cells, onStatus) {
    if (activeRun) return Promise.reject(new Error('另一格代码正在运行，请等它落地或先按“停止”。'));
    var id = 'run-' + Date.now() + '-' + (++sequence);

    return new Promise(function (resolve, reject) {
      activeRun = {
        id: id,
        resolve: resolve,
        reject: reject,
        onStatus: onStatus,
        timer: window.setTimeout(function () {
          destroyWorker('Python 工具箱加载超时，请检查网络后重试。');
        }, RUNTIME_TIMEOUT)
      };
      ensureWorker().postMessage({ id: id, cells: cells, targetIndex: cells.length - 1 });
    });
  }

  function button(label, className) {
    var node = document.createElement('button');
    node.type = 'button';
    node.className = className;
    node.textContent = label;
    return node;
  }

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) return navigator.clipboard.writeText(text);
    var textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.setAttribute('readonly', '');
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    textarea.remove();
    return Promise.resolve();
  }

  function setStatus(lab, state, message) {
    lab.wrapper.dataset.state = state;
    lab.status.textContent = message;
  }

  function setBusy(lab, busy) {
    lab.editButton.disabled = busy;
    lab.resetButton.disabled = busy;
    lab.runButton.textContent = busy ? '■ 停止' : '▶ 运行到这里';
    lab.runButton.disabled = false;
    lab.runButton.dataset.busy = busy ? 'true' : 'false';
  }

  function clearOutput(lab) {
    lab.output.hidden = true;
    lab.output.innerHTML = '';
    setStatus(lab, 'idle', lab.analysis.runnable ? '等待开跑' : '云端关卡');
  }

  function outputBlock(label, value, type) {
    var block = document.createElement('div');
    block.className = 'code-lab__output-block code-lab__output-block--' + type;
    var heading = document.createElement('strong');
    heading.textContent = label;
    var pre = document.createElement('pre');
    pre.textContent = value;
    block.appendChild(heading);
    block.appendChild(pre);
    return block;
  }

  function renderResult(lab, data, duration) {
    lab.output.innerHTML = '';
    lab.output.hidden = false;

    if (data.stdout) lab.output.appendChild(outputBlock('OUTPUT', data.stdout, 'stdout'));
    if (data.stderr) lab.output.appendChild(outputBlock('NOTE', data.stderr, 'stderr'));
    if (data.result && data.result !== 'None') lab.output.appendChild(outputBlock('RESULT', data.result, 'result'));

    (data.images || []).forEach(function (encoded, index) {
      var figure = document.createElement('figure');
      figure.className = 'code-lab__figure';
      var image = document.createElement('img');
      image.src = 'data:image/png;base64,' + encoded;
      image.alt = 'Python 运行生成的图表 ' + (index + 1);
      figure.appendChild(image);
      lab.output.appendChild(figure);
    });

    if (!data.stdout && !data.stderr && !data.result && !(data.images || []).length) {
      var quiet = document.createElement('p');
      quiet.className = 'code-lab__quiet-success';
      quiet.textContent = '✓ 执行完成。这一格主要在定义函数或变量，安静不代表没干活。';
      lab.output.appendChild(quiet);
    }

    var meta = document.createElement('p');
    meta.className = 'code-lab__output-meta';
    meta.textContent = '浏览器沙箱 · ' + (duration / 1000).toFixed(2) + ' 秒 · 结果只留在你的设备';
    lab.output.appendChild(meta);
  }

  function renderError(lab, error) {
    lab.output.innerHTML = '';
    lab.output.hidden = false;
    lab.output.appendChild(outputBlock('BUG BOSS 出现了', core.compactError(error), 'error'));
    var hint = document.createElement('p');
    hint.className = 'code-lab__error-hint';
    hint.textContent = '别慌：读最后一行 → 找行号 → 改一处 → 再 Run。工程师的升级动画通常长这样。';
    lab.output.appendChild(hint);
  }

  function renderCancelled(lab, message) {
    lab.output.innerHTML = '';
    lab.output.hidden = false;
    var note = document.createElement('p');
    note.className = 'code-lab__cancelled';
    note.textContent = '■ ' + (message || '运行已停止。修改以后，随时可以重新开跑。');
    lab.output.appendChild(note);
  }

  function currentCells(lab) {
    return pageLabs.slice(0, lab.index + 1)
      .filter(function (item) { return item.analysis.runnable; })
      .map(function (item) { return item.editor.value; });
  }

  function execute(lab) {
    if (lab.runButton.dataset.busy === 'true') {
      destroyWorker('你按下了急停，Python 已被请出实验室。', true);
      return;
    }

    var cells = currentCells(lab);
    var startedAt = performance.now();
    lab.output.hidden = false;
    lab.output.innerHTML = '<p class="code-lab__loading">正在启动实验台…</p>';
    setBusy(lab, true);
    setStatus(lab, 'running', core.statusText('loading-runtime'));

    runPython(cells, function (stage) {
      setStatus(lab, 'running', core.statusText(stage));
      var loading = lab.output.querySelector('.code-lab__loading');
      if (loading) loading.textContent = core.statusText(stage);
    }).then(function (data) {
      var duration = performance.now() - startedAt;
      setBusy(lab, false);
      setStatus(lab, 'success', '实验通过 · 战利品已入袋');
      renderResult(lab, data, duration);
      document.dispatchEvent(new CustomEvent('ai-journal:code-success', {
        detail: {
          labId: lab.id,
          path: lab.path,
          changed: lab.editor.value !== lab.original,
          duration: Math.round(duration)
        }
      }));
    }).catch(function (error) {
      setBusy(lab, false);
      if (error.cancelled) {
        setStatus(lab, 'cancelled', '已急停，可以修改后重跑');
        renderCancelled(lab, error.message);
      } else {
        setStatus(lab, 'error', '运行失败 · Bug Boss 拦路');
        renderError(lab, error);
      }
    });
  }

  function toggleEditor(lab) {
    var editing = lab.editor.hidden;
    lab.editor.hidden = !editing;
    lab.pre.hidden = editing;
    lab.editButton.textContent = editing ? '完成编辑' : '编辑';
    lab.wrapper.classList.toggle('is-editing', editing);
    if (editing) {
      lab.editor.focus();
      lab.editor.setSelectionRange(lab.editor.value.length, lab.editor.value.length);
    } else {
      syncPreview(lab);
    }
  }

  function syncPreview(lab) {
    lab.code.textContent = lab.editor.value;
    if (window.Prism && window.Prism.highlightElement) window.Prism.highlightElement(lab.code);
  }

  function buildLab(pre, code, index, path) {
    var source = code.textContent.replace(/\n$/, '');
    var analysis = core.analyzePython(source);
    var wrapper = document.createElement('section');
    wrapper.className = 'code-lab';
    wrapper.dataset.labId = core.makeLabId(path, index);
    wrapper.dataset.state = analysis.runnable ? 'idle' : 'unavailable';

    var toolbar = document.createElement('div');
    toolbar.className = 'code-lab__toolbar';
    var identity = document.createElement('div');
    identity.className = 'code-lab__identity';
    identity.innerHTML = '<span class="code-lab__light" aria-hidden="true"></span><b>PY LAB ' + String(index + 1).padStart(2, '0') + '</b>';
    var status = document.createElement('span');
    status.className = 'code-lab__status';
    status.textContent = analysis.runnable ? '等待开跑' : '云端关卡';
    identity.appendChild(status);

    var actions = document.createElement('div');
    actions.className = 'code-lab__actions';
    var runButton = button('▶ 运行到这里', 'code-lab__run');
    var editButton = button('编辑', 'code-lab__button');
    var resetButton = button('重置', 'code-lab__button');
    var copyButton = button('复制', 'code-lab__button');
    actions.appendChild(runButton);
    actions.appendChild(editButton);
    actions.appendChild(resetButton);
    actions.appendChild(copyButton);
    toolbar.appendChild(identity);
    toolbar.appendChild(actions);

    var editor = document.createElement('textarea');
    editor.className = 'code-lab__editor';
    editor.value = source;
    editor.hidden = true;
    editor.spellcheck = false;
    editor.setAttribute('aria-label', '编辑 Python 实验 ' + (index + 1));
    editor.style.height = Math.min(560, Math.max(170, source.split('\n').length * 22 + 36)) + 'px';

    var output = document.createElement('div');
    output.className = 'code-lab__output';
    output.hidden = true;
    output.setAttribute('aria-live', 'polite');

    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.appendChild(toolbar);
    wrapper.appendChild(pre);
    wrapper.appendChild(editor);
    wrapper.appendChild(output);
    pre.classList.add('code-lab__source');
    var oldCopy = pre.querySelector('.copy-code-button');
    if (oldCopy) oldCopy.remove();

    var lab = {
      id: wrapper.dataset.labId,
      index: index,
      path: path,
      original: source,
      analysis: analysis,
      wrapper: wrapper,
      toolbar: toolbar,
      pre: pre,
      code: code,
      editor: editor,
      output: output,
      status: status,
      runButton: runButton,
      editButton: editButton,
      resetButton: resetButton
    };

    if (!analysis.runnable) {
      runButton.textContent = '☁ 云端运行';
      runButton.disabled = true;
      wrapper.classList.add('is-unavailable');
      var notice = document.createElement('p');
      notice.className = 'code-lab__notice';
      notice.textContent = analysis.reason;
      wrapper.insertBefore(notice, output);
    } else {
      runButton.addEventListener('click', function () { execute(lab); });
    }
    editButton.addEventListener('click', function () { toggleEditor(lab); });
    resetButton.addEventListener('click', function () {
      editor.value = source;
      syncPreview(lab);
      clearOutput(lab);
    });
    copyButton.addEventListener('click', function () {
      copyText(editor.value).then(function () {
        copyButton.textContent = '已复制 ✓';
        window.setTimeout(function () { copyButton.textContent = '复制'; }, 1400);
      });
    });
    editor.addEventListener('keydown', function (event) {
      if (event.key !== 'Tab') return;
      event.preventDefault();
      var start = editor.selectionStart;
      editor.value = editor.value.slice(0, start) + '    ' + editor.value.slice(editor.selectionEnd);
      editor.selectionStart = editor.selectionEnd = start + 4;
    });

    return lab;
  }

  function enhance(root, path) {
    if (!core || !root) return;
    if (activeRun) destroyWorker('页面已切换，上一格实验自动停止。', true);
    pageLabs = [];
    var selector = 'pre > code.language-python, pre > code.lang-python';
    root.querySelectorAll(selector).forEach(function (code, index) {
      if (code.closest('.code-lab')) return;
      pageLabs.push(buildLab(code.parentElement, code, index, path));
    });
    document.body.classList.toggle('has-code-labs', pageLabs.length > 0);
  }

  function scrollToFirst() {
    var first = document.querySelector('.code-lab');
    if (!first) {
      window.location.hash = '/hand-coding';
      return;
    }
    first.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
    first.classList.add('is-highlighted');
    window.setTimeout(function () { first.classList.remove('is-highlighted'); }, 1200);
  }

  window.AICodeLab = {
    enhance: enhance,
    scrollToFirst: scrollToFirst,
    stop: function () { destroyWorker('实验已停止。', true); }
  };
})();
