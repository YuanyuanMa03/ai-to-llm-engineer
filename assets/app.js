(function () {
  'use strict';

  var STORAGE = {
    completed: 'ai-journal-completed-v2',
    quests: 'ai-journal-quests-v2',
    streak: 'ai-journal-streak-v2'
  };

  var LESSONS = [
    { path: '/01-fundamentals/01-math', stage: 1, code: '01-1', title: '数学基础：线代 / 概率 / 微积分', objective: '让梯度不再像玄学，先把模型的地基焊牢。' },
    { path: '/01-fundamentals/02-python', stage: 1, code: '01-2', title: 'Python 与数据栈', objective: '用向量化摆脱 for 循环，把代码跑出工程味。' },
    { path: '/01-fundamentals/03-ml-basics', stage: 1, code: '01-3', title: '机器学习三大范式', objective: '分清监督、自监督和强化学习各自在忙什么。' },
    { path: '/02-deep-learning/01-nn', stage: 2, code: '02-1', title: '反向传播与计算图', objective: '从零看懂梯度如何沿计算图一路倒车。' },
    { path: '/02-deep-learning/02-cnn-rnn', stage: 2, code: '02-2', title: 'CNN 与 RNN', objective: '补齐 Transformer 之前的两位老前辈。' },
    { path: '/02-deep-learning/03-transformer', stage: 2, code: '02-3', title: 'Transformer 结构详解', objective: '手写 Attention，并讲清为什么要除以 √dₖ。' },
    { path: '/03-pretraining/01-tokenizer', stage: 3, code: '03-1', title: 'Tokenizer：BPE / SentencePiece', objective: '看看一句人话如何被切成模型能吃的 token。' },
    { path: '/03-pretraining/02-architecture', stage: 3, code: '03-2', title: 'LLM 架构：Decoder-only / MoE', objective: '拆开现代 LLM 的发动机舱。' },
    { path: '/03-pretraining/03-scaling-laws', stage: 3, code: '03-3', title: 'Scaling Laws 与数据配比', objective: '学会算力、参数和数据之间的“炼丹配方”。' },
    { path: '/04-finetuning/01-sft', stage: 4, code: '04-1', title: '监督微调 SFT', objective: '把博览群书的模型教成会听指令的同事。' },
    { path: '/04-finetuning/02-peft', stage: 4, code: '04-2', title: 'PEFT：LoRA / QLoRA', objective: '用更少显存驯服更大模型。' },
    { path: '/04-finetuning/03-alignment', stage: 4, code: '04-3', title: '对齐：RLHF / DPO', objective: '理解模型为什么会听话，以及它怎么钻奖励的空子。' },
    { path: '/05-deployment/01-inference', stage: 5, code: '05-1', title: '推理优化：KV Cache / 量化', objective: '让模型少吃显存、少等一会儿。' },
    { path: '/05-deployment/02-rag', stage: 5, code: '05-2', title: 'RAG 检索增强', objective: '让模型先翻资料再开口，降低一本正经地胡说。' },
    { path: '/05-deployment/03-agent', stage: 5, code: '05-3', title: 'Agent 与工具调用', objective: '从“会聊天”升级到“会动手干活”。' },
    { path: '/06-frontier/01-mcp', stage: 6, code: '06-1', title: 'MCP：标准化工具接入协议', objective: '给 Agent 的工具箱装上统一插口。' }
  ];

  var STAGES = [
    { id: 1, name: '基础补给站' },
    { id: 2, name: 'Transformer 山谷' },
    { id: 3, name: '预训练矿场' },
    { id: 4, name: '对齐试炼场' },
    { id: 5, name: '部署发射台' },
    { id: 6, name: '前沿观测站' }
  ];

  var currentPath = '/';
  var toastTimer = null;

  function readJSON(key, fallback) {
    try {
      var value = window.localStorage.getItem(key);
      return value ? JSON.parse(value) : fallback;
    } catch (error) {
      return fallback;
    }
  }

  function writeJSON(key, value) {
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      // Progress is a convenience. The tutorial still works when storage is blocked.
    }
  }

  function normalizePath(value) {
    var path = String(value || '/').split('?')[0].split('#')[0];
    path = path.replace(/\.md$/, '').replace(/\/index$/, '');
    if (path.charAt(0) !== '/') path = '/' + path;
    if (path === '/README' || path === '') return '/';
    return path.replace(/\/$/, '') || '/';
  }

  function getLesson(path) {
    return LESSONS.find(function (lesson) { return lesson.path === path; });
  }

  function getCompleted() {
    return readJSON(STORAGE.completed, []).filter(function (path) {
      return Boolean(getLesson(path));
    });
  }

  function getQuestState(path) {
    var quests = readJSON(STORAGE.quests, {});
    return quests[path] || { read: false, code: false, interview: false };
  }

  function setQuestState(path, state) {
    var quests = readJSON(STORAGE.quests, {});
    quests[path] = state;
    writeJSON(STORAGE.quests, quests);
  }

  function getPlayer(completed) {
    var count = completed.length;
    var xp = count * 100;
    var level = 1;
    var title = '张量见习生';

    if (count >= 4) { level = 2; title = '注意力驯服者'; }
    if (count >= 8) { level = 3; title = '对齐炼金师'; }
    if (count >= 12) { level = 4; title = 'Agent 工程师'; }
    if (count >= LESSONS.length) { level = 5; title = '全栈大模型勇者'; }

    return {
      count: count,
      xp: xp,
      level: level,
      title: title,
      percent: Math.round((count / LESSONS.length) * 100)
    };
  }

  function updateStreak() {
    var today = new Date();
    var key = [today.getFullYear(), String(today.getMonth() + 1).padStart(2, '0'), String(today.getDate()).padStart(2, '0')].join('-');
    var yesterday = new Date(today);
    yesterday.setDate(today.getDate() - 1);
    var previousKey = [yesterday.getFullYear(), String(yesterday.getMonth() + 1).padStart(2, '0'), String(yesterday.getDate()).padStart(2, '0')].join('-');
    var streak = readJSON(STORAGE.streak, { last: '', count: 0 });

    if (streak.last === key) return streak;
    streak = { last: key, count: streak.last === previousKey ? streak.count + 1 : 1 };
    writeJSON(STORAGE.streak, streak);
    return streak;
  }

  function getStreak() {
    return readJSON(STORAGE.streak, { last: '', count: 0 });
  }

  function setText(selector, value) {
    document.querySelectorAll(selector).forEach(function (node) {
      node.textContent = value;
    });
  }

  function updatePlayerUI() {
    var completed = getCompleted();
    var player = getPlayer(completed);
    var label = 'Lv.' + player.level + ' ' + player.title;
    setText('[data-player-level]', label);
    setText('[data-player-xp]', player.xp + ' / ' + (LESSONS.length * 100) + ' XP');
    document.querySelectorAll('[data-player-progress-bar]').forEach(function (bar) {
      bar.style.width = player.percent + '%';
    });
    setText('[data-home-completed]', player.count + ' / ' + LESSONS.length + ' 关');
  }

  function updateHomeUI() {
    var completed = getCompleted();
    var player = getPlayer(completed);
    var streak = getStreak();
    var allComplete = completed.length === LESSONS.length;
    var next = LESSONS.find(function (lesson) { return completed.indexOf(lesson.path) === -1; }) || LESSONS[LESSONS.length - 1];

    document.querySelectorAll('[data-today-link]').forEach(function (link) {
      link.setAttribute('href', allComplete ? 'quiz/' : '#' + next.path);
      if (allComplete) link.setAttribute('target', '_self');
      else link.removeAttribute('target');
    });
    setText('[data-today-code]', allComplete ? 'FINAL' : next.code);
    setText('[data-today-title]', allComplete ? '面试试炼场' : next.title);
    setText('[data-today-objective]', allComplete ? '用 31 道题把最后的知识漏洞揪出来。' : next.objective);
    setText('[data-continue-label]', allComplete ? '去最终试炼' : (completed.length ? '继续 ' + next.code : '开始第 1 关'));
    setText('[data-home-level]', 'Lv.' + player.level + ' ' + player.title);
    setText('[data-home-xp]', player.xp + ' / ' + (LESSONS.length * 100) + ' XP');
    setText('[data-home-streak]', '连续学习 ' + streak.count + ' 天');
    document.querySelectorAll('[data-home-progress-bar]').forEach(function (bar) {
      bar.style.width = player.percent + '%';
    });

    STAGES.forEach(function (stage) {
      var lessons = LESSONS.filter(function (lesson) { return lesson.stage === stage.id; });
      var done = lessons.filter(function (lesson) { return completed.indexOf(lesson.path) !== -1; }).length;
      var node = document.querySelector('[data-stage="' + stage.id + '"]');
      if (!node) return;
      var status = node.querySelector('[data-stage-status]');
      var isCurrent = done < lessons.length && (done > 0 || completed.length >= LESSONS.filter(function (lesson) { return lesson.stage < stage.id; }).length);
      node.classList.toggle('is-complete', done === lessons.length);
      node.classList.toggle('is-current', isCurrent);
      if (status) {
        status.textContent = done === lessons.length ? '已通关 ' + done + ' / ' + lessons.length : (isCurrent ? '当前关卡 ' + done + ' / ' + lessons.length : '未解锁 0 / ' + lessons.length);
      }
    });
  }

  function showToast(message) {
    var toast = document.getElementById('achievementToast');
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add('is-visible');
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(function () {
      toast.classList.remove('is-visible');
    }, 3200);
  }

  function celebrate() {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    var colors = ['#ef4a43', '#f7d866', '#16847d', '#1e5ba8'];
    for (var index = 0; index < 24; index += 1) {
      var piece = document.createElement('i');
      piece.className = 'confetti';
      piece.style.setProperty('--confetti-color', colors[index % colors.length]);
      piece.style.setProperty('--confetti-x', (Math.random() * 460 - 230) + 'px');
      piece.style.setProperty('--confetti-r', (Math.random() * 700 - 350) + 'deg');
      piece.style.marginLeft = (Math.random() * 100 - 50) + 'px';
      piece.style.animationDelay = (Math.random() * 160) + 'ms';
      document.body.appendChild(piece);
      window.setTimeout(function (node) { node.remove(); }, 1200, piece);
    }
  }

  function updateQuestRail(path) {
    var rail = document.getElementById('questRail');
    var lesson = getLesson(path);
    if (!rail || !lesson) return;

    var completed = getCompleted();
    var isComplete = completed.indexOf(path) !== -1;
    var state = getQuestState(path);
    if (isComplete) state = { read: true, code: true, interview: true };

    rail.hidden = false;
    var heading = rail.querySelector('h2');
    if (heading) heading.textContent = lesson.code + ' · ' + lesson.title;

    rail.querySelectorAll('[data-quest]').forEach(function (input) {
      input.checked = Boolean(state[input.dataset.quest]);
      input.disabled = isComplete;
    });

    var allDone = state.read && state.code && state.interview;
    var button = document.getElementById('completeLesson');
    var hint = rail.querySelector('[data-quest-hint]');
    var stamp = rail.querySelector('[data-lesson-stamp]');
    button.disabled = !allDone || isComplete;
    button.classList.toggle('is-complete', isComplete);
    button.textContent = isComplete ? '本关已通关 ✓' : '完成本关 · +100 XP';
    hint.textContent = isComplete ? '这页已经拿下。下一关正在假装不紧张。' : (allDone ? '检查完毕，可以盖章了。' : '勾完三项，通关按钮才营业。');
    stamp.classList.toggle('is-visible', isComplete);
  }

  function setupQuestEvents() {
    var rail = document.getElementById('questRail');
    var button = document.getElementById('completeLesson');
    if (!rail || !button || rail.dataset.bound === 'true') return;
    rail.dataset.bound = 'true';

    rail.addEventListener('change', function (event) {
      var input = event.target.closest('[data-quest]');
      if (!input || !getLesson(currentPath)) return;
      var state = getQuestState(currentPath);
      state[input.dataset.quest] = input.checked;
      setQuestState(currentPath, state);
      updateQuestRail(currentPath);
    });

    button.addEventListener('click', function () {
      var lesson = getLesson(currentPath);
      if (!lesson || button.disabled) return;
      var completed = getCompleted();
      if (completed.indexOf(currentPath) === -1) completed.push(currentPath);
      writeJSON(STORAGE.completed, completed);
      updateQuestRail(currentPath);
      updatePlayerUI();
      celebrate();
      showToast('通关盖章！+100 XP · “' + lesson.title + '” 已收入战利品');
    });
  }

  function addCopyButtons() {
    document.querySelectorAll('.markdown-section pre').forEach(function (pre) {
      if (pre.querySelector('.copy-code-button')) return;
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'copy-code-button';
      button.textContent = '复制代码';
      button.addEventListener('click', function () {
        var code = pre.querySelector('code');
        var text = code ? code.innerText : pre.innerText;
        if (!navigator.clipboard || !navigator.clipboard.writeText) {
          showToast('浏览器没借到剪贴板权限，请手动复制一下。');
          return;
        }
        navigator.clipboard.writeText(text).then(function () {
          button.textContent = '已抄好 ✓';
          window.setTimeout(function () { button.textContent = '复制代码'; }, 1600);
        });
      });
      pre.appendChild(button);
    });
  }

  function parkQuestRail() {
    var rail = document.getElementById('questRail');
    var toast = document.getElementById('achievementToast');
    if (rail && rail.parentNode !== document.body) {
      document.body.insertBefore(rail, toast || null);
    }
  }

  function lessonPagination(path) {
    var index = LESSONS.findIndex(function (lesson) { return lesson.path === path; });
    if (index < 0) return '';
    var previous = LESSONS[index - 1];
    var next = LESSONS[index + 1];
    var previousMarkup = previous
      ? '<a href="#' + previous.path + '"><small>← 上一关</small><b>' + previous.title + '</b></a>'
      : '<a href="#/README?id=adventure-map"><small>← 返回</small><b>冒险地图</b></a>';
    var nextMarkup = next
      ? '<a href="#' + next.path + '"><small>下一关 →</small><b>' + next.title + '</b></a>'
      : '<a href="quiz/"><small>最终试炼 →</small><b>去面试题库验收</b></a>';
    return '<nav class="lesson-pagination" aria-label="课程翻页">' + previousMarkup + nextMarkup + '</nav>';
  }

  function onRoute(route) {
    currentPath = normalizePath(route && (route.path || route.file));
    var lesson = getLesson(currentPath);
    var isHome = currentPath === '/';
    document.body.classList.toggle('is-home', isHome);
    document.body.classList.toggle('is-lesson', Boolean(lesson));

    var rail = document.getElementById('questRail');
    var article = document.querySelector('.markdown-section');
    if (lesson && rail && article) {
      article.appendChild(rail);
      updateStreak();
      updateQuestRail(currentPath);
    } else if (rail) {
      rail.hidden = true;
    }

    updatePlayerUI();
    if (isHome) updateHomeUI();
    addCopyButtons();
  }

  window.aiJournalPlugin = function (hook, vm) {
    hook.beforeEach(function (markdown) {
      // Docsify replaces the article on every route. Move the persistent task
      // rail out first so it can be mounted into the next article safely.
      parkQuestRail();
      return markdown;
    });

    hook.afterEach(function (html) {
      var path = normalizePath(vm.route.file || vm.route.path);
      var lesson = getLesson(path);
      var extra = lessonPagination(path);

      if (path !== '/') {
        var file = String(vm.route.file || '').replace(/^\//, '');
        if (file && file.indexOf('README') === -1) {
          var editUrl = 'https://github.com/YuanyuanMa03/ai-to-llm-engineer/blob/main/' + file;
          extra += '<hr><p class="edit-note">📝 抓到知识 bug？<a href="' + editUrl + '" target="_blank" rel="noopener">在 GitHub 上批注这一页</a></p>';
        }
      }

      return html + extra;
    });

    hook.doneEach(function () {
      onRoute(vm.route);
    });

    hook.ready(function () {
      updatePlayerUI();
    });
  };

  setupQuestEvents();
  updatePlayerUI();

  var progressButton = document.getElementById('playerProgress');
  if (progressButton) {
    progressButton.addEventListener('click', function () {
      window.location.hash = '/README?id=adventure-map';
    });
  }

  window.AIJournal = {
    lessons: LESSONS,
    refresh: function () {
      updatePlayerUI();
      updateHomeUI();
    }
  };
})();
