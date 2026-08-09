# AI 手帐 · 大模型工程师冒险地图

<div class="hero-spread">
  <section class="hero-copy">
    <h2>把大模型，<br>学成你的主线任务。</h2>
    <p>不是收藏夹吃灰指南。16 个关卡，从梯度下降一路打到 Agent；每关都留下能讲、能跑、能写进简历的战利品。</p>
    <div class="hero-actions">
      <a class="journal-button" href="#/01-fundamentals/01-math" data-today-link><span data-continue-label>开始第 1 关</span> →</a>
      <a class="journal-button journal-button--ghost" href="#/?id=adventure-map">先看冒险地图</a>
    </div>
    <p class="margin-joke">别慌，这本手帐会帮你升级，但不会替你面试。那样违法物理定律。</p>
  </section>

  <section class="hero-dashboard" aria-label="学习进度">
    <div class="progress-note">
      <p class="progress-note__label">我的进度</p>
      <div class="progress-note__row">
        <strong data-home-level>Lv.1 张量见习生</strong>
        <span data-home-xp>0 / 1600 XP</span>
      </div>
      <div class="progress-note__bar" aria-hidden="true"><i data-home-progress-bar></i></div>
      <p class="progress-note__streak">▣ <span data-home-streak>连续学习 0 天</span> · 已完成 <span data-home-completed>0 / 16 关</span></p>
    </div>
    <div class="today-note">
      <p class="today-note__title"><span>今天先做这一页</span><span>→</span></p>
      <a class="today-note__task" href="#/01-fundamentals/01-math" data-today-link>
        <span><b data-today-code>01-1</b>　<span data-today-title>数学基础：线代 / 概率 / 微积分</span></span>
        <span aria-hidden="true">›</span>
      </a>
      <p class="today-note__objective">目标：<span data-today-objective>让梯度不再像玄学，先把模型的地基焊牢。</span></p>
    </div>
  </section>
</div>

<section class="home-section" id="adventure-map">
  <h2>你的冒险地图</h2>
  <p class="section-intro">路线不是“必须从左往右服刑”。有基础就跳关，没基础就从补给站出发；系统会把你的通关状态留在本地浏览器里。</p>

  <div class="adventure-map" role="region" aria-label="六阶段学习地图" tabindex="0">
    <ol class="stage-route">
      <li class="stage-node" data-stage="1">
        <a href="#/01-fundamentals/README">
          <span class="stage-node__icon">∇</span>
          <span class="stage-node__name">01<br>基础补给站</span>
          <span class="stage-node__status" data-stage-status>未解锁 0 / 3</span>
        </a>
      </li>
      <li class="stage-node" data-stage="2">
        <a href="#/02-deep-learning/README">
          <span class="stage-node__icon">QKᵀ</span>
          <span class="stage-node__name">02<br>Transformer 山谷</span>
          <span class="stage-node__status" data-stage-status>未解锁 0 / 3</span>
        </a>
      </li>
      <li class="stage-node" data-stage="3">
        <a href="#/03-pretraining/README">
          <span class="stage-node__icon">10¹²</span>
          <span class="stage-node__name">03<br>预训练矿场</span>
          <span class="stage-node__status" data-stage-status>未解锁 0 / 3</span>
        </a>
      </li>
      <li class="stage-node" data-stage="4">
        <a href="#/04-finetuning/README">
          <span class="stage-node__icon">ΔW</span>
          <span class="stage-node__name">04<br>对齐试炼场</span>
          <span class="stage-node__status" data-stage-status>未解锁 0 / 3</span>
        </a>
      </li>
      <li class="stage-node" data-stage="5">
        <a href="#/05-deployment/README">
          <span class="stage-node__icon">&lt;/&gt;</span>
          <span class="stage-node__name">05<br>部署发射台</span>
          <span class="stage-node__status" data-stage-status>未解锁 0 / 3</span>
        </a>
      </li>
      <li class="stage-node" data-stage="6">
        <a href="#/06-frontier/README">
          <span class="stage-node__icon">MCP</span>
          <span class="stage-node__name">06<br>前沿观测站</span>
          <span class="stage-node__status" data-stage-status>未解锁 0 / 1</span>
        </a>
      </li>
    </ol>
  </div>
</section>

<section class="home-section">
  <div class="daily-layout">
    <div>
      <h2>今天先做这一页</h2>
      <p class="section-intro">每天推进一页，比周日晚上对着 47 个收藏链接忏悔更有效。</p>
      <article class="daily-card">
        <span class="daily-card__flag" data-today-code>01-1</span>
        <h3 data-today-title>数学基础：线代 / 概率 / 微积分</h3>
        <p data-today-objective>让梯度不再像玄学，先把模型的地基焊牢。</p>
        <div class="daily-meta">
          <span>◷ 预计 25 分钟</span>
          <span>☑ 随堂小测</span>
          <span>⌁ 可运行代码</span>
        </div>
        <a class="journal-button" href="#/01-fundamentals/01-math" data-today-link>开始学习 →</a>
      </article>
    </div>
    <aside class="daily-aside">学不会？别跳关。<br>每一关的理解，<br>都是后面面试题的底气。<br><br>—— 来自未来不慌的你</aside>
  </div>
</section>

<section class="home-section">
  <h2>通关后，你会带走什么</h2>
  <p class="section-intro">参考 Microsoft AI-For-Beginners 的“课前摸底—概念—实验—测验”教学闭环，再加上大模型工程岗位真正要验收的四类产出。</p>
  <ul class="loot-strip">
    <li><b>系统掌握 LLM 全流程</b><span>从原理到工程落地，不再只会背名词。</span></li>
    <li><b>面试有底气</b><span>每关都配八股、手撕与追问题。</span></li>
    <li><b>能讲能跑的战利品</b><span>代码、实验和项目证据都能拿出来。</span></li>
    <li><b>简历亮点</b><span>把“学过”换成可复现、可验收的项目。</span></li>
  </ul>
  <p class="source-note">教学结构与部分手绘素材参考 <a href="https://github.com/microsoft/AI-For-Beginners" target="_blank" rel="noopener">Microsoft AI-For-Beginners</a>（MIT）；本教程面向大模型工程师求职场景重新组织与创作。</p>
</section>
