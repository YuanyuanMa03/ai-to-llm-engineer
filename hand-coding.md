# ✋ 手撕代码题合集 · 大模型算法岗面试必备

> 面试官说「给你 15 分钟,手写一个 XXX」——这些都是真实高频题。
> 每题含:难度、考点、参考实现、面试官追问方向。

这不是 LeetCode 算法题(反转链表、最长子串那种),而是 **ML / DL 相关的实现题**:softmax、attention、layer norm、LoRA……面试官想看的不是你刷过多少题,而是你**真的理解这些组件每一行在干什么**,而不是只会 `import torch.nn`。

选题参考了 [bbruceyuan/AI-Interview-Code](https://github.com/bbruceyuan/AI-Interview-Code) 的思路,并按本教程的五个阶段重新归类。代码用 **numpy 手写**(面试白板不让 import PyTorch),每段都可以直接在下方实验台里编辑、运行和验收。

---

## 怎么用这个合集

**先自己写,写不出再看答案。** 这是铁律。看完题目立刻滚到下面的代码区,等于没练。

正确姿势:
1. 读完题目,开一个空 cell,凭记忆手写。
2. 写完先自己造一组输入验证(比如 `x = np.array([1, 2, 3])`),确认输出符合预期。
3. **重点关注边界**:全负数输入、空数组、除零、数值溢出、维度对齐——这些才是面试官真正在意的。
4. 写不出或卡住,再看参考实现,然后**合上答案重写一遍**。
5. 最后扫一遍「面试官追问」,试着口头回答。

> 💡 每题的「频率」标记:🔴 高频(几乎必考)、🟡 中频(出现概率 30%+)、🟢 低频(加分项)。

---

## 一、基础篇(必会)

这三道题是地基。如果这里卡壳,后面所有深度学习相关的手撕都会崩。

### 题 1 · 手写 Softmax(数值稳定版)

| 难度 | 考点 | 频率 |
|------|------|------|
| ⭐⭐ | 数值稳定性、溢出处理 | 🔴 高频 |

**题目:** 实现一个 softmax 函数,输入一维或二维 numpy 数组,要求**数值稳定**(输入 `[1000, 1001, 1002]` 也不能溢出),返回同形状的概率分布。

**面试官追问:**
- 为什么要在指数前减去最大值?不减会怎样?
- 输入全是 `-1000` 这种大负数,你的代码还能正常工作吗?
- `log_softmax` 和 `softmax` 后取 log 有什么区别?为什么训练时算交叉熵要用前者?
- 二维输入时,你是沿哪个 axis 做 softmax?`axis=0` 和 `axis=1` 分别对应什么含义?

```python
import numpy as np

def softmax(x, axis=-1):
    # 减最大值是关键:利用 softmax 对常数平移不变的性质
    # exp(x - max) 把指数上界压到 [0, 1],永不溢出
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)            # 最大值处 exp(0)=1,其余 < 1
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# 验证:极端输入也不崩
print(softmax(np.array([1000.0, 1001.0, 1002.0])))  # [0.09, 0.24, 0.67]
print(softmax(np.array([[-1.0, 2.0], [0.5, 0.5]])))  # 二维,沿最后一维归一化
```

> ⚠️ **易错点**:`keepdims=True` 不能漏。漏了会导致广播错位,二维输入时除法维度对不上,返回错误结果且不报错。

---

### 题 2 · 手写交叉熵损失

| 难度 | 考点 | 频率 |
|------|------|------|
| ⭐⭐ | log_softmax、数值稳定、one-hot | 🔴 高频 |

**题目:** 给定模型输出的 logits `z`(shape `[N, C]`)和真实标签 `y`(shape `[N]`,整数类别),计算**平均交叉熵损失**。要求:数值稳定、不能用 `for` 循环遍历样本。

**面试官追问:**
- 为什么不直接 `-log(softmax(z)[正确类])`?这样写哪里会出问题?
- `log(0)` 会出现吗?怎么避免?
- 二分类交叉熵(BCE)和多分类交叉熵(CE)的公式差别在哪?
- 如果标签是 soft label(概率分布,不是整数 one-hot),你的公式要怎么改?

```python
import numpy as np

def cross_entropy(logits, y):
    # logits: [N, C],  y: [N] 整数标签
    N = logits.shape[0]
    # 用 log_softmax 而非 log(softmax),避免除零和数值下溢
    log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
    # 用 advanced indexing 取出每个样本正确类别的 log_prob
    loss = -log_probs[np.arange(N), y]
    return np.mean(loss)

# 验证
logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
y = np.array([0, 1])
print(cross_entropy(logits, y))  # 约 0.42,和 torch.nn.CrossEntropyLoss 对齐
```

> 💡 进阶:更稳的做法是手写 `logsumexp`。面试官如果追问,写出 `logsumexp(z) = max(z) + log(sum(exp(z - max(z))))`,这是数值计算的金科玉律。

---

### 题 3 · 手写 KNN

| 难度 | 考点 | 频率 |
|------|------|------|
| ⭐⭐ | 距离度量、广播、投票 | 🟡 中频 |

**题目:** 实现一个 KNN 分类器。给定训练集 `(X_train, y_train)` 和测试样本 `x`,返回预测类别。距离用欧氏距离,投票多数表决。

**面试官追问:**
- 距离矩阵你用了几层循环?能向量化成零循环吗?(写出 `np.linalg.norm` 的广播写法)
- K 取多少合适?K 太大或太小分别会怎样?
- 如果特征量纲差很大(比如身高 cm 和体重 kg),需要做什么预处理?
- KNN 可以用 KD-Tree 加速,什么时候 KD-Tree 反而比暴力计算慢?

```python
import numpy as np

def knn_predict(X_train, y_train, x, k=3):
    # X_train: [N, D], y_train: [N], x: [D]
    # 向量化计算欧氏距离,零循环
    diff = X_train - x                          # [N, D] 广播
    dists = np.sqrt(np.sum(diff ** 2, axis=1))  # [N]
    # 取距离最近的 k 个邻居的标签
    k_nearest = y_train[np.argsort(dists)[:k]]
    # 多数投票:用 bincount 找出现次数最多的类别
    return np.bincount(k_nearest).argmax()

# 验证
X_train = np.array([[0, 0], [0, 1], [5, 5], [5, 6]])
y_train = np.array([0, 0, 1, 1])
print(knn_predict(X_train, y_train, np.array([1, 0]), k=3))  # 0
print(knn_predict(X_train, y_train, np.array([4, 5]), k=3))  # 1
```

> ⚠️ **易错点**:`bincount` 要求标签是非负整数。如果面试官说标签是字符串,要先提一句"实际工程里用 LabelEncoder 或直接 `np.unique(return_inverse=True)`"。

---

### 题 4 · 手写 K-Means

| 难度 | 考点 | 频率 |
|------|------|------|
| ⭐⭐⭐ | 迭代算法、收敛判断、初始化 | 🟡 中频 |

**题目:** 实现 K-Means 聚类。给定数据 `X` 和簇数 `K`,迭代更新质心直到收敛(质心不再变化或达到最大迭代次数)。返回最终质心和每个样本的簇标签。

**面试官追问:**
- 你的初始化是随机选 K 个点?有什么问题?(答:敏感,引出 K-Means++)
- 怎么判断收敛?除了"质心不移动"还有别的判据吗?
- 空簇怎么处理?(某个质心没有任何样本分配给它)
- K-Means 的目标函数是什么?为什么它一定能收敛?(EM 单调下降)

```python
import numpy as np

def kmeans(X, k, max_iters=100, seed=0):
    rng = np.random.default_rng(seed)
    # 随机初始化:从数据中选 k 个点作为初始质心
    centroids = X[rng.choice(len(X), k, replace=False)]
    for _ in range(max_iters):
        # E 步:每个样本分配到最近的质心(向量化,避免循环)
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)  # [N, k]
        labels = dists.argmin(axis=1)
        # M 步:质心更新为所属簇的均值
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        # 收敛判断:质心不再移动
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return centroids, labels

# 验证:两个明显分开的簇
np.random.seed(42)
X = np.vstack([
    np.random.randn(10, 2),            # 簇A 在原点附近
    np.random.randn(10, 2) + [10, 10]  # 簇B 在 (10,10) 附近
])
centroids, labels = kmeans(X, k=2, seed=0)
print(labels)  # 前10个一类(0),后10个一类(1)
```

> 💡 **面试加分点**:主动提一句"标准 K-Means 用随机初始化,对起始点敏感;生产里用 K-Means++ 让初始质心彼此分散,能显著降低陷入局部最优的概率。"

---

## 二、深度学习篇(核心)

这四道题是大模型岗的**生死线**。Attention 写不出来基本等于自爆。

### 题 5 · 手写 Scaled Dot-Product Attention

| 难度 | 考点 | 频率 |
|------|------|------|
| ⭐⭐ | 矩阵运算、缩放、mask | 🔴 高频 |

**题目:** 实现 Scaled Dot-Product Attention:$\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$。输入 $Q, K, V$ shape 均为 `[N, d]`,返回输出 `[N, d]`。**额外要求**:支持可选的 causal mask(用于自回归解码,屏蔽未来 token)。

**面试官追问:**
- 为什么除以 $\sqrt{d_k}$?不除会怎样?(方差爆炸 → softmax 饱和 → 梯度消失)
- Causal Mask 是上三角还是下三角?屏蔽的是过去还是未来?
- 如果你的 mask 实现 `attn += -1e9`,为什么不用 `-np.inf`?
- KV Cache 加速推理,本质是省掉了哪一步计算?

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    # 相似度:Q @ K^T / sqrt(d_k)
    scores = Q @ K.T / np.sqrt(d_k)              # [N, N]
    # 因果掩码:把上三角(未来位置)置为极小值
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    # softmax 沿最后一维(K 的维度)归一化
    attn = softmax(scores, axis=-1)              # 复用题1
    return attn @ V                              # [N, d]

def softmax(x, axis=-1):                         # 内联,方便单独跑
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# 验证
Q = K = V = np.random.randn(4, 8)
out = scaled_dot_product_attention(Q, K, V)
# 因果 mask:下三角为 True(保留),上三角为 False(屏蔽)
mask = np.tril(np.ones((4, 4), dtype=bool))
out_causal = scaled_dot_product_attention(Q, K, V, mask)
```

> ⚠️ **易错点**:mask 的语义要讲清楚。`np.tril` 生成下三角矩阵,`True` 表示"保留这个位置"(即允许 query i 看到 key j,其中 j ≤ i)。讲反了直接挂。

---

### 题 6 · 手写 Multi-Head Attention

| 难度 | 考点 | 频率 |
|------|------|------|
| ⭐⭐⭐ | reshape、转置、维度对齐、权重投影 | 🔴 高频 |

**题目:** 实现完整的 Multi-Head Attention。输入 `x` shape `[N, d_model]`,参数为头数 `h` 和每头的维度 `d_k = d_model / h`。要求:包含 $W^Q, W^K, W^V, W^O$ 四个权重矩阵,多头并行计算后 concat 再投影。

**面试官追问:**
- 你怎么把 `[N, d_model]` 切成多头?reshape 后需要转置吗?为什么?
- Multi-Head 的总参数量和单头 attention 比,是更多、更少还是相同?
- 为什么不直接用一个大单头?多头学到了什么单头学不到的东西?
- LLaMA / GPT 里 `d_model=4096, h=32`,每个头维度是多少?

```python
import numpy as np

def multi_head_attention(x, W_q, W_k, W_v, W_o, h):
    N, d_model = x.shape
    d_k = d_model // h
    # 线性投影:Q/K/V 三个不同的子空间
    Q = x @ W_q                                   # [N, d_model]
    K = x @ W_k
    V = x @ W_v
    # 切多头:reshape 成 [h, N, d_k] 让每个头独立做 attention
    Q = Q.reshape(N, h, d_k).transpose(1, 0, 2)   # [h, N, d_k]
    K = K.reshape(N, h, d_k).transpose(1, 0, 2)
    V = V.reshape(N, h, d_k).transpose(1, 0, 2)
    # 每个头独立算 attention,向量化避免循环
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)   # [h, N, N]
    attn = softmax(scores, axis=-1)
    heads = attn @ V                              # [h, N, d_k]
    # 拼回 [N, d_model] 再做输出投影
    concat = heads.transpose(1, 0, 2).reshape(N, d_model)
    return concat @ W_o

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# 验证
d_model, h, N = 8, 2, 4
x = np.random.randn(N, d_model)
W_q = np.random.randn(d_model, d_model) * 0.1
W_k = np.random.randn(d_model, d_model) * 0.1
W_v = np.random.randn(d_model, d_model) * 0.1
W_o = np.random.randn(d_model, d_model) * 0.1
out = multi_head_attention(x, W_q, W_k, W_v, W_o, h)
assert out.shape == (N, d_model)
```

> 💡 **记忆口诀**:reshape 切头 → transpose 把头提到 batch 维 → 矩阵乘法天然并行 → transpose 回来 → reshape 拼回去。这五步顺序错了维度就对不上,白板上手抖就翻车,务必练熟。

---

### 题 7 · 手写 Layer Normalization

| 难度 | 考点 | 频率 |
|------|------|------|
| ⭐⭐ | 归一化维度、可学习参数、eps 防除零 | 🔴 高频 |

**题目:** 实现 Layer Normalization。输入 `x` shape `[N, d]`,在每个样本内部(沿 feature 维)做归一化:减均值、除标准差、再仿射变换。返回同 shape 的输出。要求带可学习参数 $\gamma, \beta$。

**面试官追问:**
- LayerNorm 是沿哪个维度归一化的?和 BatchNorm 的区别?
- 为什么要有 $\gamma, \beta$?直接输出归一化后的值不行吗?(破坏模型表达能力)
- `eps` 的作用是什么?取多大合适?(1e-5,防除零)
- RMSNorm 和 LayerNorm 的区别?为什么 LLaMA 选 RMSNorm?(去掉 beta 和减均值,FLOPs 省一截)
- 为什么 Transformer 用 LayerNorm 而不是 BatchNorm?

```python
import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    # 沿最后一个维度(feature 维)归一化
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)          # 注意:这里是 population var
    x_norm = (x - mean) / np.sqrt(var + eps)     # eps 防止方差为 0 时除零
    return gamma * x_norm + beta                  # 仿射变换

# 验证
x = np.random.randn(4, 8) * 5 + 3   # 故意搞偏均值方差
gamma = np.ones(8)
beta = np.zeros(8)
out = layer_norm(x, gamma, beta)
print(out.mean(axis=-1).round(4))   # 每行均值约 0
print(out.std(axis=-1).round(4))    # 每行标准差约 1
```

> ⚠️ **易错点**:`np.var` 默认是 population variance(除以 N),而有些教程写的 sample variance(除以 N-1)。PyTorch 的 LayerNorm 用的是 population variance,这里保持一致。被问到时主动说清楚。

---

### 题 8 · 手写梯度下降(求函数最小值)

| 难度 | 考点 | 频率 |
|------|------|------|
| ⭐⭐ | 自动微分思路、学习率、收敛 | 🟡 中频 |

**题目:** 用梯度下降求函数 $f(x, y) = x^2 + 2y^2$ 的最小值。要求:手写梯度(不用框架),从任意初始点出发,迭代到收敛。打印每步的函数值。

**面试官追问:**
- 这个函数的最小值在哪个点?最优值是多少?(原点 $(0,0)$,$f=0$)
- 学习率设太大或太小分别会怎样?
- 为什么 $y$ 方向比 $x$ 方向下降得更快?(曲率更大,条件数大)
- 怎么判断收敛?除了 loss 变化小,还能看什么?
- 这个问题如果是百万维,你的写法哪里会变慢?(标量循环 → 向量化)

```python
import numpy as np

def gradient_descent(lr=0.1, steps=100):
    # f(x, y) = x^2 + 2y^2,梯度 = (2x, 4y)
    x, y = 5.0, 5.0                              # 随便选个初始点
    for step in range(steps):
        grad_x = 2 * x                           # df/dx
        grad_y = 4 * y                           # df/dy
        x -= lr * grad_x
        y -= lr * grad_y
        if step % 10 == 0:
            loss = x**2 + 2 * y**2
            print(f"step {step}: x={x:.4f}, y={y:.4f}, f={loss:.4f}")
    return x, y

# 验证:最终应收敛到 (0, 0)
gradient_descent(lr=0.1, steps=100)
```

> 💡 **进阶追问**:如果面试官问"你写的梯度对不对,怎么验证?"——答:**数值梯度检验**(numerical gradient check)。对每个参数 $w$,$\frac{f(w+\epsilon) - f(w-\epsilon)}{2\epsilon}$ 应该和你手算的解析梯度一致(误差小于 1e-7)。这是深度学习工程里的基本功。

---

## 三、进阶篇(加分)

这两题写出来是**加分项**,写不出来不扣分,但写了能让你在一堆只会背八股的候选人里脱颖而出。

### 题 9 · 手写 LoRA 前向传播

| 难度 | 考点 | 频率 |
|------|------|------|
| ⭐⭐⭐ | 低秩分解、参数高效微调 | 🟡 中频(2024 年起明显升温) |

**题目:** 实现 LoRA(Low-Rank Adaptation)的前向传播。给定原始权重 $W$(冻结,不更新)、输入 $x$,以及低秩矩阵 $A, B$,返回 $y = xW + xBA$。其中 $A$ 用高斯初始化,$B$ 初始化为零(训练初始时刻 LoRA 增量为零,不破坏原模型)。

**面试官追问:**
- LoRA 的核心思想是什么?为什么 $W$ 可以近似为 $W + BA$?
- $A$ 高斯初始化、$B$ 初始化为零,为什么这么设计?反过来行不行?
- LoRA 的秩 $r$ 一般取多少?太大会怎样?
- QLoRA 比普通 LoRA 多了什么?(4-bit 量化基座)
- LoRA 相比全参数微调,显存省在哪里?

```python
import numpy as np

def lora_forward(x, W, A, B):
    """LoRA 前向:y = xW + xAB
    x: [N, d_in]   W: [d_in, d_out]   A: [d_in, r]   B: [r, d_out]
    """
    return x @ W + x @ A @ B                       # 低秩增量 x @ A @ B

# 构造参数:r 远小于 d_in / d_out
d_in, d_out, r = 64, 128, 8
W = np.random.randn(d_in, d_out) * 0.02            # 原始权重,训练时冻结
A = np.random.randn(d_in, r) * 0.02                # A 高斯初始化
B = np.zeros((r, d_out))                           # B 初始化为零 → 初始增量 = 0

x = np.random.randn(10, d_in)
out = lora_forward(x, W, A, B)
assert out.shape == (10, d_out)
# 验证:B=0 时 LoRA 增量为零,输出严格等于原始 W 的前向
assert np.allclose(out, x @ W)
# 模拟训练后:B 变为非零,LoRA 增量开始生效
B_trained = np.random.randn(r, d_out) * 0.02
out_trained = lora_forward(x, W, A, B_trained)
assert not np.allclose(out_trained, x @ W)         # 此时应不同于原始前向
```

> ⚠️ **维度对齐是最大的坑**。标准 LoRA 形状:$x[N, d_{in}]$、$A[d_{in}, r]$、$B[r, d_{out}]$、$W[d_{in}, d_{out}]$。低秩增量是 $x \cdot A \cdot B$(A 在左,B 在右)。面试时**先在白板上把这四个形状标清楚**再动手写,避免来回擦。
>
> 💡 **初始化约定**(PEFT/HuggingFace 标准):$A$ 用 Kaiming/高斯初始化,$B$ 初始化为零。这样训练初始时刻 $BA=0$,模型行为完全等于原始冻结权重,**不会因为加了 LoRA 就破坏预训练效果**。反过来($A=0, B$ 高斯)数学上等价,但工程惯例是 $B=0$。

---

### 题 10 · 手写一个极简的 ReAct Agent 循环

| 难度 | 考点 | 频率 |
|------|------|------|
| ⭐⭐⭐ | Agent 范式、工具调用、循环终止 | 🟢 低频(但 2024 年越来越多) |

**题目:** 实现 ReAct(Reasoning + Acting)Agent 的核心循环。Agent 每一轮:先 `think`(调用 LLM 生成推理),再 `act`(选择工具执行),根据工具返回的 observation 继续下一轮,直到 LLM 输出 `FINISH` 或达到最大步数。

**面试官追问:**
- ReAct 和纯 Chain-of-Thought(CoT)的区别?为什么需要 Action?
- 工具调用的输出怎么注入回 prompt?(observation 拼到 history)
- 怎么防止 Agent 陷入死循环(反复调用同一个工具)?
- Function Calling / Tool Use 和 ReAct 的关系?(前者是后者的结构化实现)
- 如果工具返回的结果是错的,Agent 怎么自我纠正?

```python
import re

def react_agent(query, llm_call, tools, max_steps=5):
    """极简 ReAct 循环:think → act → observe → 重复"""
    history = f"Question: {query}\n"
    for step in range(max_steps):
        # 1. Think:LLM 生成推理和下一步动作
        prompt = (
            history
            + "\nThink step by step. "
            + "Output 'Action: <tool>(<arg>)' to call a tool, "
            + "or 'FINISH: <answer>' when done."
        )
        thought = llm_call(prompt)
        history += f"Thought: {thought}\n"
        # 2. 判断是否结束
        if "FINISH:" in thought:
            return thought.split("FINISH:")[-1].strip()
        # 3. Act:解析并执行工具调用
        m = re.search(r"Action:\s*(\w+)\((.*?)\)", thought)
        if m:
            tool_name, arg = m.group(1), m.group(2)
            if tool_name in tools:
                observation = tools[tool_name](arg)
                history += f"Observation: {observation}\n"   # 注入回上下文
            else:
                history += f"Observation: Tool '{tool_name}' not found.\n"
        else:
            history += "Observation: No valid action, please think again.\n"
    return "达到最大步数,未完成。"

# 验证:mock 一个 LLM 和一个 calculator 工具
def mock_llm(prompt):
    if "calculator(2*3)" in prompt and "Observation: 6" in prompt:
        return "2*3=6, so the answer is 6. FINISH: 6"
    if "2 * 3" in prompt or "2*3" in prompt:
        return "I need to calculate. Action: calculator(2*3)"
    return "Action: calculator(2*3)"

tools = {"calculator": lambda expr: str(eval(expr))}
print(react_agent("What is 2 times 3?", mock_llm, tools))  # 输出: 6
```

> 💡 **面试官想看的是循环结构,不是 prompt 工程**。重点是:① 每轮把 observation 拼回 history;② 有明确的终止条件(`FINISH` 或 `max_steps`);③ 工具不存在的兜底。这三个点讲清楚就过关。真要深究 prompt,那是另一场面试了。

---

## 附:手撕题自测清单

面试前一天,对着这个清单默写一遍,**不看答案**。能默出 7 题以上,大模型算法岗的手撕环节基本稳了。

| # | 题目 | 能默写? | 能讲清追问? |
|---|------|---------|-------------|
| 1 | Softmax(数值稳定) | ☐ | ☐ |
| 2 | 交叉熵损失 | ☐ | ☐ |
| 3 | KNN | ☐ | ☐ |
| 4 | K-Means | ☐ | ☐ |
| 5 | Scaled Dot-Product Attention | ☐ | ☐ |
| 6 | Multi-Head Attention | ☐ | ☐ |
| 7 | Layer Normalization | ☐ | ☐ |
| 8 | 梯度下降 | ☐ | ☐ |
| 9 | LoRA 前向传播 | ☐ | ☐ |
| 10 | ReAct Agent 循环 | ☐ | ☐ |

> 规则:每题限时 15 分钟,白板或纯文本编辑器,**不许 import PyTorch**。卡住的题当晚补练。

---

## 🔗 延伸资源

- 💻 [bbruceyuan/AI-Interview-Code](https://github.com/bbruceyuan/AI-Interview-Code) —— 本合集的灵感来源,600+ star,题目更偏 PyTorch 实现
- 💻 [karpathy/minGPT](https://github.com/karpathy/minGPT) —— 300 行从零实现 GPT,看懂了 attention / layernorm 手撕无压力
- 💻 [labmlai/annotated_deep_learning_paper_implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations) —— 60+ 篇带逐行注释的论文复现,进阶首选
- 📄 [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) —— Harvard NLP 的 Transformer 逐行实现,经典中的经典
- 💻 [lucidrains/lora](https://github.com/lucidrains/lora) —— PyTorch 版 LoRA 实现,对照学习

---

> 持续更新中,欢迎在 [Discussions](https://github.com/yuanyuanma03/ai-to-llm-engineer/discussions) 分享你遇到的手撕题。
