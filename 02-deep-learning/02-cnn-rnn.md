# 02 · CNN 与 RNN:被 Transformer 取代的前辈

> 一句话:**学完这节,你能回答「为什么 Transformer 能同时取代 CNN 和 RNN?它解决了它们各自的什么根本问题」这个面试题。**

## 🤔 课前小测
先别急着学,花 30 秒回答下面 3 题(答案在文末):
1. CNN 处理图像时,卷积核做了一件什么事,让它比全连接层高效得多?
2. 写出 RNN 单步隐藏状态更新的公式。隐藏状态 $h_t$ 在数学上是什么?
3. 直觉判断:RNN 处理一个 1000 词的句子,第 1000 个词的预测,几乎完全感受不到第 1 个词的影响。这是为什么?

---

## 这节解决什么问题

在 Transformer(2017)出现之前,深度学习有两个"半壁江山":**CNN 做图像**,**RNN 做序列(文本/语音)**。它们各自统治了一个领域将近十年。但今天,大模型里你几乎看不到纯 CNN 或纯 RNN 了 —— Transformer 几乎把它们的工作都接走了。

为什么要学两个"过气"的东西?两个原因:
1. **面试高频**:"为什么 RNN 会有梯度消失""CNN 卷积的本质是什么""为什么 Transformer 取代了它们"是常考题,答不上来直接出局。
2. **理解 Transformer 必须有对照**。Transformer 的设计动机,正是为了解决 RNN 不能并行 + 长程依赖衰减的问题。没见过 RNN,你就理解不了 Attention 是在替代什么。

## 核心概念 ★

### 一、CNN:局部特征提取器

#### 1. 卷积的核心 = 局部连接 + 参数共享

全连接层看一张 $224\times 224$ 的图,要把 $224\times 224\times 3\approx15$ 万个像素展平,每个神经元都连到所有像素 → 单层就 150 亿参数,根本训不动。

卷积层的做法:**一个 $3\times3$ 的小窗口(卷积核,kernel)在图上滑动**,每个位置只看局部 $3\times3\times\text{通道数}$ 个像素,而且**滑动过程中核的参数不变(参数共享)**。一个 $3\times3\times3$ 的核只有 27 个权重。

这种设计对应两个直觉:
- **局部性**:图像里一个像素的特征,主要由它附近的像素决定,不需要看全图。
- **平移不变性**:一只猫在图的左上角和右下角,长得一样,应该用同一组特征检测器。

#### 2. 卷积输出尺寸公式

输入尺寸 $W_{\text{in}}$、卷积核大小 $K$、步长 $S$、零填充 $P$,输出尺寸:

$$W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} - K + 2P}{S} \right\rfloor + 1$$

例:$W_{\text{in}}=32, K=3, S=1, P=1 \Rightarrow W_{\text{out}}=32$(尺寸不变);$S=2\Rightarrow W_{\text{out}}=16$(下采样一半)。

#### 3. 池化与感受野

- **池化(Pooling)**:把局部窗口(如 $2\times2$)内取 max 或 average,降采样尺寸,带来一定的平移不变性。
- **感受野(Receptive Field)**:输出层一个神经元,对应原图上多大的区域。**堆叠 $n$ 层 $3\times3$ 卷积**,感受野 $= 1 + \sum_{i}(k_i-1)$。3 层 $3\times3$ 感受野是 $7\times7$,且参数量比单层 $7\times7$ 少($3\times3^2\times3=27 < 7^2=49$)。这就是 VGG、ResNet 用小核堆深的根本原因。

### 二、RNN:序列的"记忆"

#### 1. 隐藏状态更新

处理序列 $x_1, x_2, \dots, x_t$,RNN 维护一个隐藏状态 $h_t$,每一步:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$

$h_t$ 在数学上是**截至第 $t$ 步、所有历史信息的压缩**。输出可以接个分类头 $y_t = \text{softmax}(W_{hy}h_t)$。

直觉:RNN 像一个"一边读一边记笔记"的人,每读一个新词,就把旧笔记和新词融合一下,更新成新笔记。

#### 2. 梯度随时间步消失

反向传播到时间步 $t$,对 $h_0$ 的梯度是:

$$\frac{\partial L}{\partial h_0} = \frac{\partial L}{\partial h_T}\prod_{t=1}^{T}\frac{\partial h_t}{\partial h_{t-1}} = \frac{\partial L}{\partial h_T}\prod_{t=1}^{T}\tanh'(z_t)\cdot W_{hh}$$

- $|\tanh'|\le 1$,长时间步连乘 → **梯度消失**,前面的信息学不进去,也记不住。
- $|W_{hh}|>1$ → **梯度爆炸**。
- 一般情况下,超过 20-50 步梯度就基本没了。这就是课前小测里"第 1000 个词感受不到第 1 个词"的根因。

#### 3. LSTM / GRU 如何缓解

LSTM 在 $h_t$ 之外加了一条"细胞状态" $c_t$,通过三个门控制信息流动:

$$
\begin{aligned}
f_t &= \sigma(W_f\cdot[h_{t-1}, x_t]) \quad \text{(遗忘门:丢多少旧记忆)} \\
i_t &= \sigma(W_i\cdot[h_{t-1}, x_t]) \quad \text{(输入门:写多少新信息)} \\
\tilde{c}_t &= \tanh(W_c\cdot[h_{t-1}, x_t]) \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(c_t 的更新是加性的!)} \\
o_t &= \sigma(W_o\cdot[h_{t-1}, x_t]) \quad \text{(输出门)} \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

关键在于 $c_t = f_t\odot c_{t-1} + \dots$ 是**加法更新**,$f_t$ 接近 1 时梯度可以**几乎无损地流过**,大大缓解了连乘衰减。GRU 是 LSTM 的简化版,把三个门压成两个,效果相近、参数更少。

> ✅ **思考一下**:既然 LSTM 用加法更新解决了梯度消失,那为什么今天大模型还是抛弃了 LSTM?提示:梯度能流过 ≠ 计算能并行。LSTM 在 GPU 上有什么硬伤?

## 为什么这样设计

### CNN vs RNN

| 维度 | CNN | RNN |
|---|---|---|
| 处理数据类型 | 网格化(图像、谱图) | 序列(文本、语音、时序) |
| 核心机制 | 卷积核 + 参数共享 + 池化 | 隐藏状态递推 $h_t=f(h_{t-1},x_t)$ |
| 关键假设 | 局部性 + 平移不变性 | 时序依赖 |
| 并行性 | **空间维度完全并行**(每个位置独立算) | **时间维度无法并行**($h_t$ 依赖 $h_{t-1}$) |
| 核心局限 | 长程依赖靠堆深(感受野有限) | 梯度随时间步消失/爆炸 |

### 为什么 Transformer 把两者都取代了

| 维度 | CNN | RNN | Transformer(Attention) |
|---|---|---|---|
| 任意两位置的距离 | $O(\log n)$ 层(堆深) | $O(n)$ 步(逐步传递) | **$O(1)$**,任意两位置直接 attention |
| 训练并行性 | 空间并行 ✓,层间串行 | **时间维度不可并行** ✗ | **位置维度完全并行** ✓ |
| 长程依赖建模 | 一般 | 差(消失) | 强 |
| GPU 利用率 | 中 | **低**(时序串行,GPU 大量空闲) | 高(矩阵乘) |

Transformer 赢在两件事:**任意两个 token 之间一步直达**(没有长程衰减)+ **训练时序列维度可以完全并行**(GPU 友好)。RNN 在长序列上的训练效率只有 Transformer 的零头,这在数据量和模型规模爆炸的今天,是致命的。

## 代码:最小实现

### 1. 极简卷积(单通道、单核,numpy)

```python
import numpy as np

def conv2d_single(image, kernel, stride=1):
    """
    image:  (H, W)        单通道输入
    kernel: (KH, KW)      单个卷积核
    返回:    (H_out, W_out) 卷积结果
    """
    H, W      = image.shape
    KH, KW    = kernel.shape
    H_out     = (H - KH) // stride + 1
    W_out     = (W - KW) // stride + 1
    out       = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            # 取出局部感受野,与核逐元素相乘并求和
            region = image[i*stride:i*stride+KH, j*stride:j*stride+KW]
            out[i, j] = np.sum(region * kernel)
    return out

# 跑一下:用 Sobel 算子检测竖直边缘
img = np.array([
    [0,0,0,0,0],
    [0,1,1,0,0],
    [0,1,1,0,0],
    [0,1,1,0,0],
    [0,0,0,0,0],
], dtype=float)
sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
print(conv2d_single(img, sobel_x))
# 左边为正(强响应),右边为负,说明这个位置有一条竖直边缘
```

### 2. RNN 单步前向

```python
def rnn_step(x_t, h_prev, W_xh, W_hh, b):
    """
    x_t:     (in_dim,)      第 t 步输入
    h_prev:  (hidden_dim,)  上一步隐藏状态
    返回:    h_t            当前隐藏状态
    """
    # 核心公式:h_t = tanh(W_xh x_t + W_hh h_prev + b)
    h_t = np.tanh(W_xh @ x_t + W_hh @ h_prev + b)
    return h_t

# 模拟序列前向
np.random.seed(0)
in_dim, hidden_dim = 4, 3
W_xh  = np.random.randn(hidden_dim, in_dim) * 0.1
W_hh  = np.random.randn(hidden_dim, hidden_dim) * 0.1
b     = np.zeros(hidden_dim)

seq_len = 6
xs      = np.random.randn(seq_len, in_dim)  # 假装是一个长度6的序列
h       = np.zeros(hidden_dim)
for t in range(seq_len):
    h = rnn_step(xs[t], h, W_xh, W_hh, b)
    print(f"step {t}: h = {h}")
# 注意:经过若干步 tanh 连乘,梯度会快速衰减。这正是 RNN 训练长序列的硬伤。
```

> 💡 点「运行到这里」在线试跑。把 `seq_len` 改到 50,观察每一步 `h` 是否越来越趋同——那就是隐藏状态表达能力的退化。

## ⚠️ 易错点 / 面试陷阱

> **RNN 的梯度消失 ≠ 普通 MLP 的梯度消失。** 普通 MLP 是层数多导致的;RNN 是**同一个权重矩阵 $W_{hh}$ 在时间维上反复相乘**导致的,根子是"时间维展开"这件事本身。LSTM 修的就是这个,用 $c_t$ 加法更新绕开连乘。

> **感受野不是"层数 × 核大小"。** 两层 $3\times3$ 感受野是 $5\times5$,不是 $6\times6$。公式:$RF_k = RF_{k-1} + (k_k-1)\cdot\prod_{i<k}s_i$。带 stride 要乘上之前所有 stride 的累积。面试里手算感受野是常考题。

> **CNN 也做不了真正意义上的长程依赖建模。** 即使感受野覆盖全图,卷积还是"局部加权求和"的归纳偏置,缺少全局任意两位置直接交互的机制。Vision Transformer(ViT)能赢 CNN,关键就是 attention 让任意两个 patch 直接交互。所以 CNN 不是"被 Transformer 干掉",而是"全局建模能力不够,被 Attention 补上了"。

> **"RNN 不能并行"指的是训练时。** 推理(生成)时 RNN 也得一步步串行,这点和自回归 Transformer 一样。但 Transformer 训练时一次喂入整个序列、所有位置同时算 attention,这一点是 RNN 永远做不到的。

## 🎯 面试会怎么考

- **八股题**:CNN 卷积的本质是什么?为什么参数共享有效?RNN 为什么会有梯度消失?
- **手撕题**:用 numpy 写一个二维卷积运算(不调库),要求支持 stride 和 padding。
- **深挖题**:
  - 怎么增大 CNN 的感受野?(堆叠小核 / dilation / pooling / 下采样)
  - LSTM 三个门(遗忘、输入、输出)分别起什么作用?为什么 $c_t$ 的更新能缓解梯度消失?
  - 同样的长序列建模,为什么 Transformer 比 LSTM 强?(并行 + 长程直达,两个都要答)

## 📂 简历可写的项目

**项目名:用 numpy 从零实现 CNN 做手写数字分类的前向传播**

- **描述**:不依赖 PyTorch/TensorFlow,纯 numpy 实现一个 LeNet 风格的小 CNN:含 2 个卷积层(5×5 核)、2 个 max-pooling 层、1 个全连接分类头。在 MNIST 上实现前向推理,准确率 ≥ 95%;额外实现一个极简反向传播,支持 SGD 训练并收敛到 98%+。
- **技术栈**:numpy、MNIST、Matplotlib。
- **加分点**:可视化每个卷积核学到的特征图;对比"全连接 vs CNN"在相同参数量下的效果差距,分析归纳偏置的价值。

## 🚀 挑战

把上面的 `rnn_step` 循环展开成 20 步前向,在每一步记录 $h_t$ 的范数 $\|h_t\|$。再画一张图:`h_norm vs time_step`。

然后做两件事并观察:
1. 把 $W_{hh}$ 初始化成接近单位阵(例如 $I + 0.01\times\text{random}$),范数衰减会明显改善。
2. 加上一个 LSTM 风格的"细胞状态" $c_t$,即使 $h_t$ 饱和,$c_t$ 仍能缓慢累积信息。

这个练习会让你**亲眼看到**"梯度/信息随时间步衰减"和"LSTM 的缓解机制",比死记公式深刻得多。

## 🔗 延伸阅读

1. **[A Beginner's Guide to CNN(Karpathy 的 CS231n 讲义)](https://cs231n.github.io/convolutional-networks/)** — 把卷积、池化、感受野讲得最清楚的讲义,带大量图示。
2. **[Understanding LSTM Networks(colah's blog)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)** — LSTM 经典入门,三个门的图解极其直观,LSTM 必读。
3. **[The Annotated Transformer(Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/)** — 下一篇(Transformer)的预习材料,把 Attention is All You Need 全文代码化。读完这一节再看 Transformer 公式不会陌生。

---

## ✅ 课后小测(答案)

**课前小测答案:**
1. 卷积核在图上滑动,**每个位置只看一个小的局部窗口(如 3×3),且滑动过程中核的权重不变**。前者(局部连接)让参数从"全图的全部像素"降到"一个小窗口";后者(参数共享)让同一个特征检测器适用于全图任意位置。两者结合,把单层参数量从亿级降到几十几百。
2. $h_t=\tanh(W_{hh}h_{t-1}+W_{xh}x_t+b)$。隐藏状态 $h_t$ 在数学上是"截至第 $t$ 步、历史所有输入信息的**有损压缩**",它是 RNN 的"记忆"。
3. 反向传播到第 1 步的梯度是 $\prod_{t=2}^{1000}\frac{\partial h_t}{\partial h_{t-1}}$,一连串 $|\tanh'|\le1$ 的小数相乘,几乎衰减到 0。所以第 1000 个词的损失,对第 1 个词的影响微乎其微 → 长程依赖建模失败。

**掌握自检:** 如果你能不看笔记回答下面这题,说明这节过关了 ——

> 给面试官用一句话讲清楚:CNN、RNN、Transformer 三者各自的根本机制是什么?Transformer 凭什么同时取代前两者?
>
> **答**:CNN 用"局部卷积核 + 参数共享"高效提取局部特征,但长程依赖靠堆深、感受野有限;RNN 用"隐藏状态递推"建模时序,但必须串行计算、且梯度随时间步衰减;Transformer 用"全局 Attention"让任意两个位置一步直接交互,既没有长程衰减、训练时又能完全并行,所以在"长程依赖 + GPU 利用率"两条上同时击败两者,实现统一。
