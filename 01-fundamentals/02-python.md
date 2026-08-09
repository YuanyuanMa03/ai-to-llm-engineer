# 02 · Python 与数据栈

> 一句话:**学完这节,你能用 numpy 向量化写出比 for 循环快 100 倍的代码,并回答面试题「为什么向量化比循环快」。**

## 🤔 课前小测

1. `np.array([1,2,3]) + np.array([4,5,6])` 的结果是什么?这叫什么运算?
2. 一个形状 `(100, 3)` 的矩阵,它的 `axis=0` 和 `axis=1` 分别指哪个方向?
3. 为什么 numpy 数组运算比 Python list 的 for 循环快很多?(一句话)

---

## 这节解决什么问题

深度学习的代码 90% 是张量运算。如果你还在用 `for` 循环逐元素处理数据,训练会慢到无法接受。本节让你掌握**向量化思维**和 numpy / PyTorch 的张量操作,这是后面所有代码的基础。

## 核心概念 ★

### numpy 数组 vs Python list

| 特性 | Python list | numpy array |
|------|-------------|-------------|
| 元素类型 | 可混合 | 必须同质 |
| 运算 | 需循环 | 向量化(批量) |
| 速度 | 慢 | C 层实现,快 10-100× |
| 内存 | 灵活开销大 | 连续存储,省 |

**广播(Broadcasting)** 是 numpy 最强大的特性 —— 不同形状的数组能自动对齐运算:

$$
\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}_{3 \times 1} + \begin{bmatrix} 10 & 20 \end{bmatrix}_{1 \times 2} = \begin{bmatrix} 11 & 21 \\ 12 & 22 \\ 13 & 23 \end{bmatrix}_{3 \times 2}
$$

### 张量操作四件套

无论 numpy 还是 PyTorch,核心操作就四类:

| 操作 | numpy | PyTorch | 用途 |
|------|-------|---------|------|
| 改形状 | `X.reshape(...)` | `x.view(...)` / `x.reshape(...)` | batch 处理 |
| 索引 | `X[mask]` | `x[mask]` | 筛选样本 |
| 归约 | `X.sum(axis=0)` | `x.sum(dim=0)` | 算 loss / 统计 |
| 拼接 | `np.concatenate` | `torch.cat` | 合并 batch |

### 维度(axis)的直觉

对形状 `(B, C, H, W)` 的图片张量:
- `axis=0` = batch 维(哪张图)
- `axis=1` = 通道维(红绿蓝)
- `axis=2,3` = 空间维(高、宽)

`X.mean(axis=0)` =「跨所有样本求平均」→ 结果消掉第 0 维。

> ✅ **思考一下**:`X.sum(axis=1, keepdims=True)` 对一个 `(4, 3)` 的矩阵,结果是什么形状?(提示:keepdims 保留维度)

## 为什么这样设计

### 为什么向量化快?

Python 是解释型语言,每次循环都要动态类型检查。numpy 的运算在底层调 C/Fortran 实现的 BLAS 库,**一条指令处理整个数组**(SIMD 指令集),还利用 CPU 缓存局部性。

| 方式 | 10 万元素求和耗时 | 原因 |
|------|-------------------|------|
| `for` 循环 | ~5 ms | 每次循环 Python 解释开销 |
| `np.sum` | ~0.05 ms | C 层 + SIMD + 缓存友好 |

差距约 **100 倍**。这就是为什么深度学习代码必须向量化。

### 为什么张量要连续存储?

为了缓存友好。CPU/GPU 取数据是按块取的(缓存行),连续内存能一次取一整块,离散内存要多次取。这直接决定训练速度。

## 代码:最小实现

**① 向量化 vs 循环速度对比**

```python
import numpy as np
import time

N = 1_000_000
a = np.random.rand(N)
b = np.random.rand(N)

# 循环版(慢)
t0 = time.time()
s1 = 0
for i in range(N):
    s1 += a[i] * b[i]
t_for = time.time() - t0
print(f'for 循环: {t_for:.4f}s')

# 向量化(快)
t0 = time.time()
s2 = np.dot(a, b)
t_vec = time.time() - t0
print(f'向量化: {t_vec:.4f}s')
print(f'加速比: {t_for/t_vec:.1f}x')   # 通常 50-100x
```

**② 广播的实际用处:批量计算距离**

```python
# 100 个样本,每个 3 维特征
X = np.random.rand(100, 3)
# 一个查询点
q = np.random.rand(3)

# ❌ 慢:循环算每个样本到 q 的距离
# ✅ 快:广播一次性算
diff = X - q          # (100,3) - (3,) → 广播成 (100,3)
dist = np.sqrt((diff ** 2).sum(axis=1))  # (100,) 每个样本的距离
print('最近样本索引:', dist.argmin())
```

**③ PyTorch 张量(后面阶段会大量用)**

```python
import torch

x = torch.randn(4, 3)          # 随机矩阵,正态分布
print('形状:', x.shape)          # torch.Size([4, 3])
print('跨 dim0 求和:', x.sum(dim=0).shape)  # torch.Size([3])
print('reshape:', x.view(2, 6).shape)       # torch.Size([2, 6])
print('是否 GPU:', torch.cuda.is_available())   # 检查 GPU 是否可用
```

> 💡 PyTorch 的 `dim` 等价于 numpy 的 `axis`,概念完全一致。

## ⚠️ 易错点 / 面试陷阱

> **陷阱 1**:广播规则 —— 从**末尾**对齐维度。`(4,3)+(3,)` 能广播,`(4,3)+(4,)` 不能(末尾 3≠4),会报错。

> **陷阱 2**:`X.reshape` 可能返回**视图**(共享内存)或**副本**,改一个可能影响另一个。不确定时用 `.copy()`。

> **陷阱 3**:`axis` 方向别记反 —— `axis=0` 是「跨行」(消第 0 维),不是「第 0 行」。面试常考。

## 🎯 面试会怎么考

- **八股题**:「numpy 广播规则是什么」「为什么向量化比循环快」「PyTorch 的 view 和 reshape 区别」
- **手撕题**:「用 numpy 实现两个矩阵的欧氏距离矩阵(不能用循环)」「写一个 batch 归一化」
- **深挖题**:「reshape 什么时候返回视图什么时候返回副本」「einsum 你用过吗」

## 📂 简历可写的项目

**KNN 分类器纯 numpy 实现** —— 不用 sklearn,只用 numpy 实现 KNN(含距离矩阵向量化计算),在 iris / MNIST 子集上跑通并对比不同 k 值。
> 项目名:「基于 numpy 的 KNN 分类器实现与向量化优化」,技术栈:numpy + matplotlib。

## 🚀 挑战

上面②的距离计算,尝试用**爱因斯坦求和(`np.einsum` 或 `einops`)**重写一遍。再尝试:如何一次性算出 100 个样本**两两之间**的距离矩阵(形状 `(100,100)`),不用双重循环?

> 提示:广播可以用 `X[:, None, :] - X[None, :, :]` 制造三维差。

## 🔗 延伸阅读

- 📄 [numpy 广播规则官方文档](https://numpy.org/doc/stable/user/basics.broadcasting.html) —— 一页讲透广播
- 💻 [100 numpy exercises](https://github.com/rougier/numpy-100) —— 从入门到进阶
- 📄 [PyTorch 张量教程](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) —— 官方入门

---

## ✅ 课后小测(答案)

**课前小测答案:**

1. `[5, 7, 9]`,这叫**逐元素相加(element-wise)**,不是矩阵运算。
2. `axis=0` = 沿「行」方向(跨 100 个样本),`axis=1` = 沿「列」方向(跨 3 个特征)。`X.sum(axis=0)` 结果形状 `(3,)`。
3. numpy 底层是 C 实现 + SIMD 指令 + 缓存友好,避免了 Python 循环的解释开销。

**掌握自检:** 如果你能不看笔记回答下面这题,说明这节过关了 ——
> 解释广播规则,并说出 `(4,1,3)` 和 `(1,5,3)` 两个数组相加,结果的形状是什么。
> (答案:`(4,5,3)`,每个维度取 max)
