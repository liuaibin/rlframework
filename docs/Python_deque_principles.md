# CPython deque 原理

本文解释 Python `collections.deque` 在 CPython 中的底层结构，以及为什么
`append()` / `popleft()` 适合 replay buffer 这类滑动窗口场景。

> 说明：这里讲的是 CPython 实现细节。Python 语言层只承诺 `deque` 是双端队列；
> block、freeblocks 等属于 CPython 的具体实现。

## 结论

`deque` 不是普通 `list`，也不是“每个元素一个节点”的链表。CPython 的 `deque`
使用的是：

```text
固定大小 block 组成的双向链表
```

因此：

- `append()` / `appendleft()` / `pop()` / `popleft()` 都是近似 `O(1)`。
- 两端操作不会移动已有元素。
- 容量增长不是整体 `realloc`，而是在端点挂接新的固定大小 block。
- 弹空的 block 会优先进入小型缓存池，后续 append 时可以复用。
- `deque` 中保存的是 Python 对象引用，不会复制对象本体。

对于 `EpisodeReplayBuffer`：

```python
self.episodes.append(eps)
evicted_eps = self.episodes.popleft()
```

这两个操作只是在 `deque` 端点写入/读出 `SingleAgentEpisode` 的引用，不会复制
episode 内部的 observation、action、reward 等数据。

## 为什么不会复制 episode 本体

原因不是 `deque` 对 `SingleAgentEpisode` 做了什么特殊优化，而是 Python 容器的通用
语义就是：

```text
容器保存对象引用，而不是把对象本体内联到容器里
```

例如：

```python
d.append(eps)
```

在 CPython 里，`eps` 是一个指向 `SingleAgentEpisode` 对象的引用。`deque.append()`
做的事情可以理解成：

```text
1. 把 eps 这个对象的引用计数 +1
2. 把同一个 PyObject* 指针写入 rightblock.data[rightindex]
```

它不会调用：

```text
copy.copy(eps)
copy.deepcopy(eps)
eps.__copy__()
eps.__deepcopy__()
SingleAgentEpisode(...)
```

也不会遍历 `eps` 内部的字段，所以不会逐个复制 observation、action、reward。

可以把内存关系想象成：

```text
eps 变量
   │
   ▼
+----------------------+
| SingleAgentEpisode   |
| observations  ───────┼──► obs data
| actions       ───────┼──► action data
| rewards       ───────┼──► reward data
+----------------------+

deque block data[i]
   │
   └──────────────► 同一个 SingleAgentEpisode
```

`deque` 里多出来的是一份“指向 episode 的指针/引用”，不是一份新的 episode。

`popleft()` 也是类似：

```python
evicted_eps = self.episodes.popleft()
```

它可以理解成：

```text
1. 从 leftblock.data[leftindex] 读出 PyObject* 指针
2. leftindex += 1
3. deque 长度 -1
4. 把这个同一个对象引用返回给 evicted_eps
```

所以 `evicted_eps` 指向的仍然是原来的 `SingleAgentEpisode` 对象，而不是新复制出来的
对象。

可以用一个小实验验证：

```python
from collections import deque

class Episode:
    pass

eps = Episode()
eps.rewards = [1, 2, 3]

d = deque()
d.append(eps)

print(id(eps) == id(d[0]))  # True

out = d.popleft()
print(id(eps) == id(out))   # True

out.rewards.append(4)
print(eps.rewards)          # [1, 2, 3, 4]
```

如果 `append()` 或 `popleft()` 复制了 episode，那么 `id(...)` 不会相同，修改
`out.rewards` 也不会反映到 `eps.rewards` 上。

更精确地说，`deque` 操作里确实会“复制”一个很小的东西：

```text
复制 PyObject* 指针值
```

但这只是一个机器字大小的引用值，和复制 `SingleAgentEpisode` 内部的大量数组、列表、
字典不是一回事。

真正会复制 episode 数据的是显式复制操作，例如：

```python
eps = copy.deepcopy(eps)
self.episodes.append(eps)
```

这里的重成本来自 `copy.deepcopy(eps)`，不是后面的 `deque.append(eps)`。

## 和 list 的差异

`list` 底层是连续数组：

```text
[item0, item1, item2, item3, ...]
```

当右侧 append 且容量不足时，`list` 可能需要：

```text
1. 申请更大的连续内存
2. 把旧元素引用复制到新数组
3. 释放旧数组
```

而 `list.pop(0)` 更糟，因为它要把后面的元素整体向前移动：

```text
[A, B, C, D]
pop(0)
[B, C, D]

底层需要移动 B/C/D 的引用
```

所以 `list.pop(0)` 是 `O(n)`。

`deque` 底层不是一整块连续数组，而是一串 block：

```text
[block A] <--> [block B] <--> [block C]
```

容量不够时只需要挂一个新 block：

```text
[block A] <--> [block B] <--> [new block]
```

不会把所有旧元素搬迁到更大的连续数组。

## 核心数据结构

CPython 源码位置：

- `Modules/_collectionsmodule.c`
- https://github.com/python/cpython/blob/main/Modules/_collectionsmodule.c

核心结构可以简化为：

```c
#define BLOCKLEN 64
#define MAXFREEBLOCKS 16

typedef struct BLOCK {
    struct BLOCK *leftlink;
    PyObject *data[BLOCKLEN];
    struct BLOCK *rightlink;
} block;

typedef struct {
    PyObject_VAR_HEAD
    block *leftblock;
    block *rightblock;
    Py_ssize_t leftindex;
    Py_ssize_t rightindex;
    Py_ssize_t maxlen;
    Py_ssize_t numfreeblocks;
    block *freeblocks[MAXFREEBLOCKS];
} dequeobject;
```

关键字段：

```text
leftblock       最左侧 block
rightblock      最右侧 block
leftindex       第一个有效元素在 leftblock.data[] 里的位置
rightindex      最后一个有效元素在 rightblock.data[] 里的位置
maxlen          deque(maxlen=N) 的最大长度
freeblocks      缓存已摘除但可复用的 block
```

每个 block 内部是一个固定长度数组：

```text
block
+----------+----------+----------+-----+----------+
| data[0]  | data[1]  | data[2]  | ... | data[63] |
+----------+----------+----------+-----+----------+
```

`data[i]` 存的是 `PyObject*`，也就是 Python 对象引用。

## block 里面到底怎么存

一个 `block` 可以理解成：

```text
一个小数组 + 左右两个链表指针
```

结构是：

```text
+----------+------------------------------------------------+-----------+
| leftlink | data[0] data[1] data[2] ... data[63]            | rightlink |
+----------+------------------------------------------------+-----------+
```

其中：

```text
leftlink   指向左边相邻 block
rightlink  指向右边相邻 block
data       固定长度数组，长度是 BLOCKLEN
```

注意 `data` 的类型是：

```c
PyObject *data[BLOCKLEN];
```

这表示：

```text
data 是一个数组
数组里有 64 个槽位
每个槽位保存一个 PyObject* 指针
```

也就是说，block 里面不是直接塞 Python 对象本体，而是塞对象地址：

```text
data[0]  -> PyObject A
data[1]  -> PyObject B
data[2]  -> PyObject C
...
data[63] -> PyObject X
```

如果放的是 episode：

```text
block.data[0] ──► SingleAgentEpisode object
block.data[1] ──► SingleAgentEpisode object
block.data[2] ──► SingleAgentEpisode object
```

每个 `SingleAgentEpisode` 对象本体仍然在 Python 堆上，`block.data[i]` 只是指向它。

在 64 位机器上，一个 `PyObject*` 通常是 8 字节，所以一个 block 的 `data` 部分大致是：

```text
64 个指针 * 8 字节 = 512 字节
```

再加上 `leftlink` 和 `rightlink` 两个 block 指针。这里占用的是“指针数组”的空间，
不是 64 个 episode 对象本体的空间。

例如：

```python
d.append(eps0)
d.append(eps1)
d.append(eps2)
```

右端 block 内部大概变成：

```text
rightblock

data[0]  ──► eps0
data[1]  ──► eps1
data[2]  ──► eps2
data[3]      未使用
...
data[63]     未使用
```

`dequeobject` 里的 `leftindex` / `rightindex` 决定当前有效元素从哪里开始、到哪里结束。
所以 block 里不是所有槽位都一定有效，尤其是最左和最右的 block 经常只用了一部分。

比如一个 deque 只有 3 个元素，而且都在同一个 block 中：

```text
leftindex = 10
rightindex = 12

data[0]   未使用
...
data[9]   未使用
data[10]  ──► 第 1 个元素
data[11]  ──► 第 2 个元素
data[12]  ──► 第 3 个元素
data[13]  未使用
...
data[63]  未使用
```

如果元素跨多个 block：

```text
[leftblock] <--> [middle block] <--> [rightblock]

leftblock:     从 leftindex 到 data[63] 有效
middle block:  data[0] 到 data[63] 通常都有效
rightblock:    从 data[0] 到 rightindex 有效
```

所以 `deque` 的存储不是一整个连续大数组，而是多个“小的连续指针数组”串起来。

## rightindex 和 leftlink 分别干什么

这两个字段不是一类东西：

```text
rightindex  是 dequeobject 上的索引，表示右端元素在 rightblock.data[] 里的位置
leftlink    是 block 上的链表指针，表示当前 block 左边相邻的 block
```

### rightindex：定位右端最后一个有效元素

`rightindex` 要和 `rightblock` 一起看：

```text
rightblock + rightindex = deque 最右边那个元素的位置
```

比如：

```text
rightblock

data[0]  ──► item0
data[1]  ──► item1
data[2]  ──► item2
data[3]      未使用
...

rightindex = 2
```

这表示当前最右端元素是：

```text
rightblock.data[2]
```

所以 `rightindex` 主要用于：

```text
append(x)   知道新元素应该写到右侧哪个槽位
pop()       知道应该从右侧哪个槽位取元素
边界判断     判断当前 rightblock 是否已经写满
有效范围判断  判断 rightblock 里 data[0..rightindex] 哪些槽位有效
```

例如 `append(x)` 的核心就是：

```text
如果 rightindex 还没到 BLOCKLEN - 1:
    rightindex += 1
    rightblock.data[rightindex] = x
否则:
    在右侧挂一个新 block
    rightblock = newblock
    rightindex = 0
    rightblock.data[0] = x
```

所以 `rightindex` 不是指向下一个 block 的指针，它只是当前右端 block 内部的数组下标。

### leftlink：把 block 串成双向链表

`leftlink` 是每个 `block` 自己的字段：

```c
struct BLOCK *leftlink;
```

它指向左边相邻的 block：

```text
[block A] <--> [block B] <--> [block C]
                ^
                当前 block

block B.leftlink  ──► block A
block B.rightlink ──► block C
```

`leftlink` 主要用于：

```text
从右往左遍历 block
pop() 弹空 rightblock 时，快速回退到左边 block
在左侧插入/删除 block 时，维护双向链表关系
随机访问靠右侧元素时，可以从 rightblock 往左找
```

比如右侧 block 被 `pop()` 弹空时：

```text
[block A] <--> [block B empty]
```

可以通过：

```text
rightblock = rightblock.leftlink
```

快速回到 `block A`：

```text
[block A]
```

因此：

```text
rightindex 解决的是“右端元素在当前 block 的哪个数组槽位”
leftlink   解决的是“当前 block 左边还有哪个 block”
```

一个管 block 内部位置，一个管 block 之间的连接。

## 整体布局

一个非空 `deque` 可以想象成：

```text
deque object

leftblock  ─────┐
rightblock ───────────────────────┐
leftindex                         rightindex
    │                                  │
    ▼                                  ▼

[block A] <--> [block B] <--> [block C]
    │                              │
    └─ 第一个有效元素               └─ 最后一个有效元素
```

中间 block 通常是满的，只有左右两端 block 可能部分使用。

## append 的过程

执行：

```python
d.append(x)
```

如果右侧 block 还有空间，逻辑大致是：

```text
1. rightindex += 1
2. rightblock.data[rightindex] = x
3. len += 1
```

伪代码：

```c
rightindex++;
rightblock->data[rightindex] = item;
len++;
```

这是常数时间。

如果右侧 block 已满：

```text
[block A full]
```

则创建或复用一个新 block，挂到右侧：

```text
[block A full] <--> [block B]
                       ^
                    写入新元素
```

伪代码：

```c
if (rightindex == BLOCKLEN - 1) {
    block *b = newblock();
    rightblock->rightlink = b;
    b->leftlink = rightblock;
    rightblock = b;
    rightindex = -1;
}

rightindex++;
rightblock->data[rightindex] = item;
```

即使触发新 block，也只是挂接一个固定大小 block，不会移动已有元素。

## popleft 的过程

执行：

```python
d.popleft()
```

如果左侧 block 里还有元素，逻辑大致是：

```text
1. item = leftblock.data[leftindex]
2. leftindex += 1
3. len -= 1
4. return item
```

伪代码：

```c
item = leftblock->data[leftindex];
leftindex++;
len--;
return item;
```

这也是常数时间。

如果左侧 block 被弹空：

```text
[block A empty] <--> [block B] <--> [block C]
```

则把空 block 从链表上摘掉：

```text
[block B] <--> [block C]
```

摘掉的 block 会进入 `freeblocks` 缓存池，或者在缓存池满时释放。

## freeblocks 复用

`deque` 内部维护了一个小型 block 缓存池：

```text
freeblocks[MAXFREEBLOCKS]
```

当 `popleft()` 或 `pop()` 导致某个 block 彻底空掉时：

```text
如果 freeblocks 未满：
    把 block 放入 freeblocks
否则：
    真正释放 block
```

当 `append()` 或 `appendleft()` 需要新 block 时：

```text
如果 freeblocks 里有可复用 block：
    直接拿出来
否则：
    malloc 一个新 block
```

这对滑动窗口很重要：

```python
while True:
    d.append(new_item)
    d.popleft()
```

运行稳定后，右边需要新 block 时，经常可以复用左边刚弹空的 block。

所以 `deque` 并不是每次 append/popleft 都 malloc/free。

更准确的说法是：

```text
deque 可能在 block 边界发生分配或释放；
但不会整体 realloc，也不会搬迁已有元素；
并且空 block 有缓存复用机制。
```

## appendleft 和 pop

`appendleft(x)` 与 `append(x)` 对称：

```text
1. leftindex -= 1
2. leftblock.data[leftindex] = x
```

如果左侧 block 没空间，就在左边挂一个新 block。

`pop()` 与 `popleft()` 对称：

```text
1. item = rightblock.data[rightindex]
2. rightindex -= 1
3. return item
```

如果右侧 block 被弹空，就摘掉这个 block。

因此四个端点操作都是近似 `O(1)`：

```text
append
appendleft
pop
popleft
```

## 空 deque 的初始位置

空 `deque` 通常会保留一个 block，并把左右索引放在中间附近。

概念上：

```text
empty block

[ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ]
                ^
             middle
```

这样刚开始无论往左 `appendleft()`，还是往右 `append()`，都有一些可用空间，
不需要立刻申请新 block。

## 随机访问

`deque` 支持：

```python
d[i]
```

但它不适合高频随机访问中间元素。

原因是 `deque` 不是连续数组。访问中间元素时，需要从左端或右端较近的一边开始，
跨 block 找到目标位置。

因此：

```text
d[0]       O(1)
d[-1]      O(1)
d[i] 中间  O(n)
```

这也是为什么 `deque` 适合队列、滑动窗口、两端增删，不适合作为通用随机访问数组。

## deque(maxlen=N)

`deque` 可以指定最大长度：

```python
from collections import deque

d = deque(maxlen=3)
d.append(1)
d.append(2)
d.append(3)
d.append(4)

print(d)  # deque([2, 3, 4], maxlen=3)
```

超过 `maxlen` 时，它会在一端加入新元素，同时从另一端自动丢弃旧元素。

这仍然是端点操作，所以是 `O(1)`。

但对 RLlib 的 `EpisodeReplayBuffer` 不一定适用，因为 replay buffer 的 `capacity`
通常按 timestep 算，而不是按 episode 数算：

```text
episode A 长度 10
episode B 长度 500
episode C 长度 3
```

同样是 3 个 episode，占用的 timestep 数可能完全不同。

## 和 ring buffer 的关系

ring buffer 通常是：

```text
一整块固定数组 + head/tail 环形移动
```

示意：

```text
[0][1][2][3][4][5][6][7]
       ^           ^
      head        tail
```

`deque` 更像是“分段 ring buffer”：

```text
[block] <--> [block] <--> [block]
   ^                         ^
 left                      right
```

二者对比：

```text
ring buffer:
    优点：内存更连续，固定容量时非常简单
    缺点：容量变化、对象生命周期、逻辑索引映射需要自己处理

deque:
    优点：自动增长，两端操作 O(1)，无需整体搬迁，CPython 已高度优化
    缺点：中间随机访问不如 list/ring buffer，block 链表有少量指针开销
```

在 replay buffer 的 episode 队列里，`deque` 已经满足主要需求：

```text
右侧持续 append 新 episode
左侧持续 popleft 旧 episode
```

这正是 `deque` 的优势场景。

## replay buffer 场景下的性能含义

`EpisodeReplayBuffer.add()` 里常见路径：

```python
eps = copy.deepcopy(eps)
self.episodes.append(eps)
self._indices.extend((eps_idx, i) for i in range(len(eps)))

while self._num_timesteps > self.capacity and self.get_num_episodes() > 1:
    evicted_eps = self.episodes.popleft()
    self._indices = [
        idx_tuple
        for idx_tuple in self._indices
        if idx_tuple[0] != evicted_idx
    ]
```

这里：

```text
self.episodes.append(eps)      通常只是写入一个对象引用
self.episodes.popleft()        通常只是读出一个对象引用并移动 leftindex
```

真正更重的通常是：

```text
copy.deepcopy(eps)             复制 episode 内部数据
_indices.extend(...)           为每个 timestep 建 Python 索引对象
_indices 重建/过滤             扫描和复制大量 timestep 索引
metrics 更新                   Python 层统计开销
```

所以如果 replay buffer add 慢，优先排查：

```text
1. deepcopy 成本
2. _indices 的 tuple/list 分配
3. eviction 时 _indices 的重建
4. add 调用频率和 metrics 开销
5. 最后再看 deque 容器本身
```

## 小型验证

可以用下面的脚本观察 `deque` 和 `list.pop(0)` 的差异：

```python
from collections import deque
from time import perf_counter

ops = 200_000

for n in [1_000, 10_000, 100_000, 1_000_000]:
    d = deque(range(n))
    t0 = perf_counter()
    for i in range(ops):
        d.append(i)
        d.popleft()
    dt = perf_counter() - t0
    print(f"deque n={n:>9}: {dt / ops * 1e9:8.1f} ns/op pair")

for n in [1_000, 10_000, 100_000]:
    a = list(range(n))
    count = 20_000
    t0 = perf_counter()
    for i in range(count):
        a.append(i)
        a.pop(0)
    dt = perf_counter() - t0
    print(f"list  n={n:>9}: {dt / count * 1e6:8.1f} us/op pair")
```

典型现象：

```text
deque append+popleft 时间基本不随 n 增长
list append+pop(0) 时间随 n 明显增长
```

这和底层结构一致：

```text
deque.popleft() 只移动左端索引
list.pop(0)     需要移动后续元素
```

## 四个组成部分展开

可以把 CPython `deque` 理解成下面四层配合：

```text
固定大小 block + 双向链表 + 左右端索引 + block 缓存池
```

它们分别解决不同问题。

### 1. 固定大小 block：一小段连续槽位

`deque` 不是给每个元素单独分配一个节点，而是一次分配一个 block：

```text
block

+----------+------------------------------------------------+-----------+
| leftlink | data[0] data[1] data[2] ... data[63]            | rightlink |
+----------+------------------------------------------------+-----------+
```

每个 block 里有一段固定长度的 `data[]` 数组：

```c
PyObject *data[BLOCKLEN];
```

所以一个 block 内部是连续的：

```text
data[0], data[1], data[2], ..., data[63]
```

但 `data[i]` 存的是对象引用，不是对象本体：

```text
data[0] ──► object A
data[1] ──► object B
data[2] ──► object C
```

这样比“每个元素一个链表节点”更省内存，也更容易利用 CPU cache。因为一个 block
内部的多个元素指针是挨着放的。

### 2. 双向链表：把多个 block 串起来

当一个 block 放不下时，`deque` 不会像 `list` 那样申请一整块更大的连续内存，然后搬迁
所有旧元素。它会再拿一个 block，挂到左边或右边：

```text
[block A] <--> [block B] <--> [block C]
```

每个 block 通过两个指针连接相邻 block：

```text
leftlink   指向左边 block
rightlink  指向右边 block
```

例如：

```text
block B.leftlink  ──► block A
block B.rightlink ──► block C
```

这个结构的好处是：

```text
右边满了：在右边挂新 block
左边满了：在左边挂新 block
右边空了：摘掉右边 block
左边空了：摘掉左边 block
```

这些都只是改几个指针，不需要移动已有元素。

### 3. 左右端索引：标记有效元素边界

`dequeobject` 里有：

```c
block *leftblock;
block *rightblock;
Py_ssize_t leftindex;
Py_ssize_t rightindex;
```

它们共同描述当前 deque 的有效范围：

```text
leftblock + leftindex    = 第一个有效元素的位置
rightblock + rightindex  = 最后一个有效元素的位置
```

比如只有一个 block，里面有效元素在 `data[10]` 到 `data[12]`：

```text
leftindex = 10
rightindex = 12

data[0]   未使用
...
data[9]   未使用
data[10]  ──► 第 1 个元素
data[11]  ──► 第 2 个元素
data[12]  ──► 第 3 个元素
data[13]  未使用
...
data[63]  未使用
```

如果跨多个 block：

```text
[leftblock] <--> [middle block] <--> [rightblock]

leftblock:     leftindex 到 data[63] 有效
middle block:  data[0] 到 data[63] 通常都有效
rightblock:    data[0] 到 rightindex 有效
```

所以 `append()` 通常只动右端：

```text
rightindex += 1
rightblock.data[rightindex] = item
```

`popleft()` 通常只动左端：

```text
item = leftblock.data[leftindex]
leftindex += 1
```

这就是端点操作近似 `O(1)` 的关键：只改端点索引，不搬中间元素。

### 4. block 缓存池：复用摘下来的 block

`dequeobject` 里还有：

```c
Py_ssize_t numfreeblocks;
block *freeblocks[MAXFREEBLOCKS];
```

这是一个小型缓存池。

当 `popleft()` 把最左侧 block 弹空时：

```text
[empty block] <--> [block B] <--> [block C]
```

这个空 block 不一定马上 `free()` 掉，而是可能放进 `freeblocks`：

```text
freeblocks

[empty block 1]
[empty block 2]
...
```

之后 `append()` 需要新 block 时，优先从 `freeblocks` 里拿：

```text
如果 freeblocks 里有 block:
    复用旧 block
否则:
    malloc 新 block
```

这对滑动窗口很有用：

```python
d.append(new_item)
d.popleft()
```

长期运行时，左边刚摘下来的 block，经常可以被右边后续 append 复用，减少频繁
`malloc/free`。

把四者连起来看，一次典型的滑动窗口过程是：

```text
1. append 在 rightblock[rightindex + 1] 写入新对象引用
2. 如果 rightblock 满了，就在右边挂一个 block
3. popleft 从 leftblock[leftindex] 读出旧对象引用
4. 如果 leftblock 空了，就摘下来放进 freeblocks
5. 后面右边需要新 block 时，优先复用 freeblocks 里的 block
```

所以这句话可以翻译成更直白的版本：

```text
deque 用一段段固定大小的小数组保存对象引用；
小数组之间用双向链表连接；
左右端索引记录当前队列从哪里开始、到哪里结束；
不用的空小数组会先缓存起来，后续继续复用。
```

## 总结

`deque` 的核心原理是：

```text
固定大小 block + 双向链表 + 左右端索引 + block 缓存池
```

因此：

```text
append/popleft 不会整体扩容
append/popleft 不会移动已有元素
append/popleft 通常只是端点索引移动和对象引用读写
block 边界才会挂接/摘除固定大小 block
摘掉的 block 还可能被 freeblocks 复用
```

在 RLlib `EpisodeReplayBuffer` 这类滑动窗口中，`deque` 本身已经非常适合
episode 队列。性能优化更应该集中在 `deepcopy`、`_indices` 和 eviction 索引维护上。
