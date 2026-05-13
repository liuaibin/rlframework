# EpisodeReplayBuffer 中 deque 的 block 复用机制

`EpisodeReplayBuffer` 使用 `collections.deque` 维护 episode 队列：

```python
self.episodes.append(eps)
evicted_eps = self.episodes.popleft()
```

这里的 `append()` 和 `popleft()` 主要维护的是 `SingleAgentEpisode` 对象引用，不会把 episode 内部的 observation、action、reward 等数据复制进 deque。

## deque 的内部结构

CPython 的 `deque` 不是普通链表，也不是 `list` 那种连续大数组。它更像是“固定大小 block 组成的双向链表”：

```text
deque object
  leftblock                                      rightblock
      |                                              |
      v                                              v
  block A <------------> block B <------------> block C

block 内部:
  [slot0][slot1][slot2] ... [slotN]
```

每个 `slot` 存的是 Python 对象引用：

```text
slot -> SingleAgentEpisode object
```

因此，`self.episodes` 本身只保存 episode 引用，真正的大对象数据仍然在 `SingleAgentEpisode` 内部。

## append(eps) 如何维护元素

执行：

```python
self.episodes.append(eps)
```

大致过程是：

```text
1. 检查 rightblock 右侧是否还有空 slot
2. 如果有空位：
     rightindex += 1
     rightblock[rightindex] = eps
3. 如果没有空位：
     优先从 freeblocks 缓存中取一个 block
     如果缓存为空，再申请新 block
     将新 block 接到右侧
     把 eps 引用写入新 block
4. deque 长度 +1
```

示意：

```text
append 前:

block A
[eps1][eps2][eps3][empty]
                 ^
             rightindex

append(eps4) 后:

block A
[eps1][eps2][eps3][eps4]
                       ^
                   rightindex
```

如果当前 block 已满：

```text
append 前:

block A
[eps1][eps2][eps3][eps4]
                       ^
                   rightindex

append(eps5) 后:

block A                         block B
[eps1][eps2][eps3][eps4] <----> [eps5][empty][empty][empty]
                                  ^
                              rightindex
```

这里新增的是 `eps5` 的引用，不是复制 `eps5` 对象本体。

## popleft() 如何维护元素

执行：

```python
evicted_eps = self.episodes.popleft()
```

大致过程是：

```text
1. 读取 leftblock[leftindex] 中的对象引用
2. 清空这个 slot
3. leftindex += 1
4. deque 长度 -1
5. 如果 leftblock 已经全部弹空：
     从 block 链表中摘掉这个 block
     优先放入 deque 自己的 freeblocks 缓存
     如果缓存已满，才真正释放 block
6. 返回刚才读取到的对象引用
```

示意：

```text
popleft 前:

block A
[eps1][eps2][eps3][eps4]
  ^
leftindex

popleft() 后:

block A
[empty][eps2][eps3][eps4]
          ^
      leftindex

返回 eps1
```

如果整个 block 被弹空：

```text
popleft 前:

block A                         block B
[eps1][empty][empty][empty] <-> [eps2][eps3][empty][empty]
  ^
leftindex

popleft() 后:

block B
[eps2][eps3][empty][empty]
  ^
leftindex

block A 进入 freeblocks 缓存，后续 append 可复用。
```

## block 复用是怎么做到的

`deque` 内部维护了一个小型 block 缓存池，可以理解为：

```text
deque object
  leftblock
  rightblock
  leftindex
  rightindex
  freeblocks = [空闲 block, 空闲 block, ...]
```

当 `popleft()` 把左侧某个 block 弹空时，通常不会马上把这个 block 还给系统，而是放入 `freeblocks`：

```text
block A 被弹空
  -> 从 deque 链表摘掉
  -> 放入 freeblocks 缓存
```

当后续 `append()` 发现右侧 block 满了，需要新 block 时，会优先复用缓存：

```text
append 需要新 block
  -> 先从 freeblocks 取
  -> freeblocks 没有可用 block 时才申请新 block
```

因此滑动窗口场景下，block 会循环使用：

```text
左侧 popleft 弹空 block A
  -> block A 进入 freeblocks

右侧 append 需要扩容
  -> 从 freeblocks 取出 block A
  -> block A 接到右侧继续使用
```

这就是为什么 `deque.append()` 和 `deque.popleft()` 不会每次都触发 malloc/free。

## 空 deque 的特殊处理

当 deque 被弹到完全为空时，CPython 通常不会把最后一个 block 也立刻释放掉，而是保留一个空 block，并把左右索引放到中间附近。

这样下一次执行：

```python
d.append(x)
d.appendleft(y)
```

两边通常都有可用空间，不需要马上重新申请 block。

## 和 list.pop(0) 的区别

`list.pop(0)` 需要移动后续所有元素：

```text
[eps1, eps2, eps3, eps4]
 pop(0)
[eps2, eps3, eps4]
```

后面的引用需要整体左移，因此是 `O(n)`。

`deque.popleft()` 只需要移动左侧索引：

```text
leftindex += 1
```

必要时切换到下一个 block，因此是 `O(1)`。

## episode 对象什么时候释放

block 复用只复用 deque 容器的 slot 内存，不代表 `SingleAgentEpisode` 对象本身被复用。

执行：

```python
evicted_eps = self.episodes.popleft()
```

之后：

```text
deque 不再引用这个 episode
evicted_eps 局部变量仍然引用这个 episode
```

所以 episode 对象不会在 `popleft()` 的瞬间立即释放。它要等到没有任何引用后才会释放。

在 `EpisodeReplayBuffer.add()` 中，一般是：

```text
连续 eviction:
  上一个 evicted_eps 被下一轮赋值覆盖后，如果没有其他引用，就可以释放

最后一次 eviction:
  evicted_eps 通常等 add() 返回后，局部变量销毁，才可以释放
```

另外，即使 Python 对象已经释放，进程 RSS 也不一定马上下降，因为 Python、NumPy、PyTorch 或系统 allocator 都可能缓存内存用于后续复用。

## 对 EpisodeReplayBuffer 的性能结论

在 `EpisodeReplayBuffer.add()` 中：

```python
self.episodes.append(eps)
self.episodes.popleft()
```

本身不是主要性能瓶颈：

```text
append(eps)   O(1)
popleft()     O(1)
```

真正更值得关注的是：

```python
eps = copy.deepcopy(eps)
self._indices.extend(...)
self._indices = new_indices
```

原因是：

- `copy.deepcopy(eps)` 可能复制 episode 内部大量数据。
- `_indices.extend(...)` 会为每个 timestep 增加索引。
- eviction 时重建 `_indices` 可能扫描大量 timestep 索引。
- `_indices` 是 timestep 粒度，长度可能远大于 `self.episodes` 的 episode 数量。

因此，如果要优化 `EpisodeReplayBuffer.add()`，优先级通常是：

```text
1. 减少 deepcopy 成本
2. 减少 _indices 的 tuple/list 分配
3. 减少 eviction 时 _indices 的重建扫描
4. 再考虑 deque 容器本身
```

`deque` 的 block 复用机制已经让 `append()` / `popleft()` 在滑动窗口场景下足够高效。
