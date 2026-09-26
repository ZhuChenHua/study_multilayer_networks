"""
论文实验所需的最小实现：ER 网络生成、随机失效、BFS 连通分支、MCGC、理论曲线。

论文对应关系：
  er_layer         -> 第 5 节“生成平均度为 c 的 ER 网络”
  random_failure   -> 第 2 节“每个节点以概率 q 被独立移除”
  largest_component-> 第 2 节 R(q) = |C_max(G_q)| / N
  mc_gc            -> 第 4 节相互连通巨分支 MCGC
  y_critical       -> 第 3、4 节的临界点 p_c = 1/c 与 y_c = c(1-q_c)
  theory_S         -> 第 3、4 节的自洽方程 S = p(1-e^{-cS})^M
"""

import random
from collections import deque

import numpy as np


def er_layer(N, c, seed=None):
    """平均度为 c 的 ER 随机图（边数 m = cN/2，随机抽边去重），返回邻接表。"""

    rng = random.Random(seed)
    m = round(c * N / 2)
    edges = set()
    while len(edges) < m:
        u, v = rng.randrange(N), rng.randrange(N)
        if u != v:
            edges.add((u, v) if u < v else (v, u))
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def random_failure(adj, q, seed=None):
    """随机攻击：每个节点以概率 q 独立失效，返回幸存节点列表。"""

    rng = random.Random(seed)
    return [i for i in range(len(adj)) if rng.random() >= q]


def largest_component(adj, active):
    """BFS 求诱导子图 G[active] 的最大连通分支 C_max。"""

    alive, seen, best = set(active), set(), []
    for start in active:
        if start in seen:
            continue
        component, queue = [], deque([start])
        seen.add(start)
        while queue:
            u = queue.popleft()
            component.append(u)
            for v in adj[u]:
                if v in alive and v not in seen:
                    seen.add(v)
                    queue.append(v)
        if len(component) > len(best):
            best = component
    return best


def mc_gc(layers, active, max_iter=100):
    """MCGC：反复取各层最大连通分支的交集，直到集合不再变化。"""

    current = set(active)
    for _ in range(max_iter):
        new = set(largest_component(layers[0], current))
        for adj in layers[1:]:
            new &= set(largest_component(adj, current))
        if new == current:
            break
        current = new
    return current


def y_critical(M):
    """
    临界参数 y_c = c p_c = c(1-q_c)。

    M = 1 时退化为 y_c = 1。M > 1 时由 g_y(x) = x - y(1-e^{-x})^M = 0 与 g_y'(x) = 0
    得 y_c = max_x x/(1-e^{-x})^M，其极大点满足 e^x = 1 + Mx，用二分法求解。
    M = 2 时给出 x_c ≈ 1.2564、y_c ≈ 2.4554。
    """

    if M == 1:
        return 1.0
    lo, hi = 1e-9, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if np.exp(mid) - 1 - M * mid > 0:
            hi = mid
        else:
            lo = mid
    x = (lo + hi) / 2
    return float(x / (1 - np.exp(-x)) ** M)


def q_critical(M, c):
    """临界移除概率 q_c = 1 - y_c / c。"""

    return 1 - y_critical(M) / c


def theory_S(q, c, M=1):
    """
    理论巨分支比例：S = p(1-e^{-cS})^M 的最大非平凡根。

    M >= 2 时阈值附近有一大一小两个正根，物理上稳定的是较大的那个，
    因此先在网格上找最后一个变号区间，再二分细化。
    """

    p = 1 - q
    f = lambda s: p * (1 - np.exp(-c * s)) ** M - s
    grid = np.linspace(1e-12, 1.0, 20001)
    values = f(grid)
    sign_change = np.nonzero(values[:-1] * values[1:] < 0)[0]
    if sign_change.size == 0:
        return 0.0
    lo, hi, f_lo = grid[sign_change[-1]], grid[sign_change[-1] + 1], values[sign_change[-1]]
    for _ in range(60):
        mid = (lo + hi) / 2
        if f_lo * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return float((lo + hi) / 2)
