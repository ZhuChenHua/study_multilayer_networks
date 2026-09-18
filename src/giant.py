"""
渗流（percolation）
在一个网络中随机保留一部分节点（或边），当保留比例 p 超过某个临界值 p_c 时，网络中会突然出现一个连通整个网络的大分量（巨分量）。这个现象叫渗流。
"""

import random
import networkx as nx
from matplotlib import pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # Windows 系统使用黑体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号


def er_layer(N, c):
    """
    Args:
        N: 节点数
        c: 每个节点的平均度
    Returns:
        ER 随机图，networkx.Graph 对象
    """

    return nx.gnp_random_graph(N, c / (N - 1))


def giant(G, active):
    """
    最大连通分量
    Args:
        G: networkx 图
        active: 节点集合
    Returns:
        最大连通分量的节点集合
    """
    sub = G.subgraph(active)  # 从原图 G 中提取出仅包含 active 中节点的子图
    if len(sub) == 0:
        return set()  # 如果子图为空，返回空集合
    # 找出子图中所有连通分量（返回生成器，每个分量是一个节点集合），key=len 表示选出节点数最多的那个连通分量
    return max(nx.connected_components(sub), key=len)


# ============================================================
# 演示：单层网络巨分量随 p 的变化
# ============================================================
if __name__ == "__main__":
    random.seed(0)

    N, c = 100, 3.0
    G = er_layer(N, c)
    print(G)
    nx.draw(G)
    plt.show()

    print(f"N = {N}, c = {c}, 理论 p_c = 1/c = {1/c:.3f}")
    print(f"{'p':>6} {'巨分量比例 S':>12} {'是否渗流':>10}")

    for p in [0.1, 0.2, 0.3, 0.33, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        S_list = []
        for _ in range(10):  # 每次 p 做 10 次实验取平均
            active = {i for i in range(N) if random.random() < p}
            S_list.append(len(giant(G, active)) / N)
        S = sum(S_list) / len(S_list)
        percolating = "是" if S > 0.1 else "否"
        print(f"{p:>6.2f} {S:>14.3f} {percolating:>10}")
