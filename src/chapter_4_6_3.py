"""
4.6.3 K-core 渗流

k-core = 反复删掉度数小于 k 的节点后剩下的核心。
多层网络：节点必须同时在每一层的 k-core 里，才算属于多层 k-core。

关键结论：k-core 随删除比例的变化是突跳的（不连续相变），
而普通巨分量在单层网络中是连续出现的。

预期结果：k=1（等价巨分量）曲线在 p 较小处升起；k=2、k=3 的曲线在 p 更大处突然跳起来，跳变幅度比 k=1 明显——这就是原文说的"duplex 上 k-core 总是不连续地出现"。
"""

import random
import networkx as nx
import matplotlib.pyplot as plt
from giant import er_layer

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def k_core_size(G, k, active):
    """求子图 G[active] 中 k-core 的节点数"""
    sub = G.subgraph(active)
    return len(nx.k_core(sub, k))


def multi_k_core(Gs, k, p, max_iter=200):
    """
    多层 k-core：节点必须在每一层的 k-core 中才算存活。
    结构和 mcgc_simple 一样，只是把 giant 换成 k_core。
    """
    N = Gs[0].number_of_nodes()
    active = {i for i in range(N) if random.random() < p}

    for _ in range(max_iter):
        cores = []
        for G in Gs:
            sub = G.subgraph(active)
            cores.append(set(nx.k_core(sub, k).nodes()))
        new = set.intersection(*cores)
        if new == active:
            break
        active = new
    return len(active) / N


if __name__ == "__main__":
    random.seed(0)
    N, c, n_avg = 500, 4.0, 10
    p_list = [0.05 + 0.9 * i / 29 for i in range(30)]

    plt.figure(figsize=(8, 5))
    for k, color in zip([1, 2, 3], ["C0", "C1", "C2"]):
        S = [
            sum(
                multi_k_core([er_layer(N, c), er_layer(N, c)], k, p)
                for _ in range(n_avg)
            )
            / n_avg
            for p in p_list
        ]
        plt.plot(p_list, S, "o-", color=color, label=f"k={k}", markersize=4)

    plt.xlabel("p")
    plt.ylabel("k-core 比例")
    plt.title("4.6.3 多层 k-core 随 p 的变化")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
