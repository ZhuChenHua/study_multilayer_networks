"""
4.6.3 K‑core 渗流

k‑core = 反复删掉度数小于 k 的节点后剩下的核心。

多层网络：节点必须同时在每一层的 k‑core 里，才算属于多层 k‑core。
关键结论：k‑core 随删除比例的变化是突跳的（不连续相变），而普通巨分量在单层网络中是连续出现的。

预期结果：k=1（等价巨分量）曲线在 p 较小处升起；k=2、k=3 的曲线在 p 更大处突然跳起来，跳变幅度比 k=1 明显——这就是原文说的"duplex 上 k‑core 总是不连续地出现"。

依赖说明：er_layer 函数来自 giant.py 文件，请保证同目录下存在该文件。
"""

import random
import networkx as nx
import matplotlib.pyplot as plt
from giant import er_layer

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def k_core_size(G, k, active):
    """
    求子图 G[active] 中 k‑core 的节点数

    Parameters
    ----------
    G : nx.Graph
        原始网络
    k : int
        k‑core 的 k 值
    active : set
        当前存活节点集合

    Returns
    -------
    int
        k‑core 包含的节点数量
    """
    sub = G.subgraph(active)
    return len(nx.k_core(sub, k))


def multi_k_core(Gs, k, p, max_iter=200):
    """
    多层网络的多层 k‑core 不动点计算（duplex双层网络）
    算法流程：
        1. 以概率 p 随机采样初始存活节点集合 active
        2. 每一轮：对每一层网络，在当前active节点导出子图上求该层的k‑core
        3. 将所有层的k‑core节点做交集，作为新一轮active集合
        4. 迭代直到节点集合不再变化（达到不动点）或达到最大迭代次数
        5. 返回稳态多层k‑core占全部节点的比例

    Parameters
    ----------
    Gs : list[nx.Graph]
        多层网络，列表中每个元素代表一层图，所有层节点编号对齐
    k : int
        k‑core参数
    p : float
        节点初始存活概率 0~1
    max_iter : int, optional
        最大迭代轮次，防止死循环，默认200

    Returns
    -------
    float
        稳态多层k‑core节点占总节点的比例
    """
    N = Gs[0].number_of_nodes()
    # 按概率p随机初始化存活节点
    active = {i for i in range(N) if random.random() < p}

    for _ in range(max_iter):
        cores = []
        for G in Gs:
            sub = G.subgraph(active)
            core_nodes = set(nx.k_core(sub, k).nodes())
            cores.append(core_nodes)
        # 所有层k‑core取交集：必须每一层都在k‑core中才算存活
        new_active = set.intersection(*cores)
        # 集合没有变化，到达不动点，停止迭代
        if new_active == active:
            break
        active = new_active

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
    plt.ylabel("k‑core 比例")
    plt.title("4.6.3 多层 k‑core 随 p 的变化")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
