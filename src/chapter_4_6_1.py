"""
4.6.1 经典渗流

没有相互依赖：层内和层间链接同等对待，全部合并成一张投影网络。
节点只要通过任意链接连到巨分量，就属于巨分量。

对比：单层 ER vs 两层投影。投影网络平均度约 2c，所以 p_c 更小。
"""

import random
import networkx as nx
import matplotlib.pyplot as plt
from giant import er_layer, giant

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def projected(Gs):
    """
    把多层网络合并成一张投影网络：所有层内边直接叠加
    Args:
        Gs: 多层网络列表
    Returns:
        投影网络
    """
    P = nx.Graph()
    P.add_nodes_from(range(Gs[0].number_of_nodes()))
    for G in Gs:
        P.add_edges_from(G.edges())
    return P


def compute_single(G, p_list, n_avg):
    """
    单图节点渗流：对一张图执行节点随机保留仿真，计算巨分量占比S
    Args:
        G: networkx 图
        p_list: 保留比例列表
        n_avg: 每个 p 做多少次实验取平均
    Returns:
        S_list: 保留比例列表对应的 S 值列表
    """

    N = G.number_of_nodes()
    S_list = []
    for p in p_list:
        S_sum = 0.0
        for _ in range(n_avg):
            active = {i for i in range(N) if random.random() < p}
            S_sum += len(giant(G, active)) / N
        S_list.append(S_sum / n_avg)
    return S_list


if __name__ == "__main__":
    random.seed(0)
    N, c, n_avg = 500, 3.0, 20
    p_list = [0.05 + 0.9 * i / 29 for i in range(30)]

    G1 = er_layer(N, c)
    G2 = er_layer(N, c)
    P = projected([G1, G2])

    plt.figure(figsize=(8, 5))
    plt.plot(
        p_list, compute_single(G1, p_list, n_avg), "o-", label="单层 ER", markersize=4
    )
    plt.plot(
        p_list,
        compute_single(P, p_list, n_avg),
        "s-",
        label="两层投影（经典渗流）",
        markersize=4,
    )

    plt.axvline(
        1 / c, ls="--", c="gray", alpha=0.6, label=f"单层理论 p_c=1/c={1/c:.2f}"
    )
    plt.axvline(
        1 / (2 * c),
        ls=":",
        c="red",
        alpha=0.6,
        label=f"投影理论 p_c=1/(2c)={1/(2*c):.2f}",
    )

    plt.xlabel("p")
    plt.ylabel("S")
    plt.title("4.6.1 经典渗流：单层 vs 投影")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
