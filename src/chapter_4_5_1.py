"""
4.5.1 链接重叠下的渗流

overlap ∈ [0, 1]，越大两层共享边越多：
  overlap=1 退化为单层，连续相变；
  overlap=0 完全独立，相互依赖导致突跳；
  中间值介于两者之间，观察转变是否变平滑。
"""

import random
import networkx as nx
from matplotlib import pyplot as plt
from giant import er_layer
from mcgc import mcgc_simple

random.seed(0)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def duplex_overlap(N, c, overlap):
    """
    生成两层部分重叠的 duplex：先复制 G1 中 overlap 比例的边，再随机补边到 G2
    Args:
        N: 节点数
        c: 平均度
        overlap: 两层重叠比例
    Returns:
        Gs: 两层网络列表
    """
    G1 = er_layer(N, c)
    G2 = nx.Graph()
    G2.add_nodes_from(range(N))

    # 从 G1 中随机抽 overlap 比例的边，复制到 G2（共享边）
    edges = list(G1.edges())
    G2.add_edges_from(random.sample(edges, int(len(edges) * overlap)))

    # 随机补边，使 G2 边数与 G1 相同（平均度仍约 c）
    for i, j in random.sample(
        [(i, j) for i in range(N) for j in range(i + 1, N)], 10 * N
    ):
        if G2.number_of_edges() >= len(edges):
            break
        G2.add_edge(i, j)
    return [G1, G2]


if __name__ == "__main__":
    random.seed(0)
    N, c, n_avg = 500, 3.0, 20
    p_list = [0.05 + 0.9 * i / 29 for i in range(30)]

    plt.figure(figsize=(8, 5))
    for ov, color in zip([0.0, 0.5, 1.0], ["C3", "C2", "C0"]):
        S = [
            sum(mcgc_simple(duplex_overlap(N, c, ov), p) for _ in range(n_avg)) / n_avg
            for p in p_list
        ]
        plt.plot(p_list, S, "o-", color=color, label=f"overlap={ov}", markersize=4)

    plt.axvline(1 / c, ls="--", c="gray", alpha=0.6, label=f"1/c={1/c:.2f}")
    plt.axvline(2.455 / c, ls=":", c="red", alpha=0.6, label=f"2.455/c={2.455/c:.2f}")
    plt.xlabel("p")
    plt.ylabel("S")
    plt.title("4.5.1 链接重叠对 MCGC 的影响")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
