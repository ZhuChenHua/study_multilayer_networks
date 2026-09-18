"""
4.5.2 度关联下的渗流

两层节点之间的度关联有三种典型设置：
  MP (maximally-positive): 两层高度节点是同一批 → p_c 最小
  MN (maximally-negative): 一层高度节点在另一层是低度节点 → p_c 最大
  UC (uncorrelated):       随机配对 → p_c 居中

构造方法：先生成两层 BA 网络，再按度数排名重新标记第二层的节点。
"""

import random
import networkx as nx
import matplotlib.pyplot as plt
from mcgc import mcgc_simple

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def ba_layer(N, m):
    """BA 无标度网络层，平均度约 2m"""
    return nx.barabasi_albert_graph(N, m)


def align(G1, G2, mode):
    """
    重新标记 G2 的节点，使两层之间形成指定的度关联：
      MP: G2 中度数高的节点对齐 G1 中度数高的节点
      MN: G2 中度数高的节点对齐 G1 中度数低的节点
      UC: 随机打乱
    """
    n = G1.number_of_nodes()
    order1 = sorted(range(n), key=lambda v: G1.degree(v), reverse=True)  # G1 度数降序

    if mode == "MP":
        order2 = sorted(range(n), key=lambda v: G2.degree(v), reverse=True)
    elif mode == "MN":
        order2 = sorted(range(n), key=lambda v: G2.degree(v))  # G2 度数升序
    else:  # UC
        order2 = list(range(n))
        random.shuffle(order2)

    # 把 G2 中度数排名第 k 的节点改名为 G1 中度数排名第 k 的节点
    mapping = {old: new for old, new in zip(order2, order1)}
    return nx.relabel_nodes(G2, mapping)


def duplex_correlated(N, m, mode):
    """生成一对两层网络，并施加指定的度关联"""
    G1 = ba_layer(N, m)
    G2 = ba_layer(N, m)
    G2 = align(G1, G2, mode)
    return [G1, G2]


def compute_curve(N, m, mode, p_list, n_avg):
    """对一组 p 值，计算 MCGC 比例的平均值"""
    S_list = []
    for p in p_list:
        S_sum = 0.0
        for _ in range(n_avg):
            Gs = duplex_correlated(N, m, mode)
            S_sum += mcgc_simple(Gs, p)
        S_list.append(S_sum / n_avg)
    return S_list


if __name__ == "__main__":
    random.seed(0)
    N, m, n_avg = 500, 2, 20
    p_list = [0.05 + 0.9 * i / 29 for i in range(30)]

    plt.figure(figsize=(8, 5))
    for mode in ["MP", "UC", "MN"]:
        S_list = compute_curve(N, m, mode, p_list, n_avg)
        plt.plot(p_list, S_list, "o-", label=mode, markersize=4)

    plt.xlabel("p")
    plt.ylabel("S")
    plt.title("4.5.2 度关联对 MCGC 的影响")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
