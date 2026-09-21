"""
MCGC = Mutually Connected Giant Component（互连巨分量）

一个节点要在多层网络中“真正存活”，必须在所有层里都能通过该层的连接到达一个宏观大簇。只要它在某一层掉出巨分量，它在其他层的副本也会跟着失效。这种“层间互相拖累”正是相互依赖网络脆弱性的根源。

假设两层 A、B，完全相互依赖。
- 初始时，A 层有一个巨分量 G_a，B 层有一个巨分量 G_b
- 某个节点 i 同时在 G_a 和 G_b 中 → i 存活
- 但 G_a 里可能有些节点不在 G_b 中，这些节点会因 B 层失效而死亡
- 它们一死，A 层的巨分量 G_a 可能缩小，甚至把 i 踢出去
- 所以需要重新计算 A 层当前存活节点下的巨分量，再和 B 层取交集
这个过程会反复进行，直到某一轮交集不再变化，剩下的节点才是 MCGC。
"""

import random
import networkx as nx
from matplotlib import pyplot as plt
from lessons.giant import giant


def mcgc_simple(Gs, p, max_iter=200):
    """
    完全相互依赖多层网络的 MCGC

    原理：节点存活 <=> 它在每一层的巨分量中。
    每轮：各层巨分量取交集，重复到收敛。
    Args:
        Gs: 各层图组成的列表，所有层节点编号相同
        p: 节点存活概率
        max_iter: 最大迭代次数
    Returns:
        MCGC 大小占节点总数的比例
    """

    N = Gs[0].number_of_nodes()
    # active 是初始存活节点集合，每个节点以概率 p 存活
    active = {i for i in range(N) if random.random() < p}
    for _ in range(max_iter):
        giants = [giant(G, active) for G in Gs]
        # 取交集
        new = set.intersection(*giants)
        # 如果这一轮和上一轮的活跃集合一样，说明已经收敛，跳出。
        if new == active:
            break
        active = new
    return len(active) / N


if __name__ == "__main__":
    from lessons.giant import er_layer

    N, c = 100, 4.0
    Gs = [er_layer(N, c) for _ in range(2)]
    for G in Gs:
        nx.draw(G, with_labels=True)
        plt.show()
    print("MCGC 大小占节点总数的比例 = ", mcgc_simple(Gs, 0.5))
    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f"p = {p}, MCGC 大小占节点总数的比例 = {mcgc_simple(Gs, p):.3f}")
