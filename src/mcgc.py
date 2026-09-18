import random
from giant import giant


def mcgc_simple(Gs, p, max_iter=200):
    """
    完全相互依赖多层网络的 MCGC（简化版）

    原理：节点存活 <=> 它在每一层的巨分量中。
    每轮：各层巨分量取交集，重复到收敛。
    """
    N = Gs[0].number_of_nodes()
    active = {i for i in range(N) if random.random() < p}
    for _ in range(max_iter):
        giants = [giant(G, active) for G in Gs]
        new = set.intersection(*giants)
        if new == active:
            break
        active = new
    return len(active) / N


if __name__ == "__main__":
    pass
