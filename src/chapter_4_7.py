"""
4.7 多层网络上的级联
"""

import random
from matplotlib import pyplot as plt
from giant import er_layer

random.seed(0)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def watts_cascade(Gs, seed_frac, phi, max_iter=100):
    """任意一层中激活邻居比例 >= phi，节点就激活"""
    N = Gs[0].number_of_nodes()
    active = {i for i in range(N) if random.random() < seed_frac}
    for _ in range(max_iter):
        new = set(active)
        for i in range(N):
            if i in active:
                continue
            for G in Gs:
                nb = list(G.neighbors(i))
                if nb and sum(1 for j in nb if j in active) / len(nb) >= phi:
                    new.add(i)
                    break
        if new == active:
            break
        active = new
    return len(active) / N


random.seed(0)
N, c, n_avg = 300, 4.0, 10
G1, G2 = er_layer(N, c), er_layer(N, c)
S1 = sum(watts_cascade([G1], 0.01, 0.2) for _ in range(n_avg)) / n_avg
S2 = sum(watts_cascade([G1, G2], 0.01, 0.2) for _ in range(n_avg)) / n_avg

plt.figure(figsize=(4, 3))
plt.bar(["单层", "多层"], [S1, S2], color=["C0", "C3"])
plt.ylabel("最终激活比例")
plt.title("4.7 Watts 阈值级联")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
