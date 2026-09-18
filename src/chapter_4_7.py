"""
4.7 多层网络上的级联

正例：阈值 φ 小 → 一个小种子就引爆全局级联。
反例：阈值 φ 大 → 级联被抑制，停在种子附近。
"""

import random
from matplotlib import pyplot as plt
from giant import er_layer

random.seed(0)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def threshold_cascade(Gs, phi, seed_frac=0.05, max_iter=200):
    """节点在每一层都有 >= phi 比例邻居已激活时被激活"""
    N = Gs[0].number_of_nodes()
    active = set(random.sample(range(N), int(seed_frac * N)))
    history = [len(active) / N]
    for _ in range(max_iter):
        new = set(active)
        for i in range(N):
            if i in active:
                continue
            if all(
                len(set(G.neighbors(i)) & active) / max(len(set(G.neighbors(i))), 1)
                >= phi
                for G in Gs
            ):
                new.add(i)
        if new == active:
            break
        active = new
        history.append(len(active) / N)
    return history


N, c = 200, 6.0
Gs = [er_layer(N, c) for _ in range(2)]

plt.figure(figsize=(7, 4))
for phi in [0.15, 0.35, 0.55]:
    h = threshold_cascade(Gs, phi)
    plt.plot(range(len(h)), h, "o-", label=f"phi={phi}")

plt.xlabel("迭代步")
plt.ylabel("激活节点比例")
plt.title("4.7 多层阈值级联")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("4_7_cascade.png", dpi=120)
plt.show()
