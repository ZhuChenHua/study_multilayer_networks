"""
4.6 其他渗流问题

正例：所有边（层内 + 层间）都只提供连通性 → 经典渗流，平滑。
反例：层间是"依赖"而非"连通" → 相互依赖渗流，突跳。
"""

import random
import networkx as nx
from matplotlib import pyplot as plt
from giant import er_layer
from mcgc import mcgc_simple

random.seed(0)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def classical_percolation(Gs, p):
    """把所有边（层内 + 层间依赖）都当成普通边，求巨分量"""
    M, N = len(Gs), Gs[0].number_of_nodes()
    big = nx.Graph()
    for a, G in enumerate(Gs):
        for u, v in G.edges():
            big.add_edge((a, u), (a, v))
    for a in range(M):
        for b in range(a + 1, M):
            for i in range(N):
                big.add_edge((a, i), (b, i))
    active = [n for n in big.nodes() if random.random() < p]
    sub = big.subgraph(active)
    if len(sub) == 0:
        return 0.0
    return len(max(nx.connected_components(sub), key=len)) / (M * N)


M, N, c = 2, 100, 4.0
Gs = [er_layer(N, c) for _ in range(M)]
p_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]

cl, mc = [], []
for p in p_list:
    cl.append(sum(classical_percolation(Gs, p) for _ in range(5)) / 5)
    mc.append(sum(mcgc_simple(Gs, p) for _ in range(5)) / 5)

plt.figure(figsize=(7, 4))
plt.plot(p_list, cl, "o-", label="经典渗流（边=连通）")
plt.plot(p_list, mc, "s-", label="MCGC（边=依赖）")
plt.xlabel("p")
plt.ylabel("S")
plt.title("4.6 经典渗流 vs MCGC")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("4_6_classical_vs_mcgc.png", dpi=120)
plt.show()
