"""
4.5 相关性与嵌入空间对多层渗流的影响

正例：两层边完全一样（overlap=1） → 退化为单层，转变平滑。
反例：两层边完全独立（overlap=0） → 相互依赖渗流，阈值附近突跳。
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
    """层 2 复制层 1 的 overlap 比例边，其余随机补足"""
    G1 = er_layer(N, c)
    G2 = nx.Graph()
    G2.add_nodes_from(G1.nodes())
    for u, v in G1.edges():
        if random.random() < overlap:
            G2.add_edge(u, v)
    while G2.number_of_edges() < G1.number_of_edges():
        u, v = random.sample(range(N), 2)
        G2.add_edge(u, v)
    return [G1, G2]


N, c = 100, 4.0
p_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]

plt.figure(figsize=(7, 4))
for overlap in [0.0, 1.0]:
    S_list = []
    for p in p_list:
        S = sum(mcgc_simple(duplex_overlap(N, c, overlap), p) for _ in range(5)) / 5
        S_list.append(S)
    plt.plot(p_list, S_list, "o-", label=f"overlap={overlap}")

plt.xlabel("p（初始存活概率）")
plt.ylabel("MCGC 比例 S")
plt.title("4.5 连边重叠对 MCGC 的影响")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("4_5_overlap.png", dpi=120)
plt.show()
