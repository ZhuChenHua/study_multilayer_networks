"""
4.6.2 对抗网络的渗流

两层节点互相"对抗"：
  节点 i 在 A 层活跃 ⇔ A 层有活跃邻居，且 B 层中 i 的邻居都不活跃。
  节点 i 在 B 层活跃 ⇔ B 层有活跃邻居，且 A 层中 i 的邻居都不活跃。

用消息传递在边上迭代：σ_α[i→j] 表示去掉边 (i,j) 后 i 是否在层 α 的渗流簇中。
可能出现双稳：要么 A 层渗流、B 层不渗流，要么反过来。
"""

import random
import networkx as nx
import matplotlib.pyplot as plt
from giant import er_layer

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def update_messages(G_self, G_other, m_self, m_other, s):
    """
    更新一层的所有消息：
      σ[i→j] = s[i] · [1 - ∏_{k∈N_self(i)\\j}(1-σ_self[k→i])] · ∏_{k∈N_other(i)}(1-σ_other[k→i])
    """
    new = {}
    for i in G_self.nodes():
        for j in G_self.neighbors(i):
            # 本层：至少一个别的邻居能到 i
            p_self = 1.0
            for k in G_self.neighbors(i):
                if k != j:
                    p_self *= 1 - m_self.get((k, i), 0)
            # 另一层：i 的邻居全部不到 i（对抗条件）
            p_other = 1.0
            for k in G_other.neighbors(i):
                p_other *= 1 - m_other.get((k, i), 0)
            new[i, j] = s[i] * (1 - p_self) * p_other
    return new


def antagonistic(G_A, G_B, p, max_iter=100):
    """对抗网络的渗流簇比例 S_A（消息传递版）"""
    N = G_A.number_of_nodes()
    s = {i: 1 if random.random() < p else 0 for i in range(N)}  # 1 = 未损坏

    # 消息初始化：每条有向边上的消息为 1
    m_A = {(i, j): 1 for i, j in G_A.edges()}
    m_A.update({(j, i): 1 for i, j in G_A.edges()})
    m_B = {(i, j): 1 for i, j in G_B.edges()}
    m_B.update({(j, i): 1 for i, j in G_B.edges()})

    # 迭代到收敛
    for _ in range(max_iter):
        new_A = update_messages(G_A, G_B, m_A, m_B, s)
        new_B = update_messages(G_B, G_A, m_B, m_A, s)
        if new_A == m_A and new_B == m_B:
            break
        m_A, m_B = new_A, new_B

    # 计算 S_A：i 在 A 层活跃 ⇔ A 层有活跃邻居，且 B 层 i 的邻居都不活跃
    S_A = 0
    for i in range(N):
        p_A = 1.0
        for k in G_A.neighbors(i):
            p_A *= 1 - m_A.get((k, i), 0)
        p_B = 1.0
        for k in G_B.neighbors(i):
            p_B *= 1 - m_B.get((k, i), 0)
        S_A += s[i] * (1 - p_A) * p_B
    return S_A / N


if __name__ == "__main__":
    random.seed(0)
    N, c, n_avg = 500, 3.0, 5
    p_list = [0.1 + 0.8 * i / 14 for i in range(15)]

    S_list = []
    for p in p_list:
        S_avg = (
            sum(antagonistic(er_layer(N, c), er_layer(N, c), p) for _ in range(n_avg))
            / n_avg
        )
        S_list.append(S_avg)

    plt.figure(figsize=(8, 5))
    plt.plot(p_list, S_list, "o-", label="A 层渗流簇比例 $S_A$", markersize=4)
    plt.xlabel("p")
    plt.ylabel("$S_A$")
    plt.title("4.6.2 对抗网络的渗流")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
