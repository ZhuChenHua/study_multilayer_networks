import torch

"""
ER 随机图：
    有 N 个节点，每两个节点之间有相同概率 p 连接
平均度：
    所有节点的度数的平均值 c，在 ER 图中，每个节点最多连 N-1 条边，每条边概率为 p，所以期望平均度 c = p × (N - 1)
"""


def er_layer(N, c):
    """
    静态生成模型 -- ER 随机图。生成一层 ER 网络，平均度约为 c
    Args:
        N: 节点数
        c: 平均度
    Returns:
        A: 无向邻接矩阵，形状为 [N, N]
    """

    p = c / (N - 1)
    A = (torch.rand(N, N) < p).float()  # 生成随机矩阵，小于 p 的位置为 1，否则为 0
    A = torch.triu(A, diagonal=1)  # 取上三角矩阵，对角线为 0
    return A + A.t()  # 对称化，得到无向图


if __name__ == "__main__":
    torch.manual_seed(0)
    N = 4
    c = 2
    A = er_layer(N, c)
    print(A)
