import torch
from chapter_0 import er_layer

torch.manual_seed(0)

# ============================================================
# 第2章：多层网络表示
# M 层，每层 N 个节点，A_layers[α] 是第 α 层邻接矩阵
# ============================================================
M, N, c = 2, 10, 4.0
A_layers = torch.stack([er_layer(N, c) for _ in range(M)])  # [M, N, N]
print("A_layers shape:", A_layers.shape)
print(A_layers)
print("=" * 10)

# ============================================================
# 第2章：度向量 k_i = (k_i^[1], ..., k_i^[M])
# ============================================================
k = A_layers.sum(dim=2).T  # [N, M]
print("度向量 k，形状 [N, M]:\n", k)
print("=" * 10)

# ============================================================
# 第2章：重叠度 o_i = sum_alpha k_i^[alpha]
# ============================================================
overlapping_degree = k.sum(dim=1)  # [N]
print("重叠度 o_i:", overlapping_degree)
print("=" * 10)

# ============================================================
# 第2章：投影网络 proj(M)
# 任一层有边，投影网络就有边
# ============================================================
A_proj = (A_layers.sum(dim=0) > 0).float()
print("投影网络边数:", A_proj.sum().item() / 2)
print("=" * 10)

# ============================================================
# 第2章：层间重叠 O^{αβ}
# 无向图只统计 i<j，避免重复计数
# ============================================================
O_01 = (A_layers[0] * A_layers[1]).triu(diagonal=1).sum()
print("层0和层1共同边数 O^{0,1}:", O_01.item())
print("=" * 10)

# ============================================================
# 第2章：超邻接矩阵 supra-adjacency matrix
# 对角块：层内连接；非对角块：副本连接 I_N
# ============================================================
supra = torch.zeros(M * N, M * N)
for a in range(M):
    supra[a * N : (a + 1) * N, a * N : (a + 1) * N] = A_layers[a]

for a in range(M):
    for b in range(M):
        if a != b:
            supra[a * N : (a + 1) * N, b * N : (b + 1) * N] = torch.eye(N)

print("supra shape:", supra.shape)
print("=" * 10)

# ============================================================
# 第2章：超拉普拉斯与谱性质
# L = D - A，第二小特征值 lambda2
# ============================================================
deg_supra = supra.sum(dim=1)
L_supra = torch.diag(deg_supra) - supra
eigvals = torch.linalg.eigvalsh(L_supra)
print("第二小特征值 lambda2:", eigvals[1].item())
