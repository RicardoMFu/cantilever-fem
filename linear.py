"""二维悬臂梁：Q4单元 → 高斯积分 → 组装 → 已知位移消元 → 求解。

单位统一为 m、N、Pa。仅使用 NumPy，不使用有限元库。
运行：python3 linear.py
"""
import numpy as np


def strain_matrix(xy, xi, eta):
    """输入四个节点坐标和一个标准坐标点，返回 B 和 detJ。"""
    # 每行对应一个节点；两列分别为 N 对 xi、eta 的导数。
    dn = 0.25 * np.array([
        [-(1 - eta), -(1 - xi)],
        [1 - eta, -(1 + xi)],
        [1 + eta, 1 + xi],
        [-(1 + eta), 1 - xi],
    ])
    jac = xy.T @ dn                      # 坐标映射的雅可比矩阵
    det_j = np.linalg.det(jac)
    if det_j <= 0:
        raise ValueError("节点应逆时针排列，且单元不能翻转或退化")
    grad = np.linalg.solve(jac.T, dn.T).T  # 每行得到 [N,x, N,y]
    B = np.zeros((3, 8))
    B[0, 0::2] = grad[:, 0]              # 水平正应变
    B[1, 1::2] = grad[:, 1]              # 竖向正应变
    B[2, 0::2] = grad[:, 1]              # 工程剪应变
    B[2, 1::2] = grad[:, 0]              # 工程剪应变，不额外乘 1/2
    return B, det_j


def element_stiffness(xy, D, thickness):
    """四个高斯点贡献相加，得到一个 8×8 单元矩阵。"""
    ke = np.zeros((8, 8))
    points = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    for xi in points:
        for eta in points:
            B, det_j = strain_matrix(xy, xi, eta)
            ke += thickness * (B.T @ D @ B) * det_j  # 两方向权重均为1
    return ke


def displacement_at(nodes, elements, d, x, y):
    """本题轴对齐矩形网格：定位单元，以形函数插值得到任意点位移。

    返回位移[u,v]、单元编号（从1开始）、标准坐标和四个形函数值。
    单元公共边上任选相邻单元，连续Q4位移插值给出相同结果。
    """
    tol = 1e-12
    for e, enodes in enumerate(elements):
        xy = nodes[enodes]
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        if np.all(np.array([x, y]) >= lo - tol) and np.all(np.array([x, y]) <= hi + tol):
            xi, eta = 2 * (np.array([x, y]) - lo) / (hi - lo) - 1
            N = 0.25 * np.array([(1-xi)*(1-eta), (1+xi)*(1-eta),
                                 (1+xi)*(1+eta), (1-xi)*(1+eta)])
            return N @ d.reshape(-1, 2)[enodes], e + 1, (xi, eta), N
    raise ValueError("指定点在梁的计算区域之外")


def solve_beam(nx=8, ny=2, thickness=1.0):
    """默认求解课堂推导的8×2网格；厚度明确取1 m。"""
    L, h, E, nu = 20.0, 1.0, 210e9, 0.30
    D = E / ((1 + nu) * (1 - 2 * nu)) * np.array([
        [1 - nu, nu, 0],
        [nu, 1 - nu, 0],
        [0, 0, (1 - 2 * nu) / 2],
    ])  # 平面应变本构；应变顺序 [epsilon_xx, epsilon_yy, gamma_xy]

    # 生成节点：先底行从左向右，再中间行、顶行。
    nodes = np.array([(x, y) for y in np.linspace(0, h, ny + 1)
                      for x in np.linspace(0, L, nx + 1)])
    elements = []
    for j in range(ny):
        for i in range(nx):
            n = j * (nx + 1) + i
            elements.append([n, n + 1, n + nx + 2, n + nx + 1])
    elements = np.array(elements)         # 局部顺序：左下、右下、右上、左上

    # 计算单元刚度并组装：Python编号从0开始，课堂编号从1开始。
    ndof = 2 * len(nodes)
    K = np.zeros((ndof, ndof))
    for enodes in elements:
        ke = element_stiffness(nodes[enodes], D, thickness)
        dofs = np.column_stack((2 * enodes, 2 * enodes + 1)).ravel()
        K[np.ix_(dofs, dofs)] += ke       # 共享自由度的贡献在原位置相加

    # 已知位移：左边u=v=0；右边仅v=-0.1，u保持自由。
    left = np.flatnonzero(np.isclose(nodes[:, 0], 0))
    right = np.flatnonzero(np.isclose(nodes[:, 0], L))
    fixed = np.sort(np.r_[2 * left, 2 * left + 1, 2 * right + 1])
    free = np.setdiff1d(np.arange(ndof), fixed)
    d = np.zeros(ndof)
    d[2 * right + 1] = -0.1

    # 提取自由位移系统（8×2网格为45阶）；不显式求矩阵逆。
    Kff = K[np.ix_(free, free)]
    rhs = -K[np.ix_(free, fixed)] @ d[fixed]
    d[free] = np.linalg.solve(Kff, rhs)   # LAPACK一般线性系统直接解法（LU）

    # 回代原始方程：自由位置应接近0，约束位置得到反力。
    reaction = K @ d                    # 本题已知外力向量为0
    force_scale = max(np.max(np.abs(reaction[fixed])), 1.0)
    rotation = np.column_stack((-nodes[:, 1], nodes[:, 0])).ravel()
    rigid = np.column_stack((np.tile([1., 0.], len(nodes)),
                             np.tile([0., 1.], len(nodes)), rotation))
    checks = {
        "nodes": len(nodes), "elements": len(elements), "free_dofs": len(free),
        "symmetry_relative": float(np.linalg.norm(K - K.T) / np.linalg.norm(K)),
        "free_residual_relative": float(np.max(np.abs(reaction[free])) / force_scale),
        "rigid_modes_relative": float(np.linalg.norm(K @ rigid) /
                                      (np.linalg.norm(K) * np.linalg.norm(rigid))),
        "force_balance_relative": float(np.max(np.abs(reaction.reshape(-1, 2).sum(axis=0))) /
                                        force_scale),
        "moment_balance_relative": float(abs(rotation @ reaction) / (force_scale * L)),
        "right_vertical_reaction_N": float(reaction[2 * right + 1].sum()),
        "strain_energy_J": float(0.5 * d @ K @ d),
    }
    np.linalg.cholesky(Kff)              # 核查施加约束后的刚度正定
    for key in ("symmetry_relative", "free_residual_relative", "rigid_modes_relative",
                "force_balance_relative", "moment_balance_relative"):
        assert checks[key] < 1e-8, (key, checks[key])
    return nodes, elements, K, D, d, free, fixed, reaction, checks


if __name__ == "__main__":
    for nx, ny in [(8, 2), (16, 4), (32, 8)]:
        nodes, elements, K, D, d, free, fixed, reaction, checks = solve_beam(nx, ny)
        print(f"{nx}x{ny}: {checks}")
