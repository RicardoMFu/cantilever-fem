"""规则Q4悬臂梁：2×2积分、稀疏组装、位移控制Newton迭代。单位m/N/Pa。"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from material import update


def mesh(nx, ny):
    nodes = np.array([(x, y) for y in np.linspace(0, 1, ny + 1)
                      for x in np.linspace(0, 20, nx + 1)])
    elements = np.array([[n, n+1, n+nx+2, n+nx+1]
                         for j in range(ny) for i in range(nx)
                         for n in [j*(nx+1)+i]])
    dofs = np.stack((2*elements, 2*elements+1), axis=-1).reshape(-1, 8)
    return nodes, elements, dofs


def geometry(nodes, elements):
    """预计算所有单元/高斯点的B和面积权重；厚度取1m。"""
    matrices, weights = [], []
    for xi in [-1/np.sqrt(3), 1/np.sqrt(3)]:
        for eta in [-1/np.sqrt(3), 1/np.sqrt(3)]:
            dn = .25*np.array([[-(1-eta), -(1-xi)], [1-eta, -(1+xi)],
                               [1+eta, 1+xi], [-(1+eta), 1-xi]])
            J = np.einsum('eai,aj->eij', nodes[elements], dn)
            det = np.linalg.det(J)
            if np.any(det <= 0):
                raise ValueError('单元退化或翻转')
            grad = np.einsum('ai,eij->eaj', dn, np.linalg.inv(J))
            B = np.zeros((len(elements), 3, 8))
            B[:, 0, 0::2] = grad[:, :, 0]
            B[:, 1, 1::2] = grad[:, :, 1]
            B[:, 2, 0::2] = grad[:, :, 1]
            B[:, 2, 1::2] = grad[:, :, 0]
            matrices.append(B); weights.append(det)
    return np.stack(matrices, axis=1), np.stack(weights, axis=1)


def strain_tensor(engineering):
    eps = np.zeros(engineering.shape[:-1] + (3, 3))
    eps[..., 0, 0] = engineering[..., 0]
    eps[..., 1, 1] = engineering[..., 1]
    eps[..., 0, 1] = eps[..., 1, 0] = engineering[..., 2]/2
    return eps  # eps_zz=0；并不将sigma_zz或plastic_zz设为0


def solve(nx=8, ny=2, maximum=.5, steps=50, yield_stress=250e6, H=1e9):
    nodes, elements, dofs = mesh(nx, ny)
    B, weight = geometry(nodes, elements)
    ndof = 2*len(nodes)
    left = np.flatnonzero(np.isclose(nodes[:, 0], 0))
    right = np.flatnonzero(np.isclose(nodes[:, 0], 20))*2+1
    fixed = np.unique(np.r_[2*left, 2*left+1, right])
    free = np.setdiff1d(np.arange(ndof), fixed)
    row = np.broadcast_to(dofs[:, :, None], (len(elements), 8, 8)).ravel()
    col = np.broadcast_to(dofs[:, None, :], (len(elements), 8, 8)).ravel()
    plastic = np.zeros((len(elements), 4, 3, 3))
    alpha = np.zeros((len(elements), 4))
    d = np.zeros(ndof)
    history = [[0., 0., 0., 0., 0.]]
    snapshots = {}

    def assemble(displacement):
        strain = strain_tensor(np.einsum('eqij,ej->eqi', B, displacement[dofs]))
        stress, pnew, anew, C = update(strain, plastic, alpha, yield_stress=yield_stress, H=H)
        sv = np.stack((stress[..., 0, 0], stress[..., 1, 1], stress[..., 0, 1]), axis=-1)
        fe = np.einsum('eqai,eqa,eq->ei', B, sv, weight)
        internal = np.zeros(ndof)
        np.add.at(internal, dofs.ravel(), fe.ravel())
        ke = np.einsum('eqai,eqab,eqbj,eq->eij', B, C, B, weight)
        K = coo_matrix((ke.ravel(), (row, col)), shape=(ndof, ndof)).tocsr()
        return internal, K, stress, pnew, anew

    for step, load in enumerate(np.linspace(0, maximum, steps+1)[1:], 1):
        # 自由位移沿用上一步，右端设为本步规定值。
        d[right] = -load
        for iteration in range(35):
            internal, K, stress, pnew, anew = assemble(d)
            scale = max(np.max(np.abs(internal[fixed])), 1.)
            residual = np.max(np.abs(internal[free]))/scale
            if residual < 1e-8:
                break
            correction = spsolve(K[free][:, free], -internal[free])
            # 线搜索仍从上一收敛步的材料状态试算。
            norm = np.linalg.norm(internal[free])
            factor = 1.
            for _ in range(12):
                trial = d.copy(); trial[free] += factor*correction
                trial_internal, *_ = assemble(trial)
                if np.linalg.norm(trial_internal[free]) < norm:
                    d = trial
                    break
                factor /= 2
            else:
                raise RuntimeError(f'{nx}x{ny} step {step}: 线搜索失败')
        else:
            raise RuntimeError(f'{nx}x{ny} step {step}: Newton未收敛')
        # 整体收敛后保存状态。
        plastic, alpha = pnew, anew
        force = -internal[right].sum()
        history.append([load, force, alpha.max(), iteration, residual])
        if np.isclose(load, .1) or step == steps:
            snapshots[f'{load:.3f}'] = dict(d=d.copy(), stress=stress.copy(),
                                          plastic=plastic.copy(), alpha=alpha.copy())
        balance = internal.reshape(-1, 2).sum(axis=0)
        if np.max(np.abs(balance))/max(abs(force), 1.) > 1e-7:
            raise RuntimeError('整体力不平衡')
    return nodes, elements, np.array(history), snapshots
