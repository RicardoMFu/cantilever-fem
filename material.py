"""小应变三维J2塑性；张量剪应变输入，工程剪应变切线输出。"""
import numpy as np


def update(strain, plastic, alpha, E=210e9, nu=.3, yield_stress=250e6, H=1e9):
    """对所有高斯点做弹性试算/径向返回，返回应力、新状态及平面应变切线。

    strain,plastic形状(...,3,3)，alpha形状(...)。始终从上一收敛步状态调用。
    H=d(屈服应力)/d(等效塑性应变)，不是单轴应力—总应变曲线的塑性段斜率。
    """
    G = E / (2 * (1 + nu))
    bulk = E / (3 * (1 - 2 * nu))
    eye = np.eye(3)
    ee = strain - plastic
    tr = np.trace(ee, axis1=-2, axis2=-1)
    s = 2 * G * (ee - tr[..., None, None] * eye / 3)
    q = np.sqrt(1.5 * np.sum(s * s, axis=(-2, -1)))
    safe_q = np.maximum(q, 1.)
    active = q > yield_stress + H * alpha
    da = np.maximum(q - yield_stress - H * alpha, 0) / (3 * G + H)
    beta = 1 - 3 * G * da / safe_q
    stress = bulk * tr[..., None, None] * eye + beta[..., None, None] * s
    dp = (1.5 * da / safe_q)[..., None, None] * s

    # 三个单位工程应变扰动：[eps_xx, eps_yy, gamma_xy]。
    directions = np.zeros((3, 3, 3))
    directions[0, 0, 0] = directions[1, 1, 1] = 1
    directions[2, 0, 1] = directions[2, 1, 0] = .5
    columns = []
    for de in directions:
        dq = 3 * G * np.sum(s * de, axis=(-2, -1)) / safe_q
        db = np.where(active, -3 * G * (1 / (3 * G + H) / safe_q - da / safe_q**2) * dq, 0)
        ds = (bulk * np.trace(de) * eye + 2 * G * beta[..., None, None] *
              (de - np.trace(de) * eye / 3) + db[..., None, None] * s)
        columns.append(np.stack((ds[..., 0, 0], ds[..., 1, 1], ds[..., 0, 1]), axis=-1))
    tangent = np.stack(columns, axis=-1)
    return stress, plastic + dp, alpha + da, tangent
