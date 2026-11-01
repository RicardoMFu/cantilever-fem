"""运行 python checks.py：材料一致切线、屈服、状态与弹性梁基准。"""
import numpy as np
from material import update
from fem import solve, strain_tensor


def check():
    zero = np.zeros((3, 3))
    for e in [[1e-5, -2e-6, 3e-6], [.003, -.0004, .001]]:
        e = np.array(e)
        s, p, a, C = update(strain_tensor(e), zero, 0.)
        for j in range(3):
            shift = np.eye(3)[j]*1e-8
            plus = update(strain_tensor(e+shift), zero, 0.)[0]
            minus = update(strain_tensor(e-shift), zero, 0.)[0]
            ds = (plus-minus)/2e-8
            np.testing.assert_allclose(C[:, j], ds[[0,1,0],[0,1,1]], rtol=1e-7, atol=100.)
        np.testing.assert_allclose(C, C.T, rtol=1e-12, atol=1e-4)
        np.testing.assert_allclose(np.trace(p), 0., atol=1e-15)
        if a > 0:
            dev = s-np.trace(s)*np.eye(3)/3
            np.testing.assert_allclose(np.sqrt(1.5*np.sum(dev*dev)),250e6+1e9*a)
            np.testing.assert_allclose(update(strain_tensor(e),p,a)[2],a,atol=1e-15)
    _, _, history, _ = solve(maximum=.1, steps=2, yield_stress=1e15)
    np.testing.assert_allclose(history[-1,1],2347294.585790664,rtol=1e-8)
    assert history[-1,2] == 0
    print('通过：弹塑性一致切线、对称性、屈服面、塑性不可压缩、状态提交、弹性梁反力回归。')


if __name__ == '__main__':
    check()
