"""python run.py：三网格非线性实验及图表，可重复生成results/。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from fem import solve


def main():
    out = Path(__file__).resolve().parent/'results'
    out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7,4), layout='constrained')
    summary = []
    deflection, dax = plt.subplots(figsize=(7,3.4),layout='constrained')
    for nx,ny in [(8,2),(16,4),(32,8)]:
        nodes,elements,hist,snaps=solve(nx,ny)
        np.savetxt(out/f'history_{nx}x{ny}.csv',hist,delimiter=',',
                   header='tip_displacement_m,force_N,max_equivalent_plastic_strain,Newton_iterations,relative_residual',comments='')
        for load,snap in snaps.items():
            np.savez(out/f'state_{nx}x{ny}_{load}.npz',nodes=nodes,elements=elements,**snap)
        ax.plot(hist[:,0],hist[:,1]/1000,label=f'{nx} × {ny}')
        # 该网格的线弹性参考斜率由最初弹性加载步得到。
        ax.plot(hist[:,0],hist[:,0]*hist[1,1]/hist[1,0]/1000,'--',alpha=.4)
        original=hist[np.isclose(hist[:,0],.1)][0]
        # 有历史变量时，不在边界点临时创建新的材料状态。
        # 将四个高斯点应力双线性外推至公共顶点，再取相邻两单元均值。
        g=1/np.sqrt(3)
        basis=np.array([[1,x,y,x*y] for x in [-g,g] for y in [-g,g]])
        sides=[]
        for en in [nx*(ny-1)+nx//2-1,nx*(ny-1)+nx//2]:
            xy=nodes[elements[en]];natural=2*(np.array([10.,1.])-xy.min(0))/(xy.max(0)-xy.min(0))-1
            x,y=natural
            coeff=np.linalg.solve(basis,snaps['0.500']['stress'][en,:,0,0])
            sides.append(float(np.array([1,x,y,x*y])@coeff/1e6))
        mid=np.flatnonzero(np.isclose(nodes[:,1],.5))
        dax.plot(nodes[mid,0],snaps['0.500']['d'][2*mid+1]*1000,label=f'{nx} × {ny}')
        item=dict(mesh=f'{nx}x{ny}',force_01_kN=original[1]/1000,
                  alpha_01=original[2],force_05_kN=hist[-1,1]/1000,
                  alpha_05=hist[-1,2],max_iterations=int(hist[:,3].max()),
                  max_residual=float(hist[:,4].max()),point_stress_sides_MPa=sides,
                  point_stress_mean_MPa=float(np.mean(sides)),
                  first_plastic_displacement_m=float(hist[hist[:,2]>1e-10,0][0]) if np.any(hist[:,2]>1e-10) else None)
        summary.append(item);print(item,flush=True)
        if nx==32:
            end=snaps['0.500']
            cloud,axes=plt.subplots(2,1,figsize=(8,4.4),layout='constrained')
            for panel,value,label,cmap in [(axes[0],end['stress'][...,0,0].mean(1)/1e6,r'$\sigma_{xx}$ / MPa','RdBu_r'),
                                            (axes[1],end['alpha'].mean(1),r'$\overline{\varepsilon}^{p}$','YlOrRd')]:
                pc=PolyCollection(nodes[elements],array=value,cmap=cmap,edgecolors='none')
                if cmap=='RdBu_r':pc.set_clim(-abs(value).max(),abs(value).max())
                panel.add_collection(pc);panel.set(xlim=(0,20),ylim=(0,1),xlabel='$x$ / m',ylabel='$y$ / m')
                cloud.colorbar(pc,ax=panel,label=label)
            cloud.savefig(out/'plastic_cloud.pdf');plt.close(cloud)
    ax.set(xlabel='Downward tip displacement / m',ylabel='Downward reaction magnitude / kN')
    ax.grid(alpha=.2);ax.legend(frameon=False);fig.savefig(out/'force_displacement.pdf');plt.close(fig)
    x=np.linspace(0,20,301)
    dax.plot(x,-.5*x*x*(60-x)/(2*20**3)*1000,'k--',label='Elastic E–B')
    dax.set(xlabel='$x$ / m',ylabel='Deflection / mm');dax.legend(frameon=False);dax.grid(alpha=.2)
    deflection.savefig(out/'nonlinear_deflection.pdf');plt.close(deflection)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
