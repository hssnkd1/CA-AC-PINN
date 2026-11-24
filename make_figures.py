"""Generate publication-size plots using only the saved numerical records."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
from acpinn import CircularDictionary,fixed_circle_indices
from problems import Poisson,Interface

ROOT=Path(__file__).resolve().parent
R=ROOT/'results';F=ROOT/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
                     'axes.labelsize':9,'legend.fontsize':8,'axes.spines.top':False,
                     'axes.spines.right':False,'savefig.facecolor':'white'})
BLUE='#176B92';ORANGE='#CE6B25';GREEN='#26825F';PURPLE='#8060A5'

def load(name):return json.loads((R/name).read_text())
def save(fig,name):
    fig.savefig(F/(name+'.png'),dpi=330,bbox_inches='tight')
    fig.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def main():
    cases=[load('poisson.json'),load('interface.json')]
    fig,ax=plt.subplots(2,2,figsize=(7.2,4.4),sharex='col')
    for j,data in enumerate(cases):
        n=[a['size'] for a in data['adaptive']]
        for i,key in enumerate(['relative_l2','relative_energy']):
            ax[i,j].loglog(n,[a[key] for a in data['adaptive']],'-o',color=BLUE,label='Adaptive circles')
            ax[i,j].scatter([129],[data['fixed'][key]],marker='s',color=ORANGE,label='Fixed circles',zorder=5)
            ax[i,j].grid(alpha=.2,which='both');ax[i,j].set_ylabel('Relative '+('L2 error' if i==0 else 'energy error'))
        ax[0,j].set_title('Smooth Poisson' if j==0 else 'Diffusion interface')
        ax[1,j].set_xlabel('Active coefficients')
    ax[0,0].legend();fig.tight_layout();save(fig,'figure_1_elliptic_convergence')

    p=Interface();d=CircularDictionary(2,3);fd=fixed_circle_indices(d,11)
    xx=np.linspace(0,1,201);gx,gy=np.meshgrid(xx,xx,indexing='xy');pts=np.column_stack([gx.ravel(),gy.ravel()])
    z,_=p.mapping(pts);state=np.load(R/'interface_adaptive_129.npz');fixed=np.load(R/'interface_fixed.npz')
    u=p.exact(pts)[0];ua=d.predict(z,state['ids'],state['c']);uf=fd.predict(z,range(fd.size),fixed['c'])
    fig,ax=plt.subplots(2,2,figsize=(7.2,6.0),constrained_layout=True)
    fields=[u,ua,uf,np.abs(ua-u)];titles=['Exact solution','Adaptive circles 129','Fixed circles 129','Adaptive absolute error']
    for k,a in enumerate(ax.ravel()):
        image=a.pcolormesh(gx,gy,fields[k].reshape(gx.shape),shading='auto',cmap='viridis' if k<3 else 'magma',vmin=u.min() if k<3 else 0,vmax=u.max() if k<3 else None,rasterized=True)
        a.axvline(.5,color='white',lw=.9,ls='--');a.set_aspect('equal');a.set_title(titles[k]);a.set_xlabel('x');a.set_ylabel('y');fig.colorbar(image,ax=a,shrink=.8)
    save(fig,'figure_2_interface_fields')

    # Support geometry is plotted in the coordinates in which circles are defined.
    cp=pts.copy();threshold=.5/p.s
    cp[:,0]=np.where(pts[:,0]<=threshold,pts[:,0]*p.s,.5+(pts[:,0]*p.s-.5)*p.contrast)
    background=p.exact(cp)[0].reshape(gx.shape)
    fig,ax=plt.subplots(1,3,figsize=(7.3,2.75),constrained_layout=True)
    for a,num in zip(ax,[17,65,129]):
        st=np.load(R/f'interface_adaptive_{num}.npz');ids=st['ids'];ii=ids[ids>=d.ng]-d.ng
        a.pcolormesh(gx,gy,background,shading='auto',cmap='Greys',alpha=.35,rasterized=True)
        for k in ii:
            color=BLUE if d.radii[k]>.17 else ORANGE
            a.add_patch(Circle(d.centers[k],d.radii[k],fill=False,edgecolor=color,lw=.5,alpha=.4))
        a.scatter(d.centers[ii,0],d.centers[ii,1],s=4,color=BLUE)
        a.set(xlim=(0,1),ylim=(0,1),xlabel='Transformed x coordinate',ylabel='y',title=f'{num} coefficients');a.set_aspect('equal')
    save(fig,'figure_3_selected_supports')

    fig,ax=plt.subplots(1,2,figsize=(7.3,3.25),constrained_layout=True)
    for a,data,name in zip(ax,cases,['poisson','interface']):
        a.loglog([r['seconds'] for r in data['adaptive']],[r['relative_l2'] for r in data['adaptive']],'-o',color=BLUE,label='Adaptive circles')
        for key,marker,col,label in [('fixed','s',ORANGE,'Fixed Ritz'),('strong','D',PURPLE,'Fixed strong')]:
            row=data[key];a.scatter(row['seconds'],row['relative_l2'],marker=marker,color=col,label=label,zorder=5)
        a.loglog([r['seconds'] for r in data['fem']],[r['relative_l2'] for r in data['fem']],'-^',color=GREEN,label='P1 FEM')
        pins=load(name+'_pinn.json')
        for method,marker,col in [('tanh PINN','x','#222222'),('RAR tanh PINN','+','#B2314B'),('FD gPINN','*','#76633C')]:
            rows=[r for r in pins if r['method']==method]
            a.scatter(np.mean([r['seconds'] for r in rows]),np.mean([r['relative_l2'] for r in rows]),marker=marker,color=col,s=55,label=method,zorder=6)
        a.set(title='Smooth Poisson' if name=='poisson' else 'Diffusion interface',xlabel='Construction time seconds',ylabel='Relative L2 error');a.grid(alpha=.2,which='both')
    ax[1].legend(loc='center left',bbox_to_anchor=(1.01,.5),frameon=False);save(fig,'figure_4_error_work')

    fig,ax=plt.subplots(2,2,figsize=(7.2,5.0),constrained_layout=True)
    for j,(name,title,window) in enumerate([('allen_cahn','Forced Allen-Cahn',(.40,.56)),('burgers','Viscous Burgers',(.34,.48))]):
        st=np.load(R/f'{name}_adaptive_0.002.npz');fx=np.load(R/f'{name}_fixed_0.002.npz')
        ax[0,j].plot(st['x'],st['exact'],color='black',lw=1.6,label='Exact')
        ax[0,j].plot(fx['x'],fx['u'],color=ORANGE,lw=1,label='Fixed')
        ax[0,j].plot(st['x'],st['u'],'--',color=BLUE,lw=1.2,label='Adaptive')
        ax[0,j].set(title=title,xlabel='x',ylabel='u at final time');ax[0,j].grid(alpha=.2)
        ax[1,j].semilogy(fx['x'],np.maximum(np.abs(fx['u']-fx['exact']),1.e-9),color=ORANGE,label='Fixed')
        ax[1,j].semilogy(st['x'],np.maximum(np.abs(st['u']-st['exact']),1.e-9),color=BLUE,label='Adaptive')
        ax[1,j].set(xlim=window,xlabel='x near front',ylabel='Absolute error');ax[1,j].grid(alpha=.2)
    ax[0,0].legend();save(fig,'figure_5_moving_fronts')

    evo=load('evolution.json');fig,ax=plt.subplots(1,3,figsize=(7.4,2.8),constrained_layout=True)
    for a,case,title in zip(ax,['allen_cahn','burgers','transient'],['Allen-Cahn','Burgers','Transient diffusion']):
        for method,col,mark in [('Adaptive circular',BLUE,'o'),('Fixed circular',ORANGE,'s'),('Conservative FDM' if case=='transient' else 'Central FDM',GREEN,'^')]:
            rows=sorted([r for r in evo if r['case']==case and r['method']==method],key=lambda r:r['dt'])
            a.loglog([r['dt'] for r in rows],[r['relative_l2'] for r in rows],'-'+mark,color=col,label='FDM' if 'FDM' in method else method)
        a.set(title=title,xlabel='Time step',ylabel='Final relative L2 error');a.grid(alpha=.2,which='both')
        tickvalues=[.0025,.005,.01] if case=='transient' else [.001,.002,.004]
        a.set_xticks(tickvalues,[f'{t:g}' for t in tickvalues]);a.xaxis.set_minor_formatter(NullFormatter())
        a.tick_params(axis='x',labelsize=7)
    handles,labels=ax[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.14),ncol=3,frameon=False,fontsize=8)
    save(fig,'figure_6_time_refinement')
    print('Generated six figure sets from saved results.')

if __name__=='__main__':main()
