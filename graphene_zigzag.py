import numpy as np
import scipy
from matplotlib import pyplot as plt
import kwant
import kwant.continuum


honeycomb = kwant.lattice.honeycomb()
a, b = honeycomb.sublattices
nnn_hoppings_a = (((-1, 0), a, a), ((0, 1), a, a), ((1, -1), a, a))
nnn_hoppings_b = (((1, 0), b, b), ((0, -1), b, b), ((-1, 1), b, b))


def title(p):
    return fr"$t={p['t']:.2}$, $t_2={p['t_2']:.2}$, $M={p['M']:.2}$"


def onsite(site, M):
    return M if site.family == a else -M


def nn_hopping(site1, site2, t):
    return t


def nnn_hopping(site1, site2, t_2):
    return 1j * t_2


haldane_infinite = kwant.Builder(kwant.TranslationalSymmetry(*honeycomb.prim_vecs))
haldane_infinite[honeycomb.shape(lambda pos: True, (0, 0))] = onsite
haldane_infinite[honeycomb.neighbors()] = nn_hopping
haldane_infinite[
    [kwant.builder.HoppingKind(*hopping) for hopping in nnn_hoppings_a]
] = nnn_hopping
haldane_infinite[
    [kwant.builder.HoppingKind(*hopping) for hopping in nnn_hoppings_b]
] = nnn_hopping


p = dict(t=1.0, t_2=0.0, M=0.0, phi=np.pi/2)
k = (4/3)*np.linspace(-2*np.pi, 2*np.pi, 150)


infinite_bulk=kwant.wraparound.wraparound(haldane_infinite).finalized()

#Finds out the bulk energies for k_y=0
edz=[]
for kx in k:
    p['k_x']=kx
    p['k_y']=0
    ham=infinite_bulk.hamiltonian_submatrix(params=p)
    energies,eigvec=scipy.linalg.eigh(ham)
    edz.append(energies)
plt.plot(k,edz,color='tab:red')

#k_y is not a good quantum number
W = 20
def ribbon_shape_zigzag(site):
    return -0.5 / np.sqrt(3) - 0.1 <= site.pos[1] < np.sqrt(3) * W / 2 + 0.01

#zigzag_ribbon=kwant.Builder()
zigzag_ribbon = kwant.Builder(kwant.TranslationalSymmetry((1, 0)))
zigzag_ribbon.fill(haldane_infinite, ribbon_shape_zigzag, (0, 0))

zigzag_ribbon=kwant.wraparound.wraparound(zigzag_ribbon).finalized()

k = np.linspace(0, 2*np.pi, 101)

#Introduce complex next nearest neighbour hopping in honeycomb lattice
a=0.03
p['t_2']=a
p['t_2i']=-a
e=[]
for kx in k:
    p['k_x']=kx
    ham=zigzag_ribbon.hamiltonian_submatrix(params=p)
    energies,eigvec=scipy.linalg.eigh(ham)
    e.append(energies) #save the eigenvalues in the list e

#Plot e along the y axis for each value of k in x-axis
plt.plot(k,e,color='tab:blue')
plt.ylabel('$E/t$',fontsize=20)
plt.xlabel('$k_x$',fontsize=20)
plt.ylim(-1,1)
plt.yticks(ticks=np.linspace(-1,1,3),fontsize=15)
plt.xticks(ticks=np.linspace(0,2*np.pi,3),labels=['0','$\pi$','$2\pi$'],fontsize=15)
plt.tight_layout()
plt.savefig('zigzag_ribbon_qsh.png',dpi=500)


