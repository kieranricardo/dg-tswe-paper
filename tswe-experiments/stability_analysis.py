from matplotlib import pyplot as plt
from dg_tswe.utils import Interpolate
from dg_tswe.dg_cubed_sphere_tswe import DGCubedSphereTSWE
from dg_tswe.dg_cubed_sphere_tswe_energy_cons import EngConsDGCubedSphereTSWE
from dg_tswe.dg_cubed_sphere_twse_entropy_cons import EntConsDGCubedSphereTSWE
from dg_tswe.cubed_sphere_swe import CubedSphereSWE
import numpy as np
import scipy
import os

if not os.path.exists('./plots'): os.makedirs('./plots')
if not os.path.exists('./data'): os.makedirs('./data')

plt.rcParams['font.size'] = '12'

mode = 'plot'
dev = 'cpu'

nx = ny = 64

exp_names = [f'EntCons_DG_cntr_res_6x{nx}x{ny}', f'EngCons_DG_cntr_res_6x{nx}x{ny}']
labels = ['Entropy cons.', 'Energy cons.']
for exp, label in list(zip(exp_names, labels))[1:]:
    fn_template = f"{exp}_day_{20}.npy"
    diagnostics = np.load(os.path.join('data', f"diagnostics_{fn_template}"))
    times = diagnostics[0] / (24 * 3600)
    energy = diagnostics[1]
    entropy = diagnostics[2]
    buoyancy = diagnostics[3]
    mass = diagnostics[4]

    plt.figure(1, figsize=(7, 4))

    tunit = ' (days)'
    plt.suptitle("Conservation errors")

    ax = plt.subplot(1, 2, 1)
    ax.set_ylabel("Energy error (normalized)")
    ax.set_xlabel("Time" + tunit)
    ax.plot(times, (energy - energy[0]) / energy[0], label=label)
    ax.set_yscale('symlog', linthresh=1e-15)
    ax.grid(True, which='both')

    ax = plt.subplot(1, 2, 2)
    ax.set_ylabel("Entropy error (normalized)")
    ax.set_xlabel("Time" + tunit)
    ax.plot(times, (entropy - entropy[0]) / entropy[0], label=label)
    ax.set_yscale('symlog', linthresh=1e-15)
    ax.grid(True, which='both')

    # plt.legend()
    plt.tight_layout()

plt.savefig(f'./plots/galewsky_stability.png')