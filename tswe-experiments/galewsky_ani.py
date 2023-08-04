from matplotlib import pyplot as plt
from dg_tswe.utils import Interpolate
from dg_tswe.dg_cubed_sphere_tswe import DGCubedSphereTSWE
from dg_tswe.dg_cubed_sphere_tswe_energy_cons import EngConsDGCubedSphereTSWE
from dg_tswe.dg_cubed_sphere_twse_entropy_cons import EntConsDGCubedSphereTSWE
from dg_tswe.cubed_sphere_swe import CubedSphereSWE
import numpy as np
import scipy
import os
import matplotlib.ticker as ticker
from matplotlib.animation import FFMpegWriter as MovieWriter

if not os.path.exists('./plots'): os.makedirs('./plots')
if not os.path.exists('./data'): os.makedirs('./data')
if not os.path.exists('./data'): os.makedirs('./videos')

plt.rcParams['font.size'] = '12'
plt.rcParams.update({'figure.autolayout': True})

mode = 'run'
dev = 'cpu'

nx = ny = 32
eps = 0.8
g = 9.80616
f = 7.292e-5
radius = 6.37122e6
poly_order = 3

u_0 = 80
h_0 = 10_000

def plot_sol(solver, solver_hr, sw_solver, ax1, ax2):


    def interpolate_plot_func(s, sol):
        data = plot_func(sol.faces[s.name])
        return interpolator.torch_interpolate(data)

    vmin = 9.5
    vmax = 10.6
    plot_func = lambda s: s.b_H1()
    im = solver_hr.triangular_plot(ax1, vmin=vmin, vmax=vmax, latlong=False, plot_func=lambda s: interpolate_plot_func(s, solver))
    plt.colorbar(im[0], format=ticker.FuncFormatter(fmt))

    vmin = -0.00015;
    vmax = 0.00015
    plot_func = lambda s: s.vorticity() - s.f
    for face in sw_solver.faces.values():
        fc = solver.faces[face.name]
        face.set_initial_condition(fc.u.numpy(), fc.v.numpy(), fc.w.numpy(), fc.h.numpy())
    sw_solver.boundaries()

    im = solver_hr.triangular_plot(ax2, vmin=vmin, vmax=vmax, latlong=False, plot_func=lambda s: interpolate_plot_func(s, sw_solver))
    plt.colorbar(im[0], format=ticker.FuncFormatter(fmt))

def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def initial_condition(face):

    def zonal_flow(lat):
        lat_0 = np.pi / 7
        lat_1 = 0.5 * np.pi - lat_0

        e_n = np.exp(-4 / (lat_1 - lat_0) ** 2)

        out = np.zeros_like(lat)
        mask = (lat_0 < lat) & (lat < lat_1)
        out[mask] = (u_0 / e_n) * np.exp(1 / ((lat[mask] - lat_0) * (lat[mask] - lat_1)))
        return out


    def func(lat):
        u_ = zonal_flow(lat)
        out = -radius * u_ * (2 * np.sin(lat) * f + np.tan(lat) * u_ / radius)

        return out / g

    lats = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 100_000)
    dlat = np.diff(lats).mean()
    vals = func(lats)
    h_reg = h_0 + (np.cumsum(vals) - 0.5 * (vals[0] + vals[-1])) * dlat
    h_interp = scipy.interpolate.interp1d(lats, h_reg)


    lat, long = face.geometry.lat_long(face.xs, face.ys, face.zs)
    lat_vec_x, lat_vec_y, lat_vec_z, long_vec_x, long_vec_y, long_vec_z = face.geometry.lat_long_vecs(face.xs, face.ys, face.zs)
    h = h_interp(lat)

    alpha = 1 / 3
    beta = 1 / 15
    lat_2 = np.pi / 4
    h_pert = 120 * np.cos(lat) * np.exp(-(long / alpha) ** 2) * np.exp(-((lat_2 - lat) / beta) ** 2)
    h += h_pert

    u_ = zonal_flow(lat)
    u = long_vec_x * u_
    v = long_vec_y * u_
    w = long_vec_z * u_

    b_pert = np.cos(lat) * np.exp(-(long / alpha) ** 2) * np.exp(-((lat_2 - lat) / beta) ** 2)
    hb = h * (g + b_pert)

    return u, v, w, h, hb



solver = DGCubedSphereTSWE(
    poly_order, nx, ny, g, f,
    eps, device=dev, solution=None, a=0.5, radius=radius, upwind=True,
    dtype=np.float64
)

sw_solver = CubedSphereSWE(
    poly_order, nx, ny, g, f,
    eps, device=dev, solution=None, a=0.5, radius=radius,
    dtype=np.float64, damping='adaptive'
)

p = 6
solver_hr = DGCubedSphereTSWE(
    p, nx, ny, g, f,
    eps, device=dev, solution=None, a=0.5, radius=radius,
    dtype=np.float64, damping='adaptive'
)
interpolator = Interpolate(3, p)

for face in solver.faces.values():
    face.set_initial_condition(*initial_condition(face))

solver.boundaries()
exp = f'DG_res_6x{nx}x{ny}'

print('Time step:', solver.get_dt())
print('Starting', exp)
print('a:', solver.faces['zp'].a, 'upwind:', solver.faces['zp'].upwind, 'res:', nx, ny)



fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

for ax in (ax1, ax2):
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")


moviewriter = MovieWriter(fps=20)
ndays = 20
tend = ndays * 3600 * 24

fame_dt = 3600
n_frames = int(tend / fame_dt)

with moviewriter.saving(fig, f"./videos/galewksy_res_6x{nx}x{ny}_order_{poly_order}_time_{ndays}.mp4", dpi=100):
    plot_sol(solver, solver_hr, sw_solver, ax1, ax2)
    moviewriter.grab_frame()

    for _ in range(n_frames):
        tend = solver.time + fame_dt
        print('Running')
        while solver.time < tend:
            dt = solver.get_dt()
            dt = min(dt, tend - solver.time)
            solver.time_step(dt=dt)

        plot_sol(solver, solver_hr, sw_solver, ax1, ax2)
        moviewriter.grab_frame()
        print('Days:', solver.time / (24 * 3600))

plt.show()
# solver.save_restart(fn_template, 'data')
