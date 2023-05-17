from matplotlib import pyplot as plt
from dg_tswe.utils import Interpolate
from dg_tswe.dg_cubed_sphere_tswe import DGCubedSphereTSWE
from dg_tswe.cubed_sphere_swe import CubedSphereSWE
import numpy as np
import scipy
import os

if not os.path.exists('./plots'): os.makedirs('./plots')
if not os.path.exists('./data'): os.makedirs('./data')

plt.rcParams['font.size'] = '12'

mode = 'run'
dev = 'cpu'

eps = 0.8
g = 9.80616
f = 7.292e-5
radius = 6.37122e6
poly_order = 3

u_0 = 80
h_0 = 10_000

def plot_sol(solver, solver_hr, exp_names, day):
    plot_func = lambda s: s.b_H1()

    def interpolate_plot_func(s):
        data = plot_func(solver.faces[s.name])
        return interpolator.torch_interpolate(data)

    for exp in exp_names:

        if day == 0:
            for face in solver.faces.values():
                face.set_initial_condition(*initial_condition(face))
            solver.boundaries()
        else:
            fn_template = f"{exp}_day_{day}.npy"
            solver.load_restart(fn_template, 'data')

        val_min = min(plot_func(face).numpy()[face.zs > 0].min() for face in solver.faces.values() if face.name != 'zn')
        val_max = max(plot_func(face).numpy()[face.zs > 0].max() for face in solver.faces.values() if face.name != 'zn')

        print(val_min, val_max)

        diff = 0.5 * (val_max - val_min)
        mean = 0.5 * (val_max + val_min)
        vmax = mean + 1.2 * diff
        vmin = mean - 1.2 * diff
        # E = solver.integrate(solver.entropy())
        # print(f'{exp} relative energy loss rate:', (E - E0) / (E0 * day * 24 * 3600))
        # print(f'{exp} adjusted energy loss rate (Wm^-2):', 3e9 * (E - E0) / (E0 * day * 24 * 3600))

        for name, face in solver.faces.items():
            data = [face.u, face.v, face.w, face.h, face.hb]
            data = [interpolator.torch_interpolate(tnsr).numpy() for tnsr in data]
            solver_hr.faces[name].set_initial_condition(*data)

        solver_hr.boundaries()

        # fig = plt.figure(figsize=(9, 3))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # plt.title(f"Relative vorticity day {day} {exp}")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")

        # im = solver.plot_solution(ax, dim=2, cmap='nipy_spectral', vmin=vmin, vmax=vmax, plot_func=plot_func)
        im = solver_hr.triangular_plot(ax, vmin=vmin, vmax=vmax, latlong=False, plot_func=interpolate_plot_func)
        plt.colorbar(im[0])
        plt.savefig(f'./plots/b_galewsky_{exp}_{int(day)}_days.png')

    solver = CubedSphereSWE(
        poly_order, nx, ny, g, f,
        eps, device=dev, solution=None, a=0.5, radius=radius,
        dtype=np.float64, damping='adaptive'
    )

    solver_hr = CubedSphereSWE(
        p, nx, ny, g, f,
        eps, device=dev, solution=None, a=0.5, radius=radius,
        dtype=np.float64, damping='adaptive'
    )
    vmin = -0.00015;
    vmax = 0.00015
    plot_func = lambda s: s.vorticity() - s.f
    for exp in exp_names:

        if day == 0:
            for face in solver.faces.values():
                face.set_initial_condition(*initial_condition(face)[:-1])
            solver.boundaries()
        else:
            fn_template = f"{exp}_day_{day}.npy"
            solver.load_restart(fn_template, 'data')

        # E = solver.integrate(solver.entropy())
        # print(f'{exp} relative energy loss rate:', (E - E0) / (E0 * day * 24 * 3600))
        # print(f'{exp} adjusted energy loss rate (Wm^-2):', 3e9 * (E - E0) / (E0 * day * 24 * 3600))

        for name, face in solver.faces.items():
            data = [face.u, face.v, face.w, face.h]
            data = [interpolator.torch_interpolate(tnsr).numpy() for tnsr in data]
            solver_hr.faces[name].set_initial_condition(*data)

        solver_hr.boundaries()

        # fig = plt.figure(figsize=(9, 3))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # plt.title(f"Relative vorticity day {day} {exp}")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")

        # im = solver.plot_solution(ax, dim=2, cmap='nipy_spectral', vmin=vmin, vmax=vmax, plot_func=plot_func)
        im = solver_hr.triangular_plot(ax, vmin=vmin, vmax=vmax, latlong=False, plot_func=interpolate_plot_func)
        plt.colorbar(im[0])
        plt.savefig(f'./plots/vort_galewsky_{exp}_{int(day)}_days.png')


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


if mode == 'run':

    nx = ny = 16
    exp_names = [f'DG_res_6x{nx}x{ny}', f'DG_cntr_res_6x{nx}x{ny}']

    for exp in exp_names:
        if 'cntr' in exp:
            a = 0.0
            upwind = False
        else:
            a = 0.5
            upwind = True
        solver = DGCubedSphereTSWE(
            poly_order, nx, ny, g, f,
            eps, device=dev, solution=None, a=a, radius=radius, upwind=upwind,
            dtype=np.float64
        )
        for face in solver.faces.values():
            face.set_initial_condition(*initial_condition(face))

        solver.boundaries()
        print('Time step:', solver.get_dt())
        print('Starting', exp)
        print('a:', solver.faces['zp'].a, 'upwind:', solver.faces['zp'].upwind, 'res:', nx, ny)

        for i in range(20):
            print('Running day', i)
            tend = solver.time + 3600 * 24
            while solver.time < tend:
                dt = solver.get_dt()
                dt = min(dt, tend - solver.time)
                solver.time_step(dt=dt)

            fn_template = f"{exp}_day_{i+1}.npy"
            solver.save_restart(fn_template, 'data')

        solver.save_diagnostics(fn_template, 'data')

elif mode == 'plot':
    nx = ny = 16
    solver = DGCubedSphereTSWE(
        poly_order, nx, ny, g, f,
        eps, device=dev, solution=None, a=0.5, radius=radius,
        dtype=np.float64, damping='adaptive'
    )
    for face in solver.faces.values():
        face.set_initial_condition(*initial_condition(face))
    E0 = solver.integrate(solver.entropy())

    p = 6
    solver_hr = DGCubedSphereTSWE(
        p, nx, ny, g, f,
        eps, device=dev, solution=None, a=0.5, radius=radius,
        dtype=np.float64, damping='adaptive'
    )

    interpolator = Interpolate(3, p)

    exp_names = [f'DG_res_6x{nx}x{ny}', f'DG_cntr_res_6x{nx}x{ny}']
    labels = ['Diss.', 'Cons.']

    for exp, label in zip(exp_names, labels):
        fn_template = f"{exp}_day_{20}.npy"
        solver.plot_diagnostics(fn_template, 'data', 1, label)
    plt.savefig(f'./plots/galewsky_conservation.png')

    for day in [0, 7, 16]:
        plot_sol(solver, solver_hr, exp_names, day)

    # plt.show()
