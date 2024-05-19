from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import linregress
from dg_tswe.dg_cubed_sphere_tswe import DGCubedSphereTSWE
import os

if not os.path.exists('./plots'): os.makedirs('./plots')
if not os.path.exists('./data'): os.makedirs('./data')

plt.rcParams['font.size'] = '12'

mode = 'plot'
dev = 'cpu'

eps = 0.8
g = 9.80616
f = 7.292e-5
radius = 6.37122e6
poly_order = 3
#
# angle = 30 * (np.pi / 180)


# s = g + a
# a \grad h + 0.5 * h * grad a
# (a / grad a) = -0.5 h / grad h

# u * du/dy + g grad h = 0 ???


def initial_condition(face):
    lat, long = face.geometry.lat_long(face.xs, face.ys, face.zs)
    lat_vec_x, lat_vec_y, lat_vec_z, long_vec_x, long_vec_y, long_vec_z = face.geometry.lat_long_vecs(face.xs, face.ys, face.zs)

    # lam = long
    # theta = lat
    u_0 = 2 * np.pi * 6.37122e6 / (12 * 24 * 3600)
    h_0 = 2.94e4 / g
    u_ = u_0 * np.cos(lat)
    h = h_0 - (1 / g) * (face.geometry.radius * f * u_0 + 0.5 * u_0 ** 2) * np.sin(lat) ** 2

    u = long_vec_x * u_
    v = long_vec_y * u_
    w = long_vec_z * u_

    b = g * (1 + 0.05 * (h_0 / h) ** 2)

    return u, v, w, h, h * b


run = False
exps = ['diss.', 'cons.']
coeffs = [0.5, 0.0]
ups = [True, False]
markers = {exps[0]: '--', exps[1]: '-'}

ns = np.array([3, 5, 10, 15, 30]) #, 25, 30])
grid_spacing = 360 / (ns * 4 * 3)
tend = 5.0 * 24 * 3600

# plot IC
solver = DGCubedSphereTSWE(
    poly_order, 30, 30, g, f,
    eps, device=dev, solution=None, a=0.5, radius=radius,
    dtype=np.float64, damping=None
)

for face in solver.faces.values():
    face.set_initial_condition(*initial_condition(face))

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")

vmin = min((s.hb / s.h).min() for s in solver.faces.values())
vmax = max((s.hb / s.h).max() for s in solver.faces.values())
im = solver.triangular_plot(ax, latlong=False, vmin=vmin, vmax=vmax, plot_func=lambda s: s.hb / s.h, n=20)
plt.colorbar(im[0])
plt.savefig('./plots/steady_state_ic.png')


for exp, a, up in zip(exps, coeffs, ups):
    h_errors = []
    b_errors = []
    vel_errors = []
    if run:
        for n in ns:
            print('Running', n)
            solver = DGCubedSphereTSWE(
                poly_order, n, n, g, f,
                eps, device=dev, solution=None, a=a, radius=radius,
                dtype=np.float64, upwind=up, damping=None,
            )

            solver0 = DGCubedSphereTSWE(
                poly_order, n, n, g, f,
                eps, device=dev, solution=None, a=0.5, radius=radius,
                dtype=np.float64, damping=None
            )

            for face in solver.faces.values():
                face.set_initial_condition(*initial_condition(face))

            for face in solver0.faces.values():
                face.set_initial_condition(*initial_condition(face))

            h_norm = np.sqrt(sum(face.integrate(face.h ** 2) for face in solver.faces.values()))
            b_norm = np.sqrt(sum(face.integrate(face.hb ** 2) for face in solver.faces.values()))
            vel_norm = np.sqrt(sum(face.integrate(face.u ** 2 + face.v ** 2 + face.w ** 2) for face in solver.faces.values()))

            while solver.time < tend:
                dt = solver.get_dt()
                dt = min(dt, tend - solver.time)
                solver.time_step(dt=dt)

            h_error = sum(f1.integrate((f1.h - f2.h) ** 2) for f1, f2 in zip(solver.faces.values(), solver0.faces.values()))
            h_error = np.sqrt(h_error) / h_norm

            vel_error = sum(f1.integrate((f1.u - f2.u) ** 2 + (f1.v - f2.v) ** 2 + (f1.w - f2.w) ** 2) for f1, f2 in zip(solver.faces.values(), solver0.faces.values()))
            vel_error = np.sqrt(vel_error) / vel_norm

            b_error = sum(f1.integrate((f1.hb - f2.hb) ** 2) for f1, f2 in zip(solver.faces.values(), solver0.faces.values()))
            b_error = np.sqrt(b_error) / b_norm

            h_errors.append(h_error)
            b_errors.append(b_error)
            vel_errors.append(vel_error)

        h_errors = np.array(h_errors)
        b_errors = np.array(b_errors)
        vel_errors = np.array(vel_errors)

        np.save(f'data/{exp}_h_errors.npy', h_errors)
        np.save(f'data/{exp}_b_errors.npy', b_errors)
        np.save(f'data/{exp}_vel_errors.npy', vel_errors)
        np.save(f'data/{exp}_grid_spacing.npy', grid_spacing)

    h_errors = np.load(f'data/{exp}_h_errors.npy')
    b_errors = np.load(f'data/{exp}_b_errors.npy')
    vel_errors = np.load(f'data/{exp}_vel_errors.npy')
    grid_spacing = np.load(f'data/{exp}_grid_spacing.npy')

    r, *_ = linregress(np.log(grid_spacing), np.log(np.array(h_errors)))
    print('h convergence:', r)
    r, *_ = linregress(np.log(grid_spacing), np.log(np.array(b_errors)))
    print('b convergence:', r)
    r, *_ = linregress(np.log(grid_spacing), np.log(np.array(vel_errors)))
    print('vel convergence:', r)


    plt.figure(2)
    plt.ylabel("Relative L2 error")
    plt.xlabel("Resolution (degrees)")
    plt.loglog(grid_spacing, h_errors, f'r{markers[exp]}', label=f"h {exp}")
    plt.loglog(grid_spacing, b_errors, f'b{markers[exp]}', label=f"hb {exp}")
    plt.loglog(grid_spacing, vel_errors, f'g{markers[exp]}', label=f"u {exp}")


plt.plot(grid_spacing, (1 / 6) * h_errors[0] * (grid_spacing[0] ** (-4)) * (grid_spacing ** (4)), linestyle='dotted', color='black', label='4th order')
plt.plot(grid_spacing, 6 * h_errors[0] * (grid_spacing[0] ** (-3)) * (grid_spacing ** (3)), linestyle='dashdot', color='black', label='3rd order')
plt.xticks(grid_spacing, [str(x) for x in grid_spacing])
plt.grid()

handles, labels = plt.gca().get_legend_handles_labels()
order = [5, 2, 4, 1, 3, 0, 6, 7]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc=4)
plt.savefig('./plots/steady_state_error.png')
plt.show()



