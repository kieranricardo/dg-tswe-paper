from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import linregress
from dg_tswe.dg_cubed_sphere_tswe import DGCubedSphereTSWE

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


# vort = -dudy + f
# u^T  = u j

# vort * u^T + g dgdy + u * dudy
# = (-dudy + f) * u + g dgdy + u * dudy = f * u + g dgdy

# s = g + k
# f * u + g dgdy = 0
# k * dhdy + 0.5 * h * dkdy = 0 -- (1 / h ^ 2) ?

# def initial_condition(face):
#     lat, long = face.geometry.lat_long(face.xs, face.ys, face.zs)
#     lat_vec_x, lat_vec_y, lat_vec_z, long_vec_x, long_vec_y, long_vec_z = face.geometry.lat_long_vecs(face.xs, face.ys, face.zs)
#     u_0 = 2 * np.pi * 6.37122e6 / (12 * 24 * 3600)
#     h_0 = 2.94e4 / g
#
#     h = h_0 - (face.geometry.radius * f * u_0 / g) * np.sin(lat)
#     u_ = u_0 * np.cos(lat)
#     b = g * (1 + 0.05 * (h_0 / h) ** 2)
#
#     u = long_vec_x * u_
#     v = long_vec_y * u_
#     w = long_vec_z * u_
#
#     return u, v, w, h, h * b


#(F1p + F2p) * (F1 + F2p) = F1p

exps = ['diss.', 'cons.']
coeffs = [0.5, 0.0]
ups = [True, False]

ns = np.array([3, 5, 10, 15, 30]) #, 25, 30])
grid_spacing = 360 / (ns * 4 * 3)
tend = 5.0 * 24 * 3600

for exp, a, up in zip(exps, coeffs, ups):
    h_errors = []
    b_errors = []
    vel_errors = []
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
            # if solver.time > 3600 * 24:
            #     solver.damping = 'adaptive'
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

    r, *_ = linregress(np.log(grid_spacing), np.log(np.array(h_errors)))
    print('h convergence:', r)
    r, *_ = linregress(np.log(grid_spacing), np.log(np.array(b_errors)))
    print('b convergence:', r)
    r, *_ = linregress(np.log(grid_spacing), np.log(np.array(vel_errors)))
    print('vel convergence:', r)

    plt.figure(1)
    plt.ylabel("Relative L2 error")
    plt.xlabel("Resolution")
    plt.loglog(grid_spacing, h_errors, '--o', label=f"h {exp}")
    plt.loglog(grid_spacing, b_errors, '--o', label=f"hb {exp}")
    plt.loglog(grid_spacing, vel_errors, '--o', label=f"u {exp}")


plt.plot(grid_spacing, (1 / 3) * h_errors[0] * (grid_spacing[0] ** (-4)) * (grid_spacing ** (4)), linestyle='--', label='4th order')
plt.plot(grid_spacing, 3 * h_errors[0] * (grid_spacing[0] ** (-3)) * (grid_spacing ** (3)), linestyle='--', label='3rd order')
plt.grid()
plt.legend()
plt.savefig('./plots/steady_state_error.png')
plt.show()



