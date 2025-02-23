import taichi as ti
import numpy as np
from numpy import linalg as LA
import math
write_to_disk = True
# Try to run on GPU. Use arch=ti.opengl on old GPUs
ti.init(arch=ti.gpu)
dim = 3
quality = 1  # Use a larger value for higher-res simulations
n_particles, n_grid = 1728 * quality ** dim, 84 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
E, nu = 0.1e4, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) *
                                               (1 - 2 * nu))  # initial Lame coefficients
x = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)  # position
v = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)  # velocity
# C = ti.Matrix(dim, dim, dt=ti.f32, shape=n_particles) # affine velocity field

''' deformation gradient '''
F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
Jp = ti.field(dtype=ti.f32, shape=n_particles)  # plastic deformation
N_values = ti.math.vec3(0.0)
N_grad_values = ti.math.vec3(0.0)
''' deformation gradient '''

"""foam attribute"""
p_m = ti.field(dtype=ti.f32, shape=n_particles)
rho = ti.field(dtype=ti.f32, shape=n_particles)
vol = ti.field(dtype=ti.f32, shape=n_particles)
weak = ti.field(dtype=ti.f32, shape=n_particles)
grid_f = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
new_grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
f = open('0.txt', 'w')
"""foam attribute"""

"""foam var"""
gravity = -10
kappa = 109.0
mu = 11.2
yield_stress = 0.1
eta = 10
h = 2.8
tear_stress = 1
eta_p = 0.3
# b_pre = np.zeros((3, 3))
b_pre = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)
"""foam var"""
# if dim == 2:
#   grid_v = ti.Vector(dim, dt=ti.f32, shape=(n_grid, n_grid)) # grid node momentum/velocity
#   grid_m = ti.var(dt=ti.f32, shape=(n_grid, n_grid)) # grid node mass
# elif dim == 3:
#   grid_v = ti.Vector(dim, dt=ti.f32, shape=(n_grid, n_grid, n_grid)) # grid node momentum/velocity
#   grid_m = ti.var(dt=ti.f32, shape=(n_grid, n_grid, n_grid)) # grid node mass
# grid node momentum/velocity
grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(
    n_grid, n_grid, n_grid))  # grid node mass

""" particle palette """
ti_palette = ti.field(dtype=ti.i64, shape=256)
ti_palette_index = ti.field(dtype=ti.i64, shape=n_particles)
""" particle palette """


def stencil_range():
    return ti.ndrange(*((3, ) * dim))


@ti.func
def N(x: float):
    abs_x = ti.abs(x)
    result = 0.0
    if abs_x < 1:
        result = 0.5 * abs_x**3 - abs_x**2 + 2 / 3
    elif abs_x < 2:
        result = -1 / 6 * abs_x**3 + abs_x**2 - 2 * abs_x + 4 / 3
    return result


@ti.func
def N_grad(x: float):
    result = 0.0
    if x < 1 and x >= 0:
        result = 1.5*x**2 - 2*x
    elif x < 2 and x >= 1:
        result = -0.5*x**2 + 2*x - 2
    elif x > -1 and x <= 0:
        result = -1.5*x**2 - 2*x
    elif x > -2 and x <= -1:
        result = 0.5*x**2 + 2*x + 2
    return result


@ti.func
def wip(i, j, k, p):
    a = N(inv_dx * (p[0] - i*dx))
    b = N(inv_dx * (p[1] - j*dx))
    c = N(inv_dx * (p[2] - k*dx))
    return a * b * c


@ti.func
def wip_precomputed(p):
    N_values = ti.Vector.zero(ti.f32, 3)
    N_grad_values = ti.Vector.zero(ti.f32, 3)

    base = (p * inv_dx - 0.5).cast(int)  # 计算粒子 p 所在的网格索引

    for d in ti.static(range(3)):
        diff = inv_dx * (p[d] - base[d] * dx)
        N_values[d] = N(diff)
        N_grad_values[d] = N_grad(diff)

    return N_values, N_grad_values


@ti.func
def wip_grad_optimized(N_values, N_grad_values):
    w_g = ti.Matrix.zero(ti.f32, 3)

    w_g[0] = inv_dx * N_grad_values[0] * N_values[1] * N_values[2]
    w_g[1] = inv_dx * N_values[0] * N_grad_values[1] * N_values[2]
    w_g[2] = inv_dx * N_values[0] * N_values[1] * N_grad_values[2]

    return w_g


@ti.func
def eig3x3(A):
    eigvals = ti.Vector([0.0, 0.0, 0.0])
    eigvecs = ti.Matrix.identity(ti.f32, 3)

    # compute trace, determinant
    tr = A.trace()
    det = A.determinant()
    I = ti.Matrix.identity(ti.f32, 3)

    # eigenvalues
    B = A - (tr / 3) * I
    q = (3 * B.determinant()) / 2

    q = max(min(q, 1.0), -1.0)

    theta = ti.acos(q) / 3
    eigvals[0] = (tr / 3) + 2 * ti.sqrt(tr**2 / 9 - det / 3) * ti.cos(theta)
    eigvals[1] = (tr / 3) + 2 * ti.sqrt(tr**2 / 9 - det / 3) * \
        ti.cos(theta + 2 * ti.math.pi / 3)
    eigvals[2] = (tr / 3) + 2 * ti.sqrt(tr**2 / 9 - det / 3) * \
        ti.cos(theta - 2 * ti.math.pi / 3)

    # compute eigenvectors
    for i in ti.static(range(3)):
        eigvecs[:, i] = power_iteration(A, eigvals[i])

    return eigvals, eigvecs


@ti.func
def power_iteration(A, eigval):
    v = ti.Vector([1.0, 0.0, 0.0])  # initialization
    for _ in range(10):  # iteratively converge
        v = (A @ v) - eigval * v
        norm_v = v.norm()
        if norm_v < 1e-6:  # avoid division by zero
            v = ti.Vector([1.0, 0.0, 0.0])  # reset to default value
        else:
            v = v / norm_v  # normalize
    return v


@ti.func
def diag_matrix(v):
    return ti.Matrix([
        [v[0], 0.0, 0.0],
        [0.0, v[1], 0.0],
        [0.0, 0.0, v[2]]
    ])


@ti.func
def svd3x3(A):
    """ 计算 3x3 矩阵的 SVD """
    U = ti.Matrix.identity(ti.f32, 3)
    V = ti.Matrix.identity(ti.f32, 3)
    Sigma = ti.Matrix.zero(ti.f32, 3, 3)

    for _ in range(10):  # iteratively converge
        eigvals, eigvecs = eig3x3(A @ A.transpose())  # 计算 U
        U = eigvecs
        Sigma = diag_matrix(ti.sqrt(eigvals))

        # compute V
        V_raw = U.transpose() @ A
        for i in ti.static(range(3)):
            norm_i = V_raw[:, i].norm()
            if norm_i > 1e-6:
                V[:, i] = V_raw[:, i] / norm_i
            else:
                V[:, i] = ti.Vector([1.0, 0.0, 0.0])  # default value

    return U, Sigma, V


@ti.kernel
def P2G():
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0, 0, 0]
        grid_m[i, j, k] = 0

    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        # fx = x[p] * inv_dx - base.cast(float)

        # Loop over 3x3 grid node neighborhood
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            # dpos = (offset.cast(float) - fx) * dx
            index = base + offset
            weight = wip(index[0], index[1], index[2], x[p])
            if weight > 0:
                grid_v[index] += weight * (p_m[p] * v[p])
                grid_m[index] += weight * p_m[p]

    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0.00001:
            grid_v[i, j, k] = grid_v[i, j, k]/grid_m[i, j, k]


@ti.kernel
def Compute_Particle_Volumes_and_Densities():
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        rho[p] = 0

        # Loop over 3x3 grid node neighborhood
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            index = base + offset
            weight = wip(index[0], index[1], index[2], x[p])
            if weight > 0:
                rho[p] += weight * grid_m[index]
        rho[p] /= dx**3
        rho[p] = max(rho[p], 1e-3)
        vol[p] = p_m[p] / rho[p]


@ti.kernel
def Dectect_Tearing_Part():
    for p in x:
        b = F[p]@F[p].transpose()
        Cpp = F[p].determinant()**(-2/3) * F[p].transpose()@b.inverse()@F[p]
        tr = Cpp[0, 0] + Cpp[1, 1] + Cpp[2, 2]
        dev = Cpp - (tr/3)*ti.Matrix.identity(ti.f32, 3)
        dev_sqr = dev@dev
        tr = ti.sqrt(dev_sqr[0, 0] + dev_sqr[1, 1] + dev_sqr[2, 2])
        if tr > tear_stress:
            weak[p] = 1
        else:
            weak[p] = 0


@ti.kernel
def Grid_Force():
    for i, j, k in grid_f:
        grid_f[i, j, k] = ti.Vector([0.0, 0.0, 0.0])

    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        Jp_val = F[p].determinant()
        b = F[p] @ F[p].transpose()
        det_b = b.determinant()
        normalized_b = (det_b**(-1/3)) * b
        tr = normalized_b.trace()
        dev = normalized_b - (tr/3) * ti.Matrix.identity(ti.f32, 3)
        stress = ti.Matrix.zero(ti.f32, 3, 3)

        if Jp_val != 0 and not (ti.math.isinf(Jp_val) or ti.math.isnan(Jp_val) or ti.math.isinf(tr) or ti.math.isnan(tr)):
            Jp_safe = max(Jp_val, 1e-6)
            stress = (1/Jp_safe) * (kappa/2 * (Jp_safe**2 - 1)
                                    * ti.Matrix.identity(ti.f32, 3) + mu * dev)
        if weak[p] == 1:
            eigvals, eigvecs = eig3x3(stress)
            for i in ti.static(range(3)):
                if eigvals[i] > 0:
                    eigvals[i] = 0
            stress = eigvecs @ diag_matrix(eigvals) @ eigvecs.transpose()

        N_values, N_grad_values = wip_precomputed(x[p])  # precompute

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            index = base + offset
            weight = N_values[0] * N_values[1] * N_values[2]
            if weight > 0:
                wip_g = wip_grad_optimized(N_values, N_grad_values)
                grid_f[index] -= Jp_val * vol[p] * stress @ wip_g


@ti.kernel
def Update_Grid_V():

    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:  # No need for epsilon here
            new_grid_v[i, j, k] = grid_v[i, j, k]
            new_grid_v[i, j, k] += dt * grid_f[i, j, k]/grid_m[i, j, k]
            new_grid_v[i, j, k][1] += dt * gravity

    for i, j, k in grid_m:
        if i < 3 and new_grid_v[i, j, k][0] < 0:
            print("Boundary")
            new_grid_v[i, j, k][0] = 0  # Boundary conditions
        if i >= n_grid - 3 and new_grid_v[i, j, k][0] > 0:
            new_grid_v[i, j, k][0] = 0
            print("Boundary")
        if j < 3 and new_grid_v[i, j, k][1] < 0:
            new_grid_v[i, j, k][1] = 0  # Boundary conditions
            print("Boundary")
        if j >= n_grid - 3 and new_grid_v[i, j, k][1] > 0:
            new_grid_v[i, j, k][1] = 0
            print("Boundary")
        if k < 3 and new_grid_v[i, j, k][2] < 0:
            print("Boundary")
            new_grid_v[i, j, k][2] = 0  # Boundary conditions
        if k >= n_grid - 3 and new_grid_v[i, j, k][2] > 0:
            print("Boundary")
            new_grid_v[i, j, k][2] = 0


@ti.kernel
def Update_Limit_Deformation_Gradient():
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        v_g = ti.Matrix.zero(ti.f32, 3, 3)

        N_values, N_grad_values = wip_precomputed(x[p])  # precompute

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            index = base + offset
            weight = N_values[0] * N_values[1] * N_values[2]
            if weight > 0:
                wip_g = wip_grad_optimized(N_values, N_grad_values)
                weight_v = new_grid_v[index].outer_product(wip_g)
                v_g += weight_v
        fp = (ti.Matrix.identity(ti.f32, 3) + dt * v_g)
        det_fp = fp.determinant()
        det_fp_safe = max(det_fp, 1e-6)
        fp_normal = (det_fp_safe**(-1/3)) * fp
        b = F[p] @ F[p].transpose()
        det_b = b.determinant()
        normalized_b = ti.Matrix.identity(ti.f32, 3)
        weight = (det_b**(-1/3))
        if not ti.math.isinf(weight) and not ti.math.isnan(weight):
            normalized_b = weight * b
        b_pre[p] = fp_normal @ normalized_b @ fp_normal.transpose()
        prev_F = F[p]
        if weak[p] == 0:
            F[p] = fp @ F[p]

        else:
            eigvals, eigvecs = eig3x3(normalized_b)
            eigvals_pre, eigvecs_pre = eig3x3(b_pre[p])
            max_val = max(eigvals[0], eigvals[1], eigvals[2])
            max_pre_val = max(eigvals_pre[0], eigvals_pre[1], eigvals_pre[2])
            f_corr = fp
            if max_pre_val > max_val:
                U, sig, V = svd3x3(fp_normal)
                fp_normal = U @ V.transpose()
                b_pre[p] = fp_normal @ normalized_b @ fp_normal.transpose()
                det_fp = fp.determinant()
                f_corr = (det_fp**(-1/3)) * fp_normal
            Jn = prev_F.determinant()
            Jcorr = f_corr.determinant()
            f_corr = ti.Matrix.identity(ti.f32, 3)
            a = Jcorr**(-1/3)
            if Jn * Jcorr > 1 and Jcorr > 1 and not (ti.math.isinf(a) or ti.math.isnan(a)):
                f_corr = (Jcorr**(-1/3)) * f_corr
            F[p] = f_corr @ prev_F


@ti.kernel
def Update_Particle_V():
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        v_PIC = ti.Matrix.zero(ti.f32, dim)
        v_FLIP = v[p]

        # Loop over 3x3 grid node neighborhood
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            weight = wip((base + offset)
                         [0], (base + offset)[1], (base + offset)[2], x[p])
            if weight > 0:
                v_PIC = v_PIC + weight * new_grid_v[base + offset]
                v_FLIP = v_FLIP + weight * \
                    (new_grid_v[base + offset] - grid_v[base + offset])
        alpha = 0.95
        v[p] = (1-alpha)*v_PIC + alpha*v_FLIP


@ti.kernel
def Plastic_Flow():
    for p in x:
        tr = b_pre[p].trace()
        dev = b_pre[p] - (tr/3) * ti.Matrix.identity(ti.f32, 3)

        s_pre = mu * dev

        norm_s_pre = ti.sqrt(
            s_pre[0, 0]**2 + s_pre[0, 1]**2 + s_pre[0, 2]**2 +
            s_pre[1, 0]**2 + s_pre[1, 1]**2 + s_pre[1, 2]**2 +
            s_pre[2, 0]**2 + s_pre[2, 1]**2 + s_pre[2, 2]**2
        )
        norm_s = 0.0
        y_s = ti.sqrt(2/3)*yield_stress
        b = F[p] @ F[p].transpose()

        if norm_s_pre > y_s:
            mu_prime = (1/3) * tr * mu
            if eta == 0 or h == 1:
                norm_s = norm_s_pre - \
                    ((norm_s_pre-y_s) / (1+eta/(2*mu_prime*dt)))
            else:
                s_min, s_max = y_s, norm_s_pre
                for i in range(3e4):
                    # 13:
                    s = (s_min+s_max)/2
                    # 14:
                    norm_s = (eta**(1/h))*(s-norm_s_pre) + 2 * \
                        mu_prime*dt*(((s-y_s)**(1/h)))
                    # 15:
                    if norm_s < 0:
                        # 16:
                        s_min = s
                    # 17:
                    else:
                        # 18:
                        s_max = s

                    E = norm_s/norm_s_pre
                    # if norm_s_pre > 0:
                    #     E = norm_s/norm_s_pre
                    # 21:
                    if abs(E) < 5e-6:
                        break
            flow = (1 / norm_s_pre) * s_pre
            s = norm_s * flow
            b = (1/mu)*s + (tr/3)*ti.Matrix.identity(ti.f32, 3)
            det_b = b.determinant()
            b *= det_b**(-1/3)
        det_F = F[p].determinant()
        Cpp = ti.Matrix.identity(ti.f32, 3)
        det_b = b.determinant()
        assert (not ti.math.isnan(det_b)), "det_b is nan"
        inv_b = ti.Matrix.identity(ti.f32, 3)
        if det_b != 0:
            inv_b = b.inverse()
        weight = det_F**(-2/3)
        a = F[p].transpose() @ inv_b @ F[p]

        det_a = a.determinant()
        assert (not ti.math.isnan(weight)), "weight is nan"
        assert (not ti.math.isnan(det_a)), "det_a is nan"
        if not ti.math.isinf(weight) and not ti.math.isnan(weight) and not ti.math.isinf(det_a) and not ti.math.isnan(det_a) and det_b != 0:
            Cpp = weight * a
        eigvals, eigvecs = eig3x3(Cpp)
        for i in ti.static(range(3)):
            if eigvals[i] >= 0:
                eigvals[i] = eigvals[i]**((ti.exp(-dt/eta_p)-1)/2)
            else:
                eigvals[i] = - \
                    ((-eigvals[i])**((ti.exp(-dt/eta_p)-1)/2))
            if ti.math.isnan(eigvals[i]) or ti.math.isinf(eigvals[i]):
                print(
                    "Warning: eigvals[i] is NaN or Inf at index", i, "value:", eigvals[i])
        Cpp = eigvecs @ diag_matrix(eigvals) @ eigvecs.transpose()
        F[p] = F[p] @ Cpp
        if ti.math.isnan(F[p]).sum() > 0:
            print("F[p] is nan")
            print("Cpp", Cpp)
            print("eigvecs", eigvecs)
            print("eigvals", eigvals)


@ti.kernel
def Particle_Position():
    for p in x:
        x[p] = x[p] + dt*v[p]


def substep():
    print("P2G")
    P2G()
    print("P2G done")
    # print("Dectect_Tearing_Part")
    # Dectect_Tearing_Part()
    # print("Dectect_Tearing_Part done")

    print("Grid_Force")
    Grid_Force()
    print("Grid_Force done")

    print("Update_Grid_V")
    Update_Grid_V()
    print("Update_Grid_V done")

    print("Update_Limit_Deformation_Gradient")
    Update_Limit_Deformation_Gradient()
    if np.isnan(np.min(F.to_numpy())):
        print("D_G_nan")
        quit()
    print("Update_Limit_Deformation_Gradient done")

    print("Update_Particle_V")
    Update_Particle_V()
    print("Update_Particle_V done")

    print("Plastic_Flow")
    Plastic_Flow()
    if np.isnan(np.min(F.to_numpy())):
        print("P_F_nan")
        quit()
    print("Plastic_Flow done")

    print("Particle_Position")
    Particle_Position()
    print("Particle_Position done")
    print("---------------------------")


def particle_log(np_x, v, F, color, depth_range):
    palette = ti_palette.to_numpy()
    f.write('palette '+str(palette)+'\n')
    f.write('depth_range '+str(depth_range)+'\n')
    for p in range(n_particles):
        f.write('particle '+str(p)+'\n')
        f.write('x:\n')
        f.write(str(np_x[p][0])+" "+str(np_x[p][1])+" "+str(np_x[p][2])+"\n")
        f.write('color:\n')
        f.write(str(color[p])+"\n")
        f.write(str(palette[color[p]])+"\n")
        f.write('v:\n')
        f.write(str(v[p][0])+" "+str(v[p][1])+" "+str(v[p][2])+"\n")
        if (abs(v[p][1]) < 0.001):
            f.write('small v\n')
        f.write('F:\n')
        for i in range(3):
            f.write(str(F[p][i, 0])+" "+str(F[p][i, 1]) +
                    " "+str(F[p][i, 2])+"\n")


group_size = n_particles  # //2


@ti.kernel
def initialize():
    base = ti.Vector([0.5, 0.6, 0.5])
    for p in x:
        # 或者 ti.Matrix.zero(ti.f32, 3, 3)
        b_pre[p] = ti.Matrix.zero(ti.f32, 3, 3)
    for i in range(12):
        for j in range(12):
            for k in range(12):
                index = i * 12 * 12 + j * 12 + k
                pos = base + 0.01 * ti.Vector([i, j, k])
                for d in ti.static(range(dim)):
                    x[index][d] = pos[d]
                v[index] = ti.Matrix.zero(ti.f32, dim)
                F[index] = ti.Matrix.identity(ti.f32, dim)
                Jp[index] = 1
                p_m[index] = (dx * 0.5)**2
                weak[index] = 0
    for i in range(84):
        progress = i / 83
        r = int(256 * (1 - progress) * 0.8)
        g = int(206 * (1 - progress) + 50)
        b = int(206 * (1 - progress) + 50)
        ti_palette[i] = ((r << 16) | (g << 8) | b)


@ti.kernel
def compute_palette_indices():
    for p in x:
        depth = x[p][2]  # 取 Z 值
        depth_min = 0.0  # 可以预先计算
        depth_max = 1.0  # 可以预先计算
        depth_normalized = (depth - depth_min) / (depth_max - depth_min)
        ti_palette_index[p] = ti_palette[ti.cast(
            depth_normalized * 84, ti.i64)]


@ti.kernel
def copy_dynamic_nd(np_x: ti.types.ndarray(), input_x: ti.template()):
    for i in x:
        for j in ti.static(range(dim)):
            np_x[i, j] = input_x[i][j]


print("P2G")
P2G()
print("P2G Done")
Compute_Particle_Volumes_and_Densities()
print("Compute_Particle_Volumes_and_Densities Done")


initialize()
gui = ti.GUI("MPM-Foam", res=int(1024 * 0.8), background_color=0x000000)

for frame in range(200):

    np_x = np.ndarray((n_particles, dim), dtype=np.float32)
    copy_dynamic_nd(np_x, x)
    if dim == 3:
        angle_x = 0.5
        angle_y = 0.3

        # project to 2D screen space
        screen_x = np_x[:, 0] - (np_x[:, 2] * angle_x)
        screen_y = np_x[:, 1] - (np_x[:, 2] * angle_y)

        # Adjust radius based on depth
        depth = np_x[:, 2].copy()
        depth_min, depth_max = depth.min(), depth.max()
        depth_range = max(depth_max - depth_min, 1e-6)  # 避免除以零
        depth_normalized = (depth - depth_min) / depth_range

        size_min, size_max = 1.0, 2.5
        radii = size_min + depth_normalized * (size_max - size_min)

        compute_palette_indices()
        ti.sync()
        # Draw Particles
        screen_pos = np.stack([screen_x, screen_y], axis=-1)
        gui.circles(screen_pos, radius=radii,
                    color=ti_palette_index.to_numpy())

        # np_x = x.to_numpy()
        # np_v = v.to_numpy()
        # np_F = F.to_numpy()
        # particle_log(np_x, np_v, np_F, ti_palette_index.to_numpy())

    else:
        screen_x = (np_x[:, 0])
        screen_y = (np_x[:, 1])

        screen_pos = np.stack([screen_x, screen_y], axis=-1)
        gui.circles(screen_pos, radius=1.5, color=0xEEEEF0)
    gui.show(f'{frame:06d}.png' if write_to_disk else None)

    for s in range(int(2e-3 // dt)):
        substep()

    # gui.show() # Change to gui.show(f'{frame:06d}.png') to write images to disk
