import taichi as ti
import numpy as np
from numpy import linalg as LA
import math
write_to_disk = True
ti.init(arch=ti.gpu) # Try to run on GPU. Use arch=ti.opengl on old GPUs
dim = 3
quality = 1 # Use a larger value for higher-res simulations
n_particles, n_grid = 1500 * quality ** dim, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
# p_vol, p_rho = ti.cast((dx * 0.5)**dim, ti.f32), ti.cast(1.4, ti.f32)
# p_mass = p_vol * p_rho
E, nu = 0.1e4, 0.2 # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # initial Lame coefficients
x = ti.Vector(dim, dt=ti.f32, shape=n_particles) # position
v = ti.Vector(dim, dt=ti.f32, shape=n_particles) # velocity
# C = ti.Matrix(dim, dim, dt=ti.f32, shape=n_particles) # affine velocity field
F = ti.Matrix(dim, dim, dt=ti.f32, shape=n_particles) # deformation gradient
Jp = ti.var(dt=ti.f32, shape=n_particles) # plastic deformation
"""foam attribute"""
p_m = ti.var(dt=ti.f32, shape=n_particles)
rho = ti.var(dt=ti.f32, shape=n_particles)
vol = ti.var(dt=ti.f32, shape=n_particles)
weak = ti.var(dt=ti.f32, shape=n_particles)
grid_f = ti.Vector(dim, dt=ti.f32, shape=(n_grid, n_grid, n_grid))
new_grid_v = ti.Vector(dim, dt=ti.f32, shape=(n_grid, n_grid, n_grid))
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
b_pre = np.zeros((3,3))
"""foam var"""
# if dim == 2:
#   grid_v = ti.Vector(dim, dt=ti.f32, shape=(n_grid, n_grid)) # grid node momentum/velocity
#   grid_m = ti.var(dt=ti.f32, shape=(n_grid, n_grid)) # grid node mass
# elif dim == 3:
#   grid_v = ti.Vector(dim, dt=ti.f32, shape=(n_grid, n_grid, n_grid)) # grid node momentum/velocity
#   grid_m = ti.var(dt=ti.f32, shape=(n_grid, n_grid, n_grid)) # grid node mass
grid_v = ti.Vector(dim, dt=ti.f32, shape=(n_grid, n_grid, n_grid)) # grid node momentum/velocity
grid_m = ti.var(dt=ti.f32, shape=(n_grid, n_grid, n_grid)) # grid node mass

def stencil_range():
  return ti.ndrange(*((3, ) * dim))

def N(x:float):
  abs_x = abs(x)
  if abs_x < 1 and abs_x >= 0:
    return 0.5*abs_x**3 - abs_x**2 + 2/3
  elif abs_x < 2 and abs_x >= 1:
    return -1/6*abs_x**3 + abs_x**2 - 2*abs_x + 4/3
  else:
    return 0

def N_grad(x:float):
  if x < 1 and x >= 0:
    return 1.5*x**2 - 2*x
  elif x < 2 and x >= 1:
    return -0.5*x**2 + 2*x - 2
  elif x > -1 and x <=0:
    return -1.5*x**2 - 2*x
  elif x > -2 and x <= -1:
    return 0.5*x**2 + 2*x + 2
  else:
    return 0

@ti.func
def wip(i, j, k, p):
  a = N(inv_dx * (p[0]- i*dx))
  b = N(inv_dx * (p[1]- j*dx))
  c = N(inv_dx * (p[2]- k*dx))
  return a * b * c

def np_wip(i, j, k, p):
  a = N(inv_dx * (p[0]- i*dx))
  b = N(inv_dx * (p[1]- j*dx))
  c = N(inv_dx * (p[2]- k*dx))
  return a * b * c

@ti.func
def wip_grad(i, j, k, p):
  w_g = ti.Matrix.zero(ti.f32, dim)
  w_g[0] = inv_dx * N_grad(inv_dx * (p[0]- i*dx)) * N(inv_dx * (p[1]- j*dx)) * N(inv_dx * (p[2]- k*dx))
  w_g[1] = inv_dx * N(inv_dx * (p[0]- i*dx)) * N_grad(inv_dx * (p[1]- j*dx)) * N(inv_dx * (p[2]- k*dx))
  w_g[2] = inv_dx * N(inv_dx * (p[0]- i*dx)) * N(inv_dx * (p[1]- j*dx)) * N_grad(inv_dx * (p[2]- k*dx))
  return w_g

def np_wip_grad(i, j, k, p):
  w_g = np.zeros(3)
  N_x = N(inv_dx * (p[0]- i*dx))
  N_y = N(inv_dx * (p[1]- j*dx))
  N_z = N(inv_dx * (p[2]- k*dx))
  x = N_grad(inv_dx * (p[0]- i*dx)) * N_y * N_z
  y = N_x * N_grad(inv_dx * (p[1]- j*dx)) * N_z
  z = N_x * N_y * N_grad(inv_dx * (p[2]- k*dx))

  w_g[0] = x
  w_g[1] = y
  w_g[2] = z
  w_g *= inv_dx
  return w_g

@ti.kernel
def P2G():
  for i, j, k in grid_m:
    grid_v[i, j, k] = [0,0,0]
    grid_m[i, j, k] = 0
  
  for p in x:
    base = (x[p] * inv_dx - 0.5).cast(int)
    # fx = x[p] * inv_dx - base.cast(float)

    for i, j ,k in ti.static(ti.ndrange(3, 3, 3)): # Loop over 3x3 grid node neighborhood
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

    for i, j ,k in ti.static(ti.ndrange(3, 3, 3)): # Loop over 3x3 grid node neighborhood
      offset = ti.Vector([i, j, k])
      index = base + offset
      weight = wip(index[0], index[1], index[2], x[p])
      if weight > 0:
        rho[p] += weight * grid_m[index]
    rho[p] /= dx**3
    vol[p] = p_m[p] / rho[p]
    
@ti.kernel
def Dectect_Tearing_Part():
  for p in x:
    b = F[p]@F[p].transpose()
    Cpp = F[p].determinant()**(-2/3) * F[p].transpose()@b.inverse()@F[p]
    tr = Cpp[0,0] + Cpp[1,1] + Cpp[2,2]
    dev = Cpp - (tr/3)*ti.Matrix.identity(ti.f32, 3)
    dev_sqr = dev@dev
    tr = ti.sqrt(dev_sqr[0,0] + dev_sqr[1,1] + dev_sqr[2,2])
    if tr > tear_stress:
      weak[p] = 1
    else:
      weak[p] = 0

def np_Grid_Force(x, v, F, grid_f):
  grid_f = np.zeros((128, 128, 128, 3))

  for p in range(n_particles):
    base = (x[p] * inv_dx - 0.5).astype(int)

    Jp = np.array(LA.det(F[p]), dtype = np.complex)
    b = F[p].dot(F[p].transpose())
    det_b = np.array(LA.det(b), dtype = np.complex)
    normalized_b = pow(det_b, -1/3) * b
    tr = normalized_b[0,0] + normalized_b[1,1] + normalized_b[2,2]
    dev = normalized_b - (tr/3)*np.identity(3)
    stress = np.zeros((3,3))
    if Jp != 0 and not np.isinf(Jp) and not np.isnan(Jp) and not np.isinf(tr) and not np.isnan(tr):
      stress = (1/Jp) * (kappa/2 * (Jp*Jp-1)*np.identity(3) + mu*dev)
      
    if weak[p] == 1:  
      eigenvalue = np.ndarray((dim, dim), dtype=np.complex64)
      w, eigenvector = LA.eig(stress)
      for i in range(dim):
        if w[i] > 0:
          eigenvalue[i, i] = 0
        else:
          eigenvalue[i, i] = w[i]
      stress = eigenvector.dot(eigenvalue).dot(eigenvector.transpose())
    
    for i in range(3): # Loop over 3x3 grid node neighborhood
      for j in range(3):
        for k in range(3):
          offset = np.array([i,j,k])
          index = base + offset
          weight = np_wip(index[0], index[1], index[2], x[p])
          if weight > 0:
            wip_g = np_wip_grad(index[0], index[1], index[2], x[p])
            grid_f[index[0], index[1], index[2]] = grid_f[index[0], index[1], index[2]] - Jp*vol[p]*stress.dot(wip_g)

@ti.kernel
def Update_Grid_V():

  for i, j, k in grid_m:
    if grid_m[i, j, k] > 0:  # No need for epsilon here
      new_grid_v[i, j, k] = grid_v[i, j, k]
      new_grid_v[i, j, k] += dt * grid_f[i, j, k]/grid_m[i, j, k]# * 1000
      new_grid_v[i, j, k][1] += dt * gravity
      # print(new_grid_v[i, j, k])
      
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

def np_Update_Limit_Deformation_Gradient(x, F, grid_v, b_pre):
  for p in range(n_particles):
    base = (x[p] * inv_dx - 0.5).astype(int)
    v_g = np.zeros((3,3))

    for i in range(3): # Loop over 3x3 grid node neighborhood
      for j in range(3):
        for k in range(3):
          offset = np.array([i,j,k])
          index = base + offset
          weight = np_wip((base + offset)[0], (base + offset)[1], (base + offset)[2], x[p])
          if weight > 0:
            weight_v = grid_v[index[0], index[1], index[2]].dot(np_wip_grad(index[0], index[1], index[2], x[p]).transpose())
            v_g = v_g + weight_v
   
    fp = np.identity(3) + dt*v_g
    det_fp = np.array(LA.det(fp), dtype = np.complex)
    fp_normal = (pow(det_fp, -1/3)) * fp
    b = F[p].dot(F[p].transpose())
    det_b = np.array(LA.det(b), dtype = np.complex)
    normalized_b = np.identity(3)
    weight = (pow(det_b, -1/3))
    if not np.isnan(weight) and not np.isinf(weight):
      normalized_b = weight * b
    b_pre = fp_normal.dot(normalized_b).dot(fp_normal.transpose())
    prev_F = F[p]
    if weak[p] == 0:
      F[p] = fp.dot(F[p])
    
    else:
      eig_value, eigenvector = LA.eig(normalized_b)
      eig_value_pre, eigenvector_pre = LA.eig(b_pre)
      max_val = np.amax(eig_value)
      max_pre_val = np.amax(eig_value_pre)
      f_corr = fp
      if max_pre_val > max_val:
        U, sig, V = LA.svd(fp_normal)
        fp_normal = U.dot(V.transpose())
        b_pre = fp_normal.dot(normalized_b).dot(fp_normal.transpose())
        det_fp = np.array(LA.det(fp), dtype = np.complex)
        f_corr = (pow(det_fp, -1/3)) * fp_normal
    
      Jn = np.array(LA.det(prev_F), dtype = np.complex)
      Jcorr = np.array(LA.det(f_corr), dtype = np.complex)
      f_corr = np.identity(3)
      a = Jcorr**(-1/3)
      if Jn*Jcorr>1 and Jcorr>1 and not np.isnan(a) and np.isinf(a):
        f_corr = Jcorr**(-1/3) * f_corr
      F[p] = f_corr.dot(prev_F)

@ti.kernel
def Update_Particle_V():
  for p in x:
    base = (x[p] * inv_dx - 0.5).cast(int)
    v_PIC = ti.Matrix.zero(ti.f32, dim)
    v_FLIP = v[p]

    for i, j ,k in ti.static(ti.ndrange(3, 3, 3)): # Loop over 3x3 grid node neighborhood
      offset = ti.Vector([i, j, k])
      weight = wip((base + offset)[0], (base + offset)[1], (base + offset)[2], x[p])
      if weight > 0:
        v_PIC = v_PIC + weight * new_grid_v[base + offset]
        v_FLIP = v_FLIP + weight * (new_grid_v[base + offset] - grid_v[base + offset])
    alpha = 0.95
    v[p] = (1-alpha)*v_PIC + alpha*v_FLIP

def np_Plastic_Flow(b_pre, F):
  #1:
  for p in range(n_particles):
    tr = b_pre[0,0] + b_pre[1,1] + b_pre[2,2]
    dev = b_pre - (tr/3)*np.identity(3)
    #2:
    s_pre = mu * dev
    s_pre_sqr = s_pre.dot(s_pre)
    #3:
    norm_s_pre = math.sqrt(s_pre_sqr[0,0] + s_pre_sqr[1,1] + s_pre_sqr[2,2])
    norm_s = 0
    y_s = math.sqrt(2/3)*yield_stress
    b = F[p].dot(F[p].transpose())
    #4 5:
    if norm_s_pre - y_s > 0:
      #7:
      mu_prime = (1/3)*tr*mu
      #8:
      if eta == 0 or h == 1:
        #9:
        norm_s = norm_s_pre - ((norm_s_pre-y_s) / (1+eta/(2*mu_prime*dt)))
      #10:
      else:
        #11:
        s_min = y_s
        s_max = norm_s_pre
        #12:
        while True:
          #13:
          s = (s_min+s_max)/2
          #14:
          norm_s = pow(eta,(1/h))*(s-norm_s_pre) + 2*mu_prime*dt*(pow((s-y_s),(1/h)))
          #15:
          if norm_s < 0:
            #16:
            s_min = s
          #17:
          else:
            #18:
            s_max = s
          E = 1
          if norm_s_pre > 0:
            E = norm_s/norm_s_pre
          #21:
          if abs(E) < 0.000001:
            break
      flow = (1 / norm_s_pre) * s_pre
      s = norm_s * flow
      b = (1/mu)*s + (tr/3)*np.identity(3)
      det_b = np.array(LA.det(b), dtype = np.complex)
      b = pow(det_b, -1/3)*b
    
    det_F = np.array(LA.det(F[p]), dtype = np.complex)
    Cpp = np.identity(3)
    inv_b = LA.inv(b)
    weight = pow(det_F, -2/3)
    a = F[p].transpose().dot(inv_b).dot(F[p])

    det_a = LA.det(a)
    if not np.isinf(weight) and not np.isnan(weight) and not np.isinf(det_a) and not np.isnan(det_a):
      Cpp = weight * a
    eigenvalue = np.ndarray((dim, dim), dtype=np.complex64)
    w, eigenvector = LA.eig(Cpp)
    
    # for i in range(3):
    #   eigenvalue[i, i] = w[i]**((math.exp(-dt/eta_p)-1)/2)
    eigenvalue = np.diag(np.power(w, (math.exp(-dt/eta_p)-1)/2))
    Cpp = eigenvector.dot(eigenvalue).dot(eigenvector.transpose())
    F[p] = F[p].dot(Cpp)

@ti.kernel
def Particle_Position():
  for p in x:
    x[p] = x[p] + dt*v[p]


def substep():
  print("P2G")
  P2G()
  print("Dectect_Tearing_Part")
  Dectect_Tearing_Part()

  print("to_numpy")
  np_x = x.to_numpy()
  np_v = v.to_numpy()
  np_F = F.to_numpy()
  np_grid_f = grid_f.to_numpy()
  print("np_Grid_Force")
  np_Grid_Force(np_x, np_v, np_F, np_grid_f)
  print("from_numpy")
  x.from_numpy(np_x)
  v.from_numpy(np_v)
  F.from_numpy(np_F)
  grid_f.from_numpy(np_grid_f)

  print("Update_Grid_V")
  Update_Grid_V()

  print("to_numpy")
  np_x = x.to_numpy()
  np_F = F.to_numpy()
  np_grid_v = grid_v.to_numpy()
  print("np_Update_Limit_Deformation_Gradient")
  np_Update_Limit_Deformation_Gradient(np_x, np_F, np_grid_v, b_pre)
  print("from_numpy")
  x.from_numpy(np_x)
  F.from_numpy(np_F)
  grid_v.from_numpy(np_grid_v)

  print("Update_Particle_V")
  Update_Particle_V()
  
  print("to_numpy")
  np_F = F.to_numpy()
  print("np_Plastic_Flow")
  np_Plastic_Flow(b_pre, np_F)
  print("from_numpy")
  F.from_numpy(np_F)

  print("Particle_Position")
  Particle_Position()

def particle_log(x, v, F):
  for p in range(n_particles):
    f.write('particle '+str(p)+'\n')
    f.write('x:\n')
    f.write(str(x[p][0])+" "+str(x[p][1])+" "+str(x[p][2])+"\n")
    f.write('v:\n')
    f.write(str(v[p][0])+" "+str(v[p][1])+" "+str(v[p][2])+"\n")
    if(abs(v[p][1]) < 0.001):
      f.write('small v\n')
    f.write('F:\n')
    for i in range(3):
      f.write(str(F[p][i,0])+" "+str(F[p][i,1])+" "+str(F[p][i,2])+"\n")

group_size = n_particles#//2
@ti.kernel
def initialize():
  length = 0.1
  for i in range(n_particles):
    x_start = ti.random() * length
    y_start = ti.random() * length
    z_start = ti.random() * length
    x_start -= length/2
    y_start -= length/2
    z_start -= length/2
    while x_start*x_start + y_start*y_start + z_start*z_start > length**2:
      x_start = ti.random() * length
      y_start = ti.random() * length
      z_start = ti.random() * length
      x_start -= length/2
      y_start -= length/2
      z_start -= length/2
    x_v = x_start + 0.3 + 0.3 * (i // group_size)
    y_v = y_start + 0.6
    z_v = z_start + 0.3 + 0.3 * (i // group_size)
    # if i // group_size == 0 or i // group_size == 1:
    #   x_v = x_start + 0.3
    #   y_v = ti.random() * 0.8 + 0.001
    pos = ti.Vector([x_v, y_v, z_v])
    for d in ti.static(range(dim)):
      x[i][d] = pos[d]

    v[i] = ti.Matrix.zero(ti.f32, dim)
    if i // group_size > 0:
      v[i][0] = -100
    
    F[i] = ti.Matrix.identity(ti.f32, dim)
    Jp[i] = 1
    p_m[i] = (dx * 0.5)**2

print("P2G")
P2G()
print("P2G Done")
Compute_Particle_Volumes_and_Densities()
print("Compute_Particle_Volumes_and_Densities Done")

@ti.kernel
def copy_dynamic_nd(np_x: ti.ext_arr(), input_x: ti.template()):
  for i in x:
    for j in ti.static(range(dim)):
      np_x[i, j] = input_x[i][j]

initialize()
gui = ti.GUI("MPM-Foam", res=1024 * 0.8, background_color=0x000000)

for frame in range(20000):
  f.close()
  f = open(f'{frame:06d}.txt', 'w')
  f.write("Frame " + str(frame)+'\n')
  for s in range(int(2e-3 // dt)):
    substep()
  
  
  np_x = np.ndarray((n_particles, dim), dtype=np.float32)
  copy_dynamic_nd(np_x, x)
  screen_x = (np_x[:, 0])
  if dim == 3:
    screen_x = ((np_x[:, 0] + np_x[:, 2]) / 2**0.5) - 0.2    
  screen_y = (np_x[:, 1])

  screen_pos = np.stack([screen_x, screen_y], axis=-1)
  gui.circles(screen_pos, radius=1.5, color=0xEEEEF0)
  gui.show(f'{frame:06d}.png' if write_to_disk else None)
  np_x = x.to_numpy()
  np_v = v.to_numpy()
  np_F = F.to_numpy()
  particle_log(np_x, np_v, np_F)
  #gui.show() # Change to gui.show(f'{frame:06d}.png') to write images to disk