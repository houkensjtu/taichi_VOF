import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, default_fp=ti.f64)

nx = 32  # Number of grid points in the x direction
ny = 32  # Number of grid points in the y direction
Lx = 1.0  # The length of the domain
Ly = 1.0  # The width of the domain

imin = 1
imax = imin + nx - 1
jmin = 1
jmax = jmin + ny - 1
Fgrad = ti.Vector.field(2, float, shape=(imax+2, jmax+2))
F = ti.field(float, shape=(imax+2, jmax+2))
x = ti.field(float, shape=imax + 3)
y = ti.field(float, shape=jmax + 3)
x.from_numpy(np.hstack((0.0, np.linspace(0, Lx, nx + 1), Lx)))  # [0, 0, ... 1, 1]
y.from_numpy(np.hstack((0.0, np.linspace(0, Ly, ny + 1), Ly)))  # [0, 0, ... 1, 1]
xm = ti.field(float, shape=imax + 2)
ym = ti.field(float, shape=jmax + 2)
dx = x[imin + 2] - x[imin + 1]
dy = y[jmin + 2] - y[jmin + 1]
dxi = 1 / dx
dyi = 1 / dy

@ti.kernel
def init():
    for i,j in F:
        F[i, j] = 1.0

@ti.kernel
def cal_fgrad():  # 11/3 Checked, no out-of-range
    '''
    Calculate the Fgrad in internal area.
    '''
    for i, j in ti.ndrange((imin, imax+1), (jmin, jmax+1)):
        Fx = (F[i+1, j-1] + F[i+1, j] + F[i+1, j+1]) - (F[i-1, j-1] + F[i-1, j] + F[i-1, j+1])
        Fy = (F[i+1, j+1] + F[i, j+1] + F[i-1, j+1]) - (F[i+1, j-1] + F[i, j-1] + F[i-1, j-1])
        dfdx = 0.5 * Fx / dx
        dfdy = 0.5 * Fy / dy
        Fgrad[i, j] = ti.Vector([dfdx, dfdy])


init()        
cal_fgrad()        


