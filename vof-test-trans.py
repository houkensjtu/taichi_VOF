import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from math import pi as pi
import os

ti.init(arch=ti.cpu, default_fp=ti.f32, debug=True)

SAVE_FIG = True
SAVE_DAT = False

nx = 100  # Number of grid points in the x direction
ny = 100 # Number of grid points in the y direction

Lx = pi  # The length of the domain
Ly = pi  # The width of the domain

# Solution parameters
dt = 1e-4  # Use smaller dt for higher density ratio
eps = 1e-6  # Threshold used in vfconv and f post processings

imin = 1
imax = imin + nx - 1
jmin = 1
jmax = jmin + ny - 1

F = ti.field(float, shape=(imax + 2, jmax + 2))
Fn = ti.field(float, shape=(imax + 2, jmax + 2))
Ftd = ti.field(float, shape=(imax + 2, jmax + 2))
Fgrad = ti.Vector.field(2, float, shape=(imax + 2, jmax + 2))
ax = ti.field(float, shape=(imax + 2, jmax + 2))
ay = ti.field(float, shape=(imax + 2, jmax + 2))
cx = ti.field(float, shape=(imax + 2, jmax + 2))
cy = ti.field(float, shape=(imax + 2, jmax + 2))
rp = ti.field(float, shape=(imax + 2, jmax + 2))
rm = ti.field(float, shape=(imax + 2, jmax + 2))

F = ti.field(float, shape=(imax + 2, jmax + 2))
u = ti.field(float, shape=(imax + 2, jmax + 2))
v = ti.field(float, shape=(imax + 2, jmax + 2))
vdiv = ti.field(float, shape=(imax + 2, jmax + 2))

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

print(f'>>> VOF scheme testing')
print(f'>>> Grid resolution: {nx} x {ny}, dt = {dt:4.2e}')


@ti.kernel
def grid_staggered():  # 11/3 Checked
    '''
    Calculate the center position of cells.
    '''
    for i in xm:  # xm[0] = 0.0, xm[33] = 1.0
        xm[i] = 0.5 * (x[i] + x[i + 1])
    for j in ym:
        ym[j] = 0.5 * (y[j] + y[j + 1])


@ti.func
def find_area(i, j, cx, cy, r):
    a = 0.0
    xcoord_ct = (i - imin) * dx + dx / 2
    ycoord_ct = (j - jmin) * dy + dy / 2
    
    xcoord_lu = xcoord_ct - dx / 2
    ycoord_lu = ycoord_ct + dy / 2
    
    xcoord_ld = xcoord_ct - dx / 2
    ycoord_ld = ycoord_ct - dy / 2
    
    xcoord_ru = xcoord_ct + dx / 2
    ycoord_ru = ycoord_ct + dy / 2
    
    xcoord_rd = xcoord_ct + dx / 2
    ycoord_rd = ycoord_ct - dy / 2

    dist_ct = ti.sqrt((xcoord_ct - cx) ** 2 + (ycoord_ct - cy) ** 2)
    dist_lu = ti.sqrt((xcoord_lu - cx) ** 2 + (ycoord_lu - cy) ** 2)
    dist_ld = ti.sqrt((xcoord_ld - cx) ** 2 + (ycoord_ld - cy) ** 2)
    dist_ru = ti.sqrt((xcoord_ru - cx) ** 2 + (ycoord_ru - cy) ** 2)
    dist_rd = ti.sqrt((xcoord_rd - cx) ** 2 + (ycoord_rd - cy) ** 2)

    if dist_lu > r and dist_ld > r and dist_ru > r and dist_rd > r:
        a = 1.0
    elif dist_lu < r and dist_ld < r and dist_ru < r and dist_rd < r:
        a = 0.0
    else:
        a = 0.5 + 0.5 * (dist_ct - r) / (ti.sqrt(2.0) * dx)
        a = var(a, 0, 1)
        
    return a

        
@ti.kernel
def set_init_F():
    # Sets the initial volume fraction
    for I in ti.grouped(F):
        F[I] = 1.0
        Fn[I] = 1.0
    '''
    # Dambreak
    # The initial volume fraction of the domain
    x1 = 0.0
    x2 = Lx / 2
    y1 = 0.0
    y2 = Ly / 3
    for i, j in F:  # [0,33], [0,33]
        if (xm[i] >= x1) and (xm[i] <= x2) and (ym[j] >= y1) and (ym[j] <= y2):
            F[i, j] = 1.0
            Fn[i, j] = F[i, j]

    # Moving square
    for i, j in F:
        x = xm[i]
        y = ym[j]
        cx, cy = 0.05, 0.02
        l = 0.01
        if ( ti.abs(x - cx) < l) and ( ti.abs(y - cy) < l):
            F[i, j] = 0.0
            Fn[i, j] = 0.0
    
    # Moving circle
    for i, j in F:
        x = xm[i]
        y = ym[j]
        cx, cy = Lx / 2, Ly * 3 / 4
        r = Lx / 10
        F[i, j] = find_area(i, j, cx, cy, r)
        Fn[i, j] = find_area(i, j, cx, cy, r)
            
    '''        
    # Slot disk
    for i, j in F:
        x = xm[i]
        y = ym[j]
        cx, cy = Lx / 2, Ly * 3 / 4
        r = Lx / 10
        F[i, j] = find_area(i, j, cx, cy, r)
        Fn[i, j] = find_area(i, j, cx, cy, r)
        sw = r / 6.0
        sh = r * 0.8
        if ti.abs(x - cx) < sw and ti.abs(y - cy + r / 4) < sh:
            F[i, j] = 1.0
            Fn[i, j] = 1.0



@ti.kernel
def init_uv():
    '''            
    for I in ti.grouped(u):
        u[I] = 0.01
        v[I] = 0.01
    '''
    # Zalesak's slot disk
    w = 0.3
    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        ux = xm[i] - dx / 2
        uy = ym[j]
        vx = xm[i]
        vy = ym[j] - dy / 2
        u[i, j] = - w * (uy - Ly / 2)
        v[i, j] = w * (vx - Lx / 2)
    '''
    # Kother Rider test
    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        ux = xm[i] - dx / 2
        uy = ym[j]
        vx = xm[i]
        vy = ym[j] - dy / 2
        # u[i, j] = ti.cos(ux) * ti.sin(uy)
        # v[i, j] = - ti.sin(vx) * ti.cos(vy)
        u[i, j] = - ti.sin(ux) ** 2 * ti.sin(2 * uy)
        v[i, j] = ti.sin(vy) ** 2 * ti.sin(2 * vx)        
    '''
            
@ti.kernel
def set_BC():
    for i in ti.ndrange(imax + 2):
        # bottom: slip 
        u[i, jmin - 1] = u[i, jmin]
        v[i, jmin] = v[i, jmin + 1]
        F[i, jmin - 1] = F[i, jmin]
        # top: open
        u[i, jmax + 1] = u[i, jmax]
        v[i, jmax + 1] = v[i, jmax]
        F[i, jmax + 1] = F[i, jmax]

    for j in ti.ndrange(jmax + 2):
        # left: slip
        u[imin, j] = u[imin + 1, j]
        v[imin - 1, j] = v[imin, j]
        F[imin - 1, j] = F[imin, j]
        # right: slip
        u[imax + 1, j] = u[imax, j]
        v[imax + 1, j] = v[imax, j]
        F[imax + 1, j] = F[imax, j]


@ti.kernel
def cal_vdiv()->float:
    d = 0.0
    ti.loop_config(serialize=True)
    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        vdiv[i, j] = ti.abs(u[i+1,j] - u[i,j] + v[i,j+1] - v[i,j])
        d += ti.abs(u[i+1,j] - u[i,j] + v[i,j+1] - v[i,j])
    return d


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


@ti.func
def var(a, b, c):
    # Find the median of a,b, and c
    center = a + b + c - ti.max(a, b, c) - ti.min(a, b, c)
    return center


@ti.kernel
def solve_VOF_upwind():
    # FCT Method described in Rudman's 1997 paper
    for I in ti.grouped(Fn):
        Fn[I] = F[I]

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        fl = u[i, j] * dt * Fn[i - 1, j] if u[i, j] > 0 else u[i, j] * dt * Fn[i, j]
        fr = u[i + 1, j] * dt * Fn[i, j] if u[i + 1, j] > 0 else u[i + 1, j] * dt * Fn[i + 1, j]
        ft = v[i, j + 1] * dt * Fn[i, j] if v[i, j + 1] > 0 else v[i, j + 1] * dt * Fn[i, j + 1]
        fb = v[i, j] * dt * Fn[i, j - 1] if v[i, j] > 0 else v[i, j] * dt * Fn[i, j]
        F[i, j] += (fl - fr + fb - ft) * dy / (dx * dy)


@ti.kernel
def solve_VOF_zalesak():
    for I in ti.grouped(Fn):
        Fn[I] = F[I]
        
    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        fl_L = u[i, j] * dt * Fn[i - 1, j] if u[i, j] >= 0 else u[i, j] * dt * Fn[i, j]
        fr_L = u[i + 1, j] * dt * Fn[i, j] if u[i + 1, j] >= 0 else u[i + 1, j] * dt * Fn[i + 1, j]
        ft_L = v[i, j + 1] * dt * Fn[i, j] if v[i, j + 1] >= 0 else v[i, j + 1] * dt * Fn[i, j + 1]
        fb_L = v[i, j] * dt * Fn[i, j - 1] if v[i, j] >= 0 else v[i, j] * dt * Fn[i, j]
        Ftd[i, j] = Fn[i, j] + (fl_L - fr_L + fb_L - ft_L) * dy / (dx * dy)
        if Ftd[i, j] > 1. or Ftd[i, j] < 0:
            Ftd[i, j] = var(0, 1, Ftd[i, j])
        
    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        fmax = ti.max(Ftd[i, j], Ftd[i - 1, j], Ftd[i + 1, j], Ftd[i, j + 1], Ftd[i, j - 1])
        fmin = ti.min(Ftd[i, j], Ftd[i - 1, j], Ftd[i + 1, j], Ftd[i, j + 1], Ftd[i, j - 1])
        
        fl_L = u[i, j] * dt * Fn[i - 1, j] if u[i, j] >= 0 else u[i, j] * dt * Fn[i, j]
        fr_L = u[i + 1, j] * dt * Fn[i, j] if u[i + 1, j] >= 0 else u[i + 1, j] * dt * Fn[i + 1, j]
        ft_L = v[i, j + 1] * dt * Fn[i, j] if v[i, j + 1] >= 0 else v[i, j + 1] * dt * Fn[i, j + 1]
        fb_L = v[i, j] * dt * Fn[i, j - 1] if v[i, j] >= 0 else v[i, j] * dt * Fn[i, j]
        
        fl_H = u[i, j] * dt * Fn[i - 1, j] if u[i, j] < 0 else u[i, j] * dt * Fn[i, j]
        fr_H = u[i + 1, j] * dt * Fn[i, j] if u[i + 1, j] < 0 else u[i + 1, j] * dt * Fn[i + 1, j]
        ft_H = v[i, j + 1] * dt * Fn[i, j] if v[i, j + 1] < 0 else v[i, j + 1] * dt * Fn[i, j + 1]
        fb_H = v[i, j] * dt * Fn[i, j - 1] if v[i, j] < 0 else v[i, j] * dt * Fn[i, j]

        ax[i + 1, j] = fr_H - fr_L
        ax[i, j] = fl_H - fl_L
        ay[i, j + 1] = ft_H - ft_L
        ay[i, j] = fb_H - fb_L

        pp = ti.max(0, ax[i, j]) - ti.min(0, ax[i + 1, j]) + ti.max(0, ay[i, j]) - ti.min(0, ay[i, j + 1])
        qp = (fmax - Ftd[i, j]) * dx
        if pp > 0:
            rp[i, j] = ti.min(1, qp / pp)
        else:
            rp[i, j] = 0.0
        pm = ti.max(0, ax[i + 1, j]) - ti.min(0, ax[i, j]) + ti.max(0, ay[i, j + 1]) - ti.min(0, ay[i, j])
        qm = (Ftd[i, j] - fmin) * dx
        if pm > 0:
            rm[i, j] = ti.min(1, qm / pm)
        else:
            rm[i, j] = 0.0
            
        if ax[i + 1, j] >= 0:
            cx[i + 1, j] = ti.min(rp[i + 1, j], rm[i, j])
        else:
            cx[i + 1, j] = ti.min(rp[i, j], rm[i + 1, j])

        if ay[i, j + 1] >= 0:
            cy[i, j + 1] = ti.min(rp[i, j + 1], rm[i, j])
        else:
            cy[i, j + 1] = ti.min(rp[i, j], rm[i, j + 1])

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        F[i, j] = Ftd[i, j] - (ax[i + 1, j] * cx[i + 1, j] - \
                               ax[i, j] * cx[i, j] + \
                               ay[i, j + 1] * cy[i, j + 1] -\
                               ay[i, j] * cy[i, j]) / (dy)


def solve_VOF_rudman():
    '''
    fct_x_sweep()
    '''
    if istep % 2 == 0:
        fct_y_sweep()
        fct_x_sweep()
    else:
        fct_x_sweep()
        fct_y_sweep()


@ti.kernel
def fct_x_sweep():
    for I in ti.grouped(Fn):
        Fn[I] = F[I]
    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        dv = dx * dy - dt * dy * (u[i + 1, j] - u[i, j])
        fl_L = u[i, j] * dt * Fn[i - 1, j] if u[i, j] >= 0 else u[i, j] * dt * Fn[i, j]
        fr_L = u[i + 1, j] * dt * Fn[i, j] if u[i + 1, j] >= 0 else u[i + 1, j] * dt * Fn[i + 1, j]
        ft_L = 0
        fb_L = 0
        Ftd[i, j] = Fn[i, j] + (fl_L - fr_L + fb_L - ft_L) * dy / (dx * dy)
        Ftd[i, j] = Ftd[i, j] * dx * dy / dv
        # if u[i, j] != u[i + 1, j]:
        #     print('>>> u[i, j] = ', u[i, j], 'u[i + 1, j] = ', u[i + 1, j], ' fl_L = ', fl_L, 'fr_L = ', fr_L, 'at ', i, j)
        if Ftd[i, j] > 1. or Ftd[i, j] < 0:
            Ftd[i, j] = var(0, 1, Ftd[i, j])
        
    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        # fmax = ti.max(Ftd[i, j], Ftd[i - 1, j], Ftd[i + 1, j], Ftd[i, j + 1], Ftd[i, j - 1])
        fmax = ti.max(Ftd[i, j], Ftd[i - 1, j], Ftd[i + 1, j])  # , Ftd[i, j + 1], Ftd[i, j - 1])        
        # fmin = ti.min(Ftd[i, j], Ftd[i - 1, j], Ftd[i + 1, j], Ftd[i, j + 1], Ftd[i, j - 1])
        fmin = ti.min(Ftd[i, j], Ftd[i - 1, j], Ftd[i + 1, j])  # , Ftd[i, j + 1], Ftd[i, j - 1])        
        
        fl_L = u[i, j] * dt * Fn[i - 1, j] if u[i, j] >= 0 else u[i, j] * dt * Fn[i, j]
        fr_L = u[i + 1, j] * dt * Fn[i, j] if u[i + 1, j] >= 0 else u[i + 1, j] * dt * Fn[i + 1, j]
        ft_L = 0
        fb_L = 0
        
        fl_H = u[i, j] * dt * Fn[i - 1, j] if u[i, j] <= 0 else u[i, j] * dt * Fn[i, j]
        fr_H = u[i + 1, j] * dt * Fn[i, j] if u[i + 1, j] <= 0 else u[i + 1, j] * dt * Fn[i + 1, j]
        ft_H = 0
        fb_H = 0

        ax[i + 1, j] = fr_H - fr_L
        ax[i, j] = fl_H - fl_L
        ay[i, j + 1] = 0  # ft_H - ft_L
        ay[i, j] = 0  # fb_H - fb_L

        pp = ti.max(0, ax[i, j]) - ti.min(0, ax[i + 1, j]) + ti.max(0, ay[i, j]) - ti.min(0, ay[i, j + 1])
        qp = (fmax - Ftd[i, j]) * dx
        if pp > 0:
            rp[i, j] = ti.min(1, qp / pp)
        else:
            rp[i, j] = 0.0
        pm = ti.max(0, ax[i + 1, j]) - ti.min(0, ax[i, j]) + ti.max(0, ay[i, j + 1]) - ti.min(0, ay[i, j])
        qm = (Ftd[i, j] - fmin) * dx
        if pm > 0:
            rm[i, j] = ti.min(1, qm / pm)
        else:
            rm[i, j] = 0.0

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):            
        if ax[i + 1, j] >= 0:
            cx[i + 1, j] = ti.min(rp[i + 1, j], rm[i, j])
        else:
            cx[i + 1, j] = ti.min(rp[i, j], rm[i + 1, j])

        if ay[i, j + 1] >= 0:
            cy[i, j + 1] = ti.min(rp[i, j + 1], rm[i, j])
        else:
            cy[i, j + 1] = ti.min(rp[i, j], rm[i, j + 1])

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        dv = dx * dy - dt * dy * (u[i + 1, j] - u[i, j])        
        F[i, j] = Ftd[i, j] - ((ax[i + 1, j] * cx[i + 1, j] - \
                               ax[i, j] * cx[i, j] + \
                               ay[i, j + 1] * cy[i, j + 1] -\
                               ay[i, j] * cy[i, j]) / (dy)) * dx * dy / dv
        F[i, j] = var(0, 1, F[i, j])


@ti.kernel
def fct_y_sweep():

    for I in ti.grouped(Fn):
        Fn[I] = F[I]

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        dv = dx * dy - dt * dx * (v[i, j + 1] - v[i, j])        
        fl_L = 0
        fr_L = 0
        ft_L = v[i, j + 1] * dt * Fn[i, j] if v[i, j + 1] >= 0 else v[i, j + 1] * dt * Fn[i, j + 1]
        fb_L = v[i, j] * dt * Fn[i, j - 1] if v[i, j] >= 0 else v[i, j] * dt * Fn[i, j]
        Ftd[i, j] = Fn[i, j] + (fl_L - fr_L + fb_L - ft_L) * dy / (dx * dy)
        Ftd[i, j] = Ftd[i, j] * dx * dy / dv
        if Ftd[i, j] > 1. or Ftd[i, j] < 0:
            Ftd[i, j] = var(0, 1, Ftd[i, j])

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        # fmax = ti.max(Ftd[i, j], Ftd[i - 1, j], Ftd[i + 1, j], Ftd[i, j + 1], Ftd[i, j - 1])
        fmax = ti.max(Ftd[i, j], Ftd[i, j - 1], Ftd[i, j + 1])
        # fmin = ti.min(Ftd[i, j], Ftd[i - 1, j], Ftd[i + 1, j], Ftd[i, j + 1], Ftd[i, j - 1])
        fmin = ti.min(Ftd[i, j], Ftd[i, j - 1], Ftd[i, j + 1]) 
        
        fl_L = 0
        fr_L = 0
        ft_L = v[i, j + 1] * dt * Fn[i, j] if v[i, j + 1] >= 0 else v[i, j + 1] * dt * Fn[i, j + 1]
        fb_L = v[i, j] * dt * Fn[i, j - 1] if v[i, j] >= 0 else v[i, j] * dt * Fn[i, j]
        
        fl_H = 0
        fr_H = 0
        ft_H = v[i, j + 1] * dt * Fn[i, j] if v[i, j + 1] <= 0 else v[i, j + 1] * dt * Fn[i, j + 1]
        fb_H = v[i, j] * dt * Fn[i, j - 1] if v[i, j] <= 0 else v[i, j] * dt * Fn[i, j]

        ax[i + 1, j] = 0  # fr_H - fr_L
        ax[i, j] = 0  # fl_H - fl_L
        ay[i, j + 1] = ft_H - ft_L
        ay[i, j] = fb_H - fb_L

        pp = ti.max(0, ax[i, j]) - ti.min(0, ax[i + 1, j]) + ti.max(0, ay[i, j]) - ti.min(0, ay[i, j + 1])
        qp = (fmax - Ftd[i, j]) * dx
        if pp > 0:
            rp[i, j] = ti.min(1, qp / pp)
        else:
            rp[i, j] = 0.0
        pm = ti.max(0, ax[i + 1, j]) - ti.min(0, ax[i, j]) + ti.max(0, ay[i, j + 1]) - ti.min(0, ay[i, j])
        qm = (Ftd[i, j] - fmin) * dx
        if pm > 0:
            rm[i, j] = ti.min(1, qm / pm)
        else:
            rm[i, j] = 0.0

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):        
        if ax[i + 1, j] >= 0:
            cx[i + 1, j] = ti.min(rp[i + 1, j], rm[i, j])
        else:
            cx[i + 1, j] = ti.min(rp[i, j], rm[i + 1, j])

        if ay[i, j + 1] >= 0:
            cy[i, j + 1] = ti.min(rp[i, j + 1], rm[i, j])
        else:
            cy[i, j + 1] = ti.min(rp[i, j], rm[i, j + 1])


    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        dv = dx * dy - dt * dx * (v[i, j + 1] - v[i, j])                
        F[i, j] = Ftd[i, j] - ((ax[i + 1, j] * cx[i + 1, j] - \
                               ax[i, j] * cx[i, j] + \
                               ay[i, j + 1] * cy[i, j + 1] -\
                               ay[i, j] * cy[i, j]) / (dy)) * dx * dy / dv
        F[i, j] = var(0, 1, F[i, j])

        
@ti.kernel
def solve_VOF_sola():
    # Method described in original VOF paper
    for I in ti.grouped(Fn):
        Fn[I] = F[I]
        
    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        f_a, f_d, f_ad, f_up = 0.0, 0.0, 0.0, 0.0
        # Flux left
        if u[i, j] > 0:
            f_d, f_a, f_up = Fn[i-1, j], Fn[i, j], Fn[ti.max(0,i-2) ,j]
        else:
            f_a, f_d, f_up = Fn[i-1, j], Fn[i, j], Fn[i+1, j]
        if ti.abs(Fgrad[i, j][0]) > ti.abs(Fgrad[i,j][1]):  # Surface orientation is vertical
            f_ad = f_a
        elif f_a < eps or f_up < eps:
            f_ad = f_a
        else:  # Surface orientation is horizontal
            f_ad = f_d
        fdm = ti.max(f_d, f_up)
        V = u[i, j] * dt
        CF = ti.max((fdm - f_ad) * ti.abs(V) - (fdm - f_d) * dx, 0.0)
        flux_l = ti.min(f_ad * ti.abs(V) / dx + CF / dx, f_d) * (u[i,j]) / (ti.abs(u[i,j]) + 1e-16)
        
        # Flux right
        if u[i+1, j] > 0:
            f_d, f_a, f_up = Fn[i, j], Fn[i+1, j], Fn[i-1, j]
        else:
            f_a, f_d, f_up = Fn[i, j], Fn[i+1, j], Fn[ti.min(i+2, imax+1) ,j]
        if ti.abs(Fgrad[i, j][0]) >  ti.abs(Fgrad[i,j][1]):  # Surface orientation is vertical
            f_ad = f_a
        elif f_a < eps or f_up < eps:
            f_ad = f_a
        else:  # Surface orientation is horizontal
            f_ad = f_d
        fdm = ti.max(f_d, f_up)            
        V = u[i+1, j] * dt
        CF = ti.max((fdm - f_ad) * ti.abs(V) - (fdm - f_d) * dx, 0.0)
        flux_r = ti.min(f_ad * ti.abs(V) / dx + CF / dx, f_d) * (u[i+1, j]) / (ti.abs(u[i+1, j]) + 1e-16)
        if i == imax:
            flux_r = 0.0
        
        # Flux top
        if v[i, j + 1] > 0:
            f_d, f_a, f_up = Fn[i, j], Fn[i, j + 1], Fn[i, j-1]
        else:
            f_a, f_d, f_up = Fn[i, j], Fn[i, j + 1], Fn[i, ti.min(j+2, jmax+1)]
        if ti.abs(Fgrad[i, j][0])  > ti.abs(Fgrad[i,j][1]):  # Surface orientation is vertical
            f_ad = f_a
        elif f_a < eps or f_up < eps:
            f_ad = f_a
        else:  # Surface orientation is horizontal
            f_ad = f_d
        fdm = ti.max(f_d, f_up)                        
        V = v[i, j + 1] * dt
        CF = ti.max((fdm - f_ad) * ti.abs(V) - (fdm - f_d) * dx, 0.0)
        flux_t = ti.min(f_ad * ti.abs(V) / dx + CF / dx, f_d) * (v[i,j+1]) / (ti.abs(v[i, j+1]) + 1e-16)
        
        # Flux bottom
        if v[i, j] > 0:
            f_d, f_a, f_up = Fn[i, j-1], Fn[i, j], Fn[i, ti.max(0, j-2)]
        else:
            f_a, f_d, f_up = Fn[i, j-1], Fn[i, j], Fn[i, j+1]
        if ti.abs(Fgrad[i, j][0]) > ti.abs(Fgrad[i,j][1]):  # Surface orientation is vertical
            f_ad = f_a
        elif f_a < eps or f_up < eps:
            f_ad = f_a
        else:  # Surface orientation is horizontal
            f_ad = f_d
        fdm = ti.max(f_d, f_up)                                    
        V = v[i, j] * dt
        CF = ti.max((fdm - f_ad) * ti.abs(V) - (fdm - f_d) * dx, 0.0)
        flux_b = ti.min(f_ad * ti.abs(V) / dx + CF /dx, f_d) * (v[i,j]) / (ti.abs(v[i,j]) + 1e-16)
        
        F[i, j] += (flux_l - flux_r - flux_t + flux_b)
        F[i, j] = var(0, 1.0, F[i, j])


@ti.kernel        
def post_process_f():
    for i, j in F:
        Fl = F[ti.max(i-1, 0), j]
        Fr = F[ti.min(i+1, imax+1), j]
        Fb = F[i, ti.max(j-1, 0)]
        Ft = F[i, ti.min(j+1, jmax+1)]
        if F[i, j] < eps:
            F[i, j] = 0.0
        elif F[i, j] > 1.0 - eps:
            F[i, j] = 1.0
        # elif Fl < eps or Fr < eps or Fb < eps or Ft < eps:
        #     F[i, j] = F[i, j] - 1.1 * eps
            

# Start Main-loop            
grid_staggered()
set_init_F()

istep = 0
istep_max = 500000
nstep = 1000
check_mass = np.zeros(istep_max // nstep)  # Check mass
os.makedirs('output', exist_ok=True)  # Make dir for output

while istep < istep_max:
    istep += 1
    init_uv()
    set_BC()
    # cal_fgrad()
    # solve_VOF_sola()  # Original Donor-Acceptor
    
    solve_VOF_upwind()  # Upwind scheme
    
    # solve_VOF_rudman()
    
    # solve_VOF_zalesak()
    post_process_f()
    set_BC()
    
    if (istep % nstep) == 0:  # Output data every <nstep> steps
        Fnp = F.to_numpy()
        count = istep // nstep - 1
        check_mass[count] = np.sum(abs(Fnp[imin:-1, jmin:-1]))
        div = cal_vdiv()
        print(f'>>> Number of iterations:{istep:<5d}, sum of VOF:{check_mass[count]:6.2f}; velocity divergence: {div:6.2e}')
        
        if SAVE_FIG:
            xm1 = xm.to_numpy()
            ym1 = ym.to_numpy()
            
            plt.figure(figsize=(5, 5))  # Initialize the output image        
            # plt.contour(xm1[imin:-1], ym1[jmin:-1], Fnp[imin:-1, jmin:-1].T, [0.5], cmap=plt.cm.jet)
            plt.contourf(xm1[imin:-1], ym1[jmin:-1], Fnp[imin:-1, jmin:-1].T, cmap=plt.cm.jet)            
            plt.savefig(f'output/{count:06d}.png')
            plt.close()
            
        if SAVE_DAT:
            np.savetxt(f'output/{count:06d}-F.csv', Fnp, delimiter=',')
            unp = u.to_numpy()
            vnp = v.to_numpy()
            vdivnp = vdiv.to_numpy()
            np.savetxt(f'output/{count:06d}-u.csv', unp, delimiter=',')
            np.savetxt(f'output/{count:06d}-v.csv', vnp, delimiter=',')
            np.savetxt(f'output/{count:06d}-vdiv.csv', vdivnp, delimiter=',')
            
            plt.figure(figsize=(5, 5))  # Plot the u velocity
            plt.contourf(xm1[imin:-1], ym1[jmin:-1], unp[imin:-1, jmin:-1].T, cmap=plt.cm.jet)
            plt.savefig(f'output/{count:06d}-u.png')
            plt.close()
            
            plt.figure(figsize=(5, 5))  # Plot the v velocity
            plt.contourf(xm1[imin:-1], ym1[jmin:-1], vnp[imin:-1, jmin:-1].T, cmap=plt.cm.jet)
            plt.savefig(f'output/{count:06d}-v.png')
            plt.close()
            
            plt.figure(figsize=(5, 5))  # Plot the velocity divergence field
            plt.contourf(xm1[imin:-1], ym1[jmin:-1], vdivnp[imin:-1, jmin:-1].T, cmap=plt.cm.jet)
            plt.savefig(f'output/{count:06d}-div.png')
            plt.close()
            
            
