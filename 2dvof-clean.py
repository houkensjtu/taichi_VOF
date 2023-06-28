import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

ti.init(arch=ti.gpu, default_fp=ti.f32)

SAVE_FIG = False

nx = 100  # Number of grid points in the x direction
ny = 100  # Number of grid points in the y direction

Lx = 0.1  # The length of the domain
Ly = 0.1  # The width of the domain
rho_water = 1000.0
rho_air = 20.0
nu_water = 1e-3  # kinematic viscosity, nu = mu / rho
nu_air = 1.5e-3
sigma = 0.07

gx = 0 # Gravity
gy = -9.8

dt = 1e-5  # Use smaller dt for higher density ratio
eps = 1e-6  # Threshold used in vfconv and f post processings

# Mesh information
imin = 1
imax = imin + nx - 1
jmin = 1
jmax = jmin + ny - 1
x = ti.field(float, shape=imax + 3)
y = ti.field(float, shape=jmax + 3)
xnp = np.hstack((0.0, np.linspace(0, Lx, nx + 1), Lx)).astype(np.float32)  # [0, 0, ... 1, 1]
x.from_numpy(xnp)
ynp = np.hstack((0.0, np.linspace(0, Ly, ny + 1), Ly)).astype(np.float32)  # [0, 0, ... 1, 1]
y.from_numpy(ynp)
xm = ti.field(float, shape=imax + 2)
ym = ti.field(float, shape=jmax + 2)
dx = x[imin + 2] - x[imin + 1]
dy = y[jmin + 2] - y[jmin + 1]
dxi = 1 / dx
dyi = 1 / dy

# Variables for VOF function
F = ti.field(float, shape=(imax + 2, jmax + 2))
Fn = ti.field(float, shape=(imax + 2, jmax + 2))
Ftd = ti.field(float, shape=(imax + 2, jmax + 2))
ax = ti.field(float, shape=(imax + 2, jmax + 2))
ay = ti.field(float, shape=(imax + 2, jmax + 2))
cx = ti.field(float, shape=(imax + 2, jmax + 2))
cy = ti.field(float, shape=(imax + 2, jmax + 2))
rp = ti.field(float, shape=(imax + 2, jmax + 2))
rm = ti.field(float, shape=(imax + 2, jmax + 2))

# Variables for N-S equation
u = ti.field(float, shape=(imax + 2, jmax + 2))
v = ti.field(float, shape=(imax + 2, jmax + 2))
u_star = ti.field(float, shape=(imax + 2, jmax + 2))
v_star = ti.field(float, shape=(imax + 2, jmax + 2))
p = ti.field(float, shape=(imax + 2, jmax + 2))
pt = ti.field(float, shape=(imax + 2, jmax + 2))
Ap = ti.field(float, shape=(imax + 2, jmax + 2))
rhs = ti.field(float, shape=(imax + 2, jmax + 2))
rho = ti.field(float, shape=(imax + 2, jmax + 2))
nu = ti.field(float, shape=(imax + 2, jmax + 2))

# Variables for interface reconstruction
mx1 = ti.field(float, shape=(imax + 2, jmax + 2))
my1 = ti.field(float, shape=(imax + 2, jmax + 2))
mx2 = ti.field(float, shape=(imax + 2, jmax + 2))
my2 = ti.field(float, shape=(imax + 2, jmax + 2))
mx3 = ti.field(float, shape=(imax + 2, jmax + 2))
my3 = ti.field(float, shape=(imax + 2, jmax + 2))
mx4 = ti.field(float, shape=(imax + 2, jmax + 2))
my4 = ti.field(float, shape=(imax + 2, jmax + 2))
mxsum = ti.field(float, shape=(imax + 2, jmax + 2))
mysum = ti.field(float, shape=(imax + 2, jmax + 2))
mx = ti.field(float, shape=(imax+2, jmax+2))
my = ti.field(float, shape=(imax+2, jmax+2))
karpa = ti.field(float, shape=(imax + 2, jmax + 2))  # interface curvature
magnitude = ti.field(float, shape=(imax+2, jmax+2))


# For visualization
resolution = (400, 400)
rgb_buf = ti.field(dtype=ti.f32, shape=resolution)

print(f'>>> A VOF solver written in Taichi')
print(f'>>> Grid resolution: {nx} x {ny}, dt = {dt:4.2e}')
print(f'>>> Density ratio: {rho_water / rho_air : 4.2f}, gravity : {gy : 4.2f}, sigma : {sigma : 4.2f}')
print(f'>>> Viscosity ratio: {nu_water / nu_air : 4.2f}')
print(f'>>> Please wait a few seconds to let the kernels compile...')


@ti.kernel
def grid_staggered():
    for i in xm:
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

    # Dambreak
    # The initial volume fraction of the domain
    x1 = 0.0
    x2 = Lx / 3
    y1 = 0.0
    y2 = Ly / 2
    for i, j in F:  # [0,33], [0,33]
        if (xm[i] >= x1) and (xm[i] <= x2) and (ym[j] >= y1) and (ym[j] <= y2):
            F[i, j] = 1.0
            Fn[i, j] = F[i, j]
    '''
    # Rising bubble
    for i, j in F:
        x = xm[i]
        y = ym[j]
        r = Lx / 10
        cx, cy = Lx / 2, 2 * r
        
        F[i, j] = find_area(i, j, cx, cy, r)
        Fn[i, j] = find_area(i, j, cx, cy, r)
    '''

            
@ti.kernel
def set_BC():
    for i in ti.ndrange(imax + 2):
        # bottom: slip 
        u[i, jmin - 1] = u[i, jmin]
        v[i, jmin] = 0
        F[i, jmin - 1] = F[i, jmin]
        p[i, jmin - 1] = p[i, jmin]
        rho[i, jmin - 1] = rho[i, jmin]                
        # top: open
        u[i, jmax + 1] = u[i, jmax]
        v[i, jmax + 1] = 0 #v[i, jmax]
        F[i, jmax + 1] = F[i, jmax]
        p[i, jmax + 1] = p[i, jmax]
        rho[i, jmax + 1] = rho[i, jmax]                
    for j in ti.ndrange(jmax + 2):
        # left: slip
        u[imin, j] = 0
        v[imin - 1, j] = v[imin, j]
        F[imin - 1, j] = F[imin, j]
        p[imin - 1, j] = p[imin, j]
        rho[imin - 1, j] = rho[imin, j]                
        # right: slip
        u[imax + 1, j] = 0
        v[imax + 1, j] = v[imax, j]
        F[imax + 1, j] = F[imax, j]
        p[imax + 1, j] = p[imax, j]
        rho[imax + 1, j] = rho[imax, j]                


@ti.func
def var(a, b, c):
    # Find the median of a,b, and c
    center = a + b + c - ti.max(a, b, c) - ti.min(a, b, c)
    return center


@ti.kernel
def cal_nu_rho():
    for I in ti.grouped(rho):
        F = var(0.0, 1.0, F[I])
        rho[I] = rho_air * (1 - F) + rho_water * F
        nu[I] = nu_water * F + nu_air * (1.0 - F)  # mu is kinematic viscosity


@ti.kernel
def advect_upwind():
    # Upwind scheme for advection term suggested by ltt
    for j, i in ti.ndrange((jmin, jmax + 1), (imin + 1, imax + 1)): # i = 2:32; j = 1:32
        v_here = 0.25 * (v[i - 1, j] + v[i - 1, j + 1] + v[i, j] + v[i, j + 1])
        dudx = (u[i,j] - u[i-1,j]) * dxi if u[i,j] > 0 else (u[i+1,j]-u[i,j])*dxi
        dudy = (u[i,j] - u[i,j-1]) * dyi if v_here > 0 else (u[i,j+1]-u[i,j])*dyi
        fx_karpa = - sigma * (F[i, j] - F[i - 1, j]) * (karpa[i, j] + karpa[i - 1, j]) / 2 / dx        
        u_star[i, j] = (
            u[i, j] + dt *
            (nu[i, j] * (u[i - 1, j] - 2 * u[i, j] + u[i + 1, j]) * dxi**2
             + nu[i, j] * (u[i, j - 1] - 2 * u[i, j] + u[i, j + 1]) * dyi**2
             - u[i, j] * dudx - v_here * dudy
             + gx + fx_karpa * 2 / (rho[i, j] + rho[i - 1, j]))
        )
    for j, i in ti.ndrange((jmin + 1, jmax + 1), (imin, imax + 1)): # i = 1:32; j = 2:32
        u_here = 0.25 * (u[i, j - 1] + u[i, j] + u[i + 1, j - 1] + u[i + 1, j])
        dvdx = (v[i,j] - v[i-1,j]) * dxi if u_here > 0 else (v[i+1,j] - v[i,j]) * dxi
        dvdy = (v[i,j] - v[i,j-1]) * dyi if v[i,j] > 0 else (v[i,j+1] - v[i,j]) * dyi
        fy_karpa = - sigma * (F[i, j] - F[i, j - 1]) * (karpa[i, j] + karpa[i, j - 1]) / 2 / dy        
        v_star[i, j] = (
            v[i, j] + dt *
            (nu[i, j] * (v[i - 1, j] - 2 * v[i, j] + v[i + 1, j]) * dxi**2
             + nu[i, j] * (v[i, j - 1] - 2 * v[i, j] + v[i, j + 1]) * dyi**2
             - u_here * dvdx - v[i, j] * dvdy
             + gy +  fy_karpa * 2 / (rho[i, j] + rho[i, j - 1]))
        )


@ti.kernel
def solve_p_jacobi():
    for i, j in ti.ndrange((imin, imax+1), (jmin, jmax+1)):
        assert rho[i, j] <= rho_water and rho[i, j] >= rho_air
        # The base unit of rhs is (ML^-3T^-2); pressure's dimension is (ML^-1T^-2)
        # Therefore, (ap * rhs)'s dimension is (ML^-1T^-2), which is pressure
        rhs = rho[i, j] / dt * \
            ((u_star[i + 1, j] - u_star[i, j]) * dxi +
             (v_star[i, j + 1] - v_star[i, j]) * dyi)

        # Calculate the term due to density gradient
        drhox1 = (rho[i + 1, j - 1] + rho[i + 1, j] + rho[i + 1, j + 1]) / 3
        drhox2 = (rho[i - 1, j - 1] + rho[i - 1, j] + rho[i - 1, j + 1]) / 3                
        drhodx = (dt / drhox1 - dt / drhox2) / (2 * dx)
        drhoy1 = (rho[i - 1, j + 1] + rho[i, j + 1] + rho[i + 1, j + 1]) / 3
        drhoy2 = (rho[i - 1, j - 1] + rho[i, j - 1] + rho[i + 1, j - 1]) / 3                
        drhody = (dt / drhoy1 - dt / drhoy2) / (2 * dy)
        dpdx = (p[i + 1, j] - p[i - 1, j]) / (2 * dx)
        dpdy = (p[i, j + 1] - p[i, j - 1]) / (2 * dy)
        den_corr = (drhodx * dpdx + drhody * dpdy) * rho[i, j] / dt
        if istep < 2:
            pass
        else:
            rhs -= den_corr
            
        ae = dxi ** 2 if i != imax else 0.0
        aw = dxi ** 2 if i != imin else 0.0
        an = dyi ** 2 if j != jmax else 0.0
        a_s = dyi ** 2 if j != jmin else 0.0
        ap = - 1.0 * (ae + aw + an + a_s)
        pt[i, j] = (rhs - ae * p[i+1,j] - aw * p[i-1,j] - an * p[i,j+1] - a_s * p[i,j-1]) / ap
        # assert ti.abs(pt[i, j]) < 1e8, f'>>> Pressure exploded at p[{i},{j}] = {p[i,j]}'
            
    for i, j in ti.ndrange((imin, imax+1), (jmin, jmax+1)):
        p[i, j] = pt[i, j]

            
@ti.kernel
def update_uv():
    for j, i in ti.ndrange((jmin, jmax + 1), (imin + 1, imax + 1)):
        r = (rho[i, j] + rho[i-1, j]) * 0.5
        u[i, j] = u_star[i, j] - dt / r * (p[i, j] - p[i - 1, j]) * dxi
        if u[i, j] * dt > 0.25 * dx:
            print(f'U velocity courant number > 1, u[{i},{j}] = {u[i,j]}, p[{i},{j}]={p[i,j]},\
            p[{i-1},{j}]={p[i-1,j]}, delt = {- dt / rho[i, j] * (p[i, j] - p[i - 1, j]) * dxi},\
            u_star = {u_star[i, j]}')
        #assert u[i, j] * dt < 0.25 * dx, f'U velocity courant number > 1, u[{i},{j}] = {u[i,j]}, p[{i},{j}]={p[i,j]}'
    for j, i in ti.ndrange((jmin + 1, jmax + 1), (imin, imax + 1)):
        r = (rho[i, j] + rho[i, j-1]) * 0.5
        v[i, j] = v_star[i, j] - dt / r * (p[i, j] - p[i, j - 1]) * dyi
        if v[i, j] * dt > 0.25 * dy:
            print(f'V velocity courant number > 1, v[{i},{j}] = {v[i,j]}, p[{i},{j}]={p[i,j]},\
            p[{i},{j-1}]={p[i,j-1]}, delt = {- dt / rho[i, j] * (p[i, j] - p[i - 1, j]) * dxi},\
            v_star = {v_star[i, j]}')
        #assert v[i, j] * dt < 0.25 * dy, f'V velocity courant number > 1, v[{i},{j}] = {v[i,j]}, p[{i},{j}]={p[i,j]}'


@ti.kernel
def cal_kappa():
    for j, i in ti.ndrange((jmin, jmax + 1), (imin, imax + 1)):
        karpa[i, j] = -(1 / dx / 2 * (mx[i + 1, j] - mx[i - 1, j]) + 1 / dy / 2 * (my[i, j + 1] - my[i, j - 1]))


@ti.kernel
def get_normal_young():
    for j, i in ti.ndrange((jmin, jmax + 1), (imin, imax + 1)):
        # Points in between the outermost boundaries
        mx1[i, j] = -1 / (2 * dx) * (F[i + 1, j + 1] + F[i + 1, j] - F[i, j + 1] - F[i, j])  # (i+1/2,j+1/2)
        my1[i, j] = -1 / (2 * dy) * (F[i + 1, j + 1] - F[i + 1, j] + F[i, j + 1] - F[i, j])
        mx2[i, j] = -1 / (2 * dx) * (F[i + 1, j] + F[i + 1, j - 1] - F[i, j] - F[i, j - 1])  # (i+1/2,j-1/2)
        my2[i, j] = -1 / (2 * dy) * (F[i + 1, j] - F[i + 1, j - 1] + F[i, j] - F[i, j - 1])
        mx3[i, j] = -1 / (2 * dx) * (F[i, j] + F[i, j - 1] - F[i - 1, j] - F[i - 1, j - 1])  # (i-1/2,j-1/2)
        my3[i, j] = -1 / (2 * dy) * (F[i, j] - F[i, j - 1] + F[i - 1, j] - F[i - 1, j - 1])
        mx4[i, j] = -1 / (2 * dx) * (F[i, j + 1] + F[i, j] - F[i - 1, j + 1] - F[i - 1, j])  # (i-1/2,j+1/2)
        my4[i, j] = -1 / (2 * dy) * (F[i, j + 1] - F[i, j] + F[i - 1, j + 1] - F[i - 1, j])
        # Summing of mx and my components for normal vector
        mxsum[i, j] = (mx1[i, j] + mx2[i, j] + mx3[i, j] + mx4[i, j]) / 4
        mysum[i, j] = (my1[i, j] + my2[i, j] + my3[i, j] + my4[i, j]) / 4

        # Normalizing the normal vector into unit vectors
        if abs(mxsum[i, j]) < 1e-10 and abs(mysum[i, j])< 1e-10:
            mx[i, j] = mxsum[i, j]
            my[i, j] = mysum[i, j]
        else:
            magnitude[i, j] = ti.sqrt(mxsum[i, j] * mxsum[i, j] + mysum[i, j] * mysum[i, j])
            mx[i, j] = mxsum[i, j] / magnitude[i, j]
            my[i, j] = mysum[i, j] / magnitude[i, j]
        

def solve_VOF_rudman():
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
        # dv = dx * dy
        fl_L = u[i, j] * dt * Fn[i - 1, j] if u[i, j] >= 0 else u[i, j] * dt * Fn[i, j]
        fr_L = u[i + 1, j] * dt * Fn[i, j] if u[i + 1, j] >= 0 else u[i + 1, j] * dt * Fn[i + 1, j]
        ft_L = 0
        fb_L = 0
        Ftd[i, j] = (Fn[i, j] + (fl_L - fr_L + fb_L - ft_L) * dy / (dx * dy)) * dx * dy / dv
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
        # dv = dx * dy
        fl_L = 0
        fr_L = 0
        ft_L = v[i, j + 1] * dt * Fn[i, j] if v[i, j + 1] >= 0 else v[i, j + 1] * dt * Fn[i, j + 1]
        fb_L = v[i, j] * dt * Fn[i, j - 1] if v[i, j] >= 0 else v[i, j] * dt * Fn[i, j]
        Ftd[i, j] = (Fn[i, j] + (fl_L - fr_L + fb_L - ft_L) * dy / (dx * dy)) * dx * dy / dv
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
def post_process_f():
    for i, j in F:
        F[i, j] = var(F[i, j], 0, 1)


@ti.kernel
def get_vof_field():
    r = resolution[0] // nx
    for I in ti.grouped(rgb_buf):
        rgb_buf[I] = F[I // r]
            

# Start Main-loop            
grid_staggered()
set_init_F()
istep = 0
nstep = 10
os.makedirs('output', exist_ok=True)  # Make dir for output
gui = ti.GUI('VOF Solver', resolution)

while gui.running:
    istep += 1
    cal_nu_rho()
    cal_kappa()
    get_normal_young()
    
    # Advection
    advect_upwind()
    set_BC()
    
    # Pressure projection
    for _ in range(10):
        solve_p_jacobi()

    update_uv()
    set_BC()

    solve_VOF_rudman()        
    post_process_f()
    set_BC()
    
    if (istep % nstep) == 0:  # Output data every <nstep> steps
        get_vof_field()
        rgbnp = rgb_buf.to_numpy()
        count = istep // nstep - 1
        print(f'>>> Number of iterations:{istep:<5d}, Time:{istep*dt:5.2e} sec.')
        gui.set_image(cm.Blues(rgbnp * 0.7))
        gui.show()

        if SAVE_FIG:
            xm1 = xm.to_numpy()
            ym1 = ym.to_numpy()
            fx, fy = 5, Ly / Lx * 5
            plt.figure(figsize=(fx, fy))  # Initialize the output image
            plt.contour(xm1[imin:-1], ym1[jmin:-1], Fnp[imin:-1, jmin:-1].T, [0.5], cmap=plt.cm.jet)            
            # plt.contourf(xm1[imin:-1], ym1[jmin:-1], Fnp[imin:-1, jmin:-1].T, cmap=plt.cm.jet)
            plt.savefig(f'output/{count:06d}-f.png')
            plt.close()
