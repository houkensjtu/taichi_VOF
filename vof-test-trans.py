import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import os

ti.init(arch=ti.cpu, default_fp=ti.f64, debug=True)

SAVE_FIG = True
SAVE_DAT = False
SURFACE_PRESSURE_SCHEME = 1  # 0 -> original divergence; 1 -> pressure interpolation

nx = 100  # Number of grid points in the x direction
ny = 100 # Number of grid points in the y direction
res = 100
Lx = 0.1  # The length of the domain
Ly = 0.1  # The width of the domain
rho_water = 1.0
rho_air = 0.5
nu_water = 0.001  # coefficient of kinematic viscosity
nu_air = 0.0005

# Direction and magnitude of volume force
gx = 0
gy = -1

# Solution parameters
dt = 1e-4  # Use smaller dt for higher density ratio
eps = 1e-6  # Threshold used in vfconv and f post processings

imin = 1
imax = imin + nx - 1
jmin = 1
jmax = jmin + ny - 1

F = ti.field(float, shape=(imax + 2, jmax + 2))
Fn = ti.field(float, shape=(imax + 2, jmax + 2))
Fgrad = ti.Vector.field(2, float, shape=(imax + 2, jmax + 2))

u = ti.field(float, shape=(imax + 2, jmax + 2))
v = ti.field(float, shape=(imax + 2, jmax + 2))
u_star = ti.field(float, shape=(imax + 2, jmax + 2))
v_star = ti.field(float, shape=(imax + 2, jmax + 2))
vdiv = ti.field(float, shape=(imax + 2, jmax + 2))

p = ti.field(float, shape=(imax + 2, jmax + 2))
pt = ti.field(float, shape=(imax + 2, jmax + 2))

rho = ti.field(float, shape=(imax + 2, jmax + 2))
mu = ti.field(float, shape=(imax + 2, jmax + 2))

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
    xm is imax + 1 size xm[0], xm[1], xm[2] ... xm[32]
    Here the calculation is from xm[1] to xm[32]
    '''
    for i in xm:  # xm[0] = 0.0, xm[33] = 1.0
        xm[i] = 0.5 * (x[i] + x[i + 1])
    for j in ym:
        ym[j] = 0.5 * (y[j] + y[j + 1])

        
@ti.kernel
def set_init_F():
    # Sets the initial volume fraction
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
    '''
    for I in ti.grouped(F):
        F[I] = 1.0
        Fn[I] = 1.0
    # Rising bubble
    for i, j in F:
        x = xm[i]
        y = ym[j]
        cx, cy = 0.05, 0.02
        r = 0.01
        if ( (x - cx)**2 + (y - cy)**2 < r**2):
            F[i, j] = 0.0
            Fn[i, j] = 0.0


@ti.kernel
def init_uv():
    for I in ti.grouped(u):
        u[I] = 0.001
        v[I] = 0.001
        
            
@ti.kernel
def set_BC():
    for i in ti.ndrange(imax + 2):
        # bottom: slip 
        u[i, jmin - 1] = u[i, jmin]
        v[i, jmin] = v[i, jmin + 1]
        F[i, jmin - 1] = F[i, jmin]
        p[i, jmin - 1] = p[i, jmin]
        rho[i, jmin - 1] = rho[i, jmin]                
        # top: open
        u[i, jmax + 1] = u[i, jmax]
        v[i, jmax + 1] = v[i, jmax]
        F[i, jmax + 1] = F[i, jmax]
        p[i, jmax + 1] = p[i, jmax]
        rho[i, jmax + 1] = rho[i, jmax]                
    for j in ti.ndrange(jmax + 2):
        # left: slip
        u[imin, j] = u[imin + 1, j]
        v[imin - 1, j] = v[imin, j]
        F[imin - 1, j] = F[imin, j]
        p[imin - 1, j] = p[imin, j]
        rho[imin - 1, j] = rho[imin, j]                
        # right: slip
        u[imax + 1, j] = u[imax, j]
        v[imax + 1, j] = v[imax, j]
        F[imax + 1, j] = F[imax, j]
        p[imax + 1, j] = p[imax, j]
        rho[imax + 1, j] = rho[imax, j]                


@ti.kernel
def advect_upwind():
    pass


@ti.kernel
def solve_p_jacobi():
    pass

            
@ti.kernel
def update_uv():
    pass


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
def solve_VOF():
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
        elif Fl < eps or Fr < eps or Fb < eps or Ft < eps:
            F[i, j] = F[i, j] - 1.1 * eps
            

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
    cal_fgrad()
    solve_VOF()
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
            plt.contour(xm1[imin:-1], ym1[jmin:-1], Fnp[imin:-1, jmin:-1].T, [0.5], cmap=plt.cm.jet)
            plt.savefig(f'output/{istep:06d}.png')
            plt.close()
            
        if SAVE_DAT:
            np.savetxt(f'output/{istep:06d}-F.csv', Fnp, delimiter=',')
            unp = u.to_numpy()
            vnp = v.to_numpy()
            pnp = p.to_numpy()
            vdivnp = vdiv.to_numpy()
            np.savetxt(f'output/{istep:06d}-u.csv', unp, delimiter=',')
            np.savetxt(f'output/{istep:06d}-v.csv', vnp, delimiter=',')
            np.savetxt(f'output/{istep:06d}-p.csv', pnp, delimiter=',')
            
            plt.figure(figsize=(5, 5))  # Plot the pressure field
            plt.contourf(xm1[imin:-1], ym1[jmin:-1], pnp[imin:-1, jmin:-1].T, cmap=plt.cm.jet)
            plt.savefig(f'output/{istep:06d}-p.png')
            plt.close()
            
            plt.figure(figsize=(5, 5))  # Plot the u velocity
            plt.contourf(xm1[imin:-1], ym1[jmin:-1], unp[imin:-1, jmin:-1].T, cmap=plt.cm.jet)
            plt.savefig(f'output/{istep:06d}-u.png')
            plt.close()
            
            plt.figure(figsize=(5, 5))  # Plot the v velocity
            plt.contourf(xm1[imin:-1], ym1[jmin:-1], vnp[imin:-1, jmin:-1].T, cmap=plt.cm.jet)
            plt.savefig(f'output/{istep:06d}-v.png')
            plt.close()
            
            plt.figure(figsize=(5, 5))  # Plot the pressure field
            plt.contourf(xm1[imin:-1], ym1[jmin:-1], vdivnp[imin:-1, jmin:-1].T, cmap=plt.cm.jet)
            plt.savefig(f'output/{istep:06d}-div.png')
            plt.close()
            
            
