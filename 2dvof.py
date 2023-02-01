import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import os

ti.init(arch=ti.cpu, default_fp=ti.f64, debug=True)

SAVE_FIG = True
SAVE_DAT = False
SURFACE_PRESSURE_SCHEME = 1  # 0 -> original divergence; 1 -> pressure interpolation

nx = 64  # Number of grid points in the x direction
ny = 64 # Number of grid points in the y direction
res = 64
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
p = ti.field(float, shape=(imax + 2, jmax + 2))
pt = ti.field(float, shape=(imax + 2, jmax + 2))
rho = ti.field(float, shape=(imax + 2, jmax + 2))
mu = ti.field(float, shape=(imax + 2, jmax + 2))
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

u_star = ti.field(float, shape=(imax + 2, jmax + 2))
v_star = ti.field(float, shape=(imax + 2, jmax + 2))

print(f'>>> Surface pressure scheme: {SURFACE_PRESSURE_SCHEME}')
print(f'>>> Grid resolution: {nx} x {ny}, dt = {dt:4.2e}')
print(f'>>> Density ratio: {rho_water / rho_air : 4.2f}')


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
    x2 = 0.5
    y1 = 0.0
    y2 = 0.3
    for i, j in F:  # [0,33], [0,33]
        if (xm[i] >= x1) and (xm[i] <= x2) and (ym[j] >= y1) and (ym[j] <= y2):
            F[i, j] = 1.0
            Fn[i, j] = F[i, j]
    '''
    # Rising bubble
    for i, j in F:
        x = xm[i]
        y = ym[j]
        cx, cy = 0.05, 0.02
        r = 0.01
        if y < 0.09:
            F[i, j] = 1.0
        if ( (x - cx)**2 + (y - cy)**2 < r**2):
            F[i, j] = 0.0


@ti.kernel
def set_BC():
    for i in ti.ndrange(imax + 2):
        # bottom: slip 
        u[i, jmin - 1] = u[i, jmin]
        v[i, jmin] = 0  # v[i, jmin + 1]
        F[i, jmin - 1] = F[i, jmin]
        p[i, jmin - 1] = p[i, jmin]        
        # top: open
        u[i, jmax + 1] = u[i, jmax]
        v[i, jmax + 1] = v[i, jmax]
        F[i, jmax + 1] = F[i, jmax]
        p[i, jmax + 1] = p[i, jmax]        
    for j in ti.ndrange(jmax + 2):
        # left: slip
        u[imin, j] = 0  # u[imin + 1, j]
        v[imin - 1, j] = v[imin, j]
        F[imin - 1, j] = F[imin, j]
        p[imin - 1, j] = p[imin, j]        
        # right: slip
        u[imax + 1, j] = 0  # u[imax, j]
        v[imax + 1, j] = v[imax, j]
        F[imax + 1, j] = F[imax, j]
        p[imax + 1, j] = p[imax, j]        


@ti.func
def var(a, b, c):
    # Find the median of a,b, and c
    center = a + b + c - ti.max(a, b, c) - ti.min(a, b, c)
    return center


@ti.kernel
def cal_mu_rho():  # 11/3 Checked + Modified
    # Calculate density rho and kinematic viscosity Mu
    for I in ti.grouped(rho):
        F = var(0.0, 1.0, F[I])
        rho[I] = rho_air * (1 - F) + rho_water * F
        mu[I] = nu_water * F + nu_air * (1.0 - F)  # mu is kinematic viscosity


@ti.kernel
def advect_upwind():
    # Upwind scheme for advection term suggested by ltt
    for j, i in ti.ndrange((jmin, jmax + 1), (imin + 1, imax + 1)): # i = 2:32; j = 1:32
        v_here = 0.25 * (v[i - 1, j] + v[i - 1, j + 1] + v[i, j] + v[i, j + 1])
        dudx = (u[i,j] - u[i-1,j]) * dxi if u[i,j] > 0 else (u[i+1,j]-u[i,j])*dxi
        dudy = (u[i,j] - u[i,j-1]) * dyi if v_here > 0 else (u[i,j+1]-u[i,j])*dyi        
        u_star[i, j] = (
            u[i, j] + dt *
            (mu[i, j] * (u[i - 1, j] - 2 * u[i, j] + u[i + 1, j]) * dxi**2
             + mu[i, j] * (u[i, j - 1] - 2 * u[i, j] + u[i, j + 1]) * dyi**2
             - u[i, j] * dudx - v_here * dudy
             + gx)
        )
    for j, i in ti.ndrange((jmin + 1, jmax + 1), (imin, imax + 1)): # i = 1:32; j = 2:32
        u_here = 0.25 * (u[i, j - 1] + u[i, j] + u[i + 1, j - 1] + u[i + 1, j])
        dvdx = (v[i,j] - v[i-1,j]) * dxi if u_here > 0 else (v[i+1,j] - v[i,j]) * dxi
        dvdy = (v[i,j] - v[i,j-1]) * dyi if v[i,j] > 0 else (v[i,j+1] - v[i,j]) * dyi
        v_star[i, j] = (
            v[i, j] + dt *
            (mu[i, j] * (v[i - 1, j] - 2 * v[i, j] + v[i + 1, j]) * dxi**2
             + mu[i, j] * (v[i, j - 1] - 2 * v[i, j] + v[i, j + 1]) * dyi**2
             - u_here * dvdx - v[i, j] * dvdy
             + gy)
        )


@ti.kernel
def solve_p_jacobi(n:ti.i32):
    ti.loop_config(serialize=True)
    for s in range(n):
        for i, j in ti.ndrange((imin, imax+1), (jmin, jmax+1)):
            assert rho[i, j] <= rho_water and rho[i, j] >= rho_air
            # The base unit of rhs is (ML^-3T^-2); pressure's dimension is (ML^-1T^-2)
            # Therefore, (ap * rhs)'s dimension is (ML^-1T^-2), which is pressure
            rhs = - rho[i, j] / dt * \
                 ((u_star[i + 1, j] - u_star[i, j]) * dxi +
                  (v_star[i, j + 1] - v_star[i, j]) * dyi)

            # Temporary surface pressure solution; need to be modified
            is_surface = False
            fc = ti.abs(F[i, j])
            fl = ti.abs(F[i-1, j])
            fr = ti.abs(F[i+1, j])
            ft = ti.abs(F[i, j+1])
            fb = ti.abs(F[i, j-1])
            sf_eps = 1e-2  # This eps affects surface shape; need fine-tuning
            if fc < 1.0 - sf_eps and fc > sf_eps \
               and (fl < sf_eps or fr < sf_eps or ft < sf_eps or fb < sf_eps):            
                is_surface = True
            else:
                is_surface = False
            if SURFACE_PRESSURE_SCHEME != 0 and is_surface:
                rhs = 0.0
                
            ae = - 1.0 * dxi ** 2 if i != imax else 0.0
            aw = - 1.0 * dxi ** 2 if i != imin else 0.0
            an = - 1.0 * dyi ** 2 if j != jmax else 0.0
            a_s = - 1.0 * dyi ** 2 if j != jmin else 0.0
            ap = -1.0 * (ae + aw + an + a_s)
            pt[i, j] = (rhs - ae * p[i+1,j] - aw * p[i-1,j] - an * p[i,j+1] - a_s * p[i,j-1]) / ap
            assert ti.abs(pt[i, j]) < 1e6, f'>>> Pressure exploded at p[i,j] = {p[i,j]}'
            
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
        assert u[i, j] * dt < 0.25 * dx, f'U velocity courant number > 1, u[{i},{j}] = {u[i,j]}, p[{i},{j}]={p[i,j]}'
    for j, i in ti.ndrange((jmin + 1, jmax + 1), (imin, imax + 1)):
        r = (rho[i, j] + rho[i, j-1]) * 0.5
        v[i, j] = v_star[i, j] - dt / r * (p[i, j] - p[i, j - 1]) * dyi
        if v[i, j] * dt > 0.25 * dy:
            print(f'V velocity courant number > 1, v[{i},{j}] = {v[i,j]}, p[{i},{j}]={p[i,j]},\
            p[{i},{j-1}]={p[i,j-1]}, delt = {- dt / rho[i, j] * (p[i, j] - p[i - 1, j]) * dxi},\
            v_star = {v_star[i, j]}')
        assert v[i, j] * dt < 0.25 * dy, f'V velocity courant number > 1, v[{i},{j}] = {v[i,j]}, p[{i},{j}]={p[i,j]}'


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


@ti.kernel
def solve_VOF():
    # Method described in original VOF paper
    for I in ti.grouped(Fn):
        Fn[I] = F[I]
        
    ti.loop_config(serialize=True)
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
    cal_mu_rho()
    advect_upwind()
    set_BC()
    
    solve_p_jacobi(n=100)
    update_uv()
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
            plt.savefig(f'output/{istep:05d}.png')
            plt.close()
            
        if SAVE_DAT:
            np.savetxt(f'output/{istep:05d}-F.csv', Fnp, delimiter=',')
            unp = u.to_numpy()
            vnp = v.to_numpy()
            pnp = p.to_numpy()
            vdivnp = vdiv.to_numpy()
            np.savetxt(f'output/{istep:05d}-u.csv', unp, delimiter=',')
            np.savetxt(f'output/{istep:05d}-v.csv', vnp, delimiter=',')
            np.savetxt(f'output/{istep:05d}-p.csv', pnp, delimiter=',')
            
            plt.figure(figsize=(5, 5))  # Plot the pressure field
            plt.contourf(xm1[imin:-1], ym1[jmin:-1], pnp[imin:-1, jmin:-1].T, cmap=plt.cm.jet)
            plt.savefig(f'output/{istep:05d}-p.png')
            plt.close()
            
            plt.figure(figsize=(5, 5))  # Plot the u velocity
            plt.contourf(xm1[imin:-1], ym1[jmin:-1], unp[imin:-1, jmin:-1].T, cmap=plt.cm.jet)
            plt.savefig(f'output/{istep:05d}-u.png')
            plt.close()
            
            plt.figure(figsize=(5, 5))  # Plot the v velocity
            plt.contourf(xm1[imin:-1], ym1[jmin:-1], vnp[imin:-1, jmin:-1].T, cmap=plt.cm.jet)
            plt.savefig(f'output/{istep:05d}-v.png')
            plt.close()
            
            plt.figure(figsize=(5, 5))  # Plot the pressure field
            plt.contourf(xm1[imin:-1], ym1[jmin:-1], vdivnp[imin:-1, jmin:-1].T, cmap=plt.cm.jet)
            plt.savefig(f'output/{istep:05d}-div.png')
            plt.close()
            
            
