import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import os
'''
Donor-acceptor scheme now basically works
but simulation is still unstable once density modified
also unstable when resolution increased to > 64 x 64
Seems reason is in update_puv; misterious divergence happened.
'''
ti.init(arch=ti.cpu, default_fp=ti.f64, debug=True)

SAVE_FIG = True
SAVE_DAT = True

nx = 32  # Number of grid points in the x direction
ny = 32  # Number of grid points in the y direction
res = 32
Lx = 1.0  # The length of the domain
Ly = 1.0  # The width of the domain

# Physical parameters
rho_water = 1.0
rho_air = 0.5
nu_water = 0.001  # coefficient of kinematic viscosity
nu_air = 0.0005

# Direction and magnitude of volume force
gx = 0
gy = -1

# The initial volume fraction of the domain
x1 = 0.0
x2 = 0.5
y1 = 0.0
y2 = 0.3

# Solution parameters
dt = 1e-3

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
N = nx * ny
L = ti.linalg.SparseMatrixBuilder(N, N, max_num_triplets=N * 6)
Ati = ti.field(float, shape=(N, N))
# The variables in the Poisson solver
u_star = ti.field(float, shape=(imax + 2, jmax + 2))
v_star = ti.field(float, shape=(imax + 2, jmax + 2))
R = ti.field(float, shape=((imax - imin + 1) * (imax - imin + 1)))
pv = ti.field(float, shape=((imax - imin + 1) * (imax - imin + 1)))


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
def set_init_F():  # 11/3 Checked
    # Sets the initial volume fraction
    for i, j in F:  # [0,33], [0,33]
        if (xm[i] >= x1) and (xm[i] <= x2) and (ym[j] >= y1) and (ym[j] <= y2):
            F[i, j] = 1.0


@ti.kernel
def Laplace_operator(A: ti.types.sparse_matrix_builder(), Ati:ti.template()):
    # TODO: Check Is the coefficients of matrix A wrong?
    for i, j in ti.ndrange(res, res):
        row = i * res + j
        if row != 0:
            center = 0.0
            if j != 0:
                A[row, row - 1] += -1.0 * dxi**2
                Ati[row, row - 1] += -1.0 * dxi**2                
                center += 1.0
            if j != res - 1:
                A[row, row + 1] += -1.0 * dxi**2
                Ati[row, row + 1] += -1.0 * dxi**2                
                center += 1.0
            if i != 0:
                A[row, row - res] += -1.0 * dxi**2
                Ati[row, row - res] += -1.0 * dxi**2                
                center += 1.0
            if i != res - 1:
                A[row, row + res] += -1.0 * dxi**2
                Ati[row, row + res] += -1.0 * dxi**2                
                center += 1.0
            A[row, row] += center * dxi**2
            Ati[row, row] += center * dxi**2

    for i in ti.ndrange(res * res):
        if i == 0:
            A[0, i] += 1
            Ati[0, i] += 1            
        else:
            A[0, i] += 0
            Ati[0, i] += 0            


@ti.kernel
def set_BC():
    # Set the wall as impenetrable and slip
    for i in ti.ndrange(imax + 2):
        # bottom
        u[i, jmin - 1] = u[i, jmin]
        # v[i, jmin - 1] = v[i, jmin] # v[i, 0] = v[i, 1] both not touched in advect ?
        F[i, jmin - 1] = F[i, jmin]
        # top
        u[i, jmax + 1] = u[i, jmax]
        v[i, jmax + 1] = v[i, jmax]
        F[i, jmax + 1] = F[i, jmax]
    for j in ti.ndrange(jmax + 2):
        # left
        # u[imin - 1, j] = u[imin, j]  # u[0, j] = u[1, j] both not touched in advect ?
        v[imin - 1, j] = v[imin, j]
        F[imin - 1, j] = F[imin, j]
        # right
        u[imax + 1, j] = u[imax, j]
        v[imax + 1, j] = v[imax, j]
        F[imax + 1, j] = F[imax, j]


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
        # mu[I] = (nu_water * rho_water + nu_air * rho_air) / rho[I] 
        mu[I] = nu_water * F + nu_air * (1.0 - F)


@ti.kernel
def advect():
    # TODO: Is this scheme better, or the original MAC scheme better?
    # Solving Pressure Poisson Equation Using Projection Method
    for j, i in ti.ndrange((jmin, jmax + 1), (imin + 1, imax + 1)): # i = 2:32; j = 1:32
        v_here = 0.25 * (v[i - 1, j] + v[i - 1, j + 1] + v[i, j] + v[i, j + 1])
        u_star[i, j] = (
            u[i, j] + dt *
            (mu[i, j] * (u[i - 1, j] - 2 * u[i, j] + u[i + 1, j]) * dxi**2
             + mu[i, j] * (u[i, j - 1] - 2 * u[i, j] + u[i, j + 1]) * dyi**2
             - u[i, j] * (u[i + 1, j] - u[i - 1, j]) * 0.5 * dxi
             - v_here * (u[i, j + 1] - u[i, j - 1]) * 0.5 * dyi
             + gx)
        )
    for j, i in ti.ndrange((jmin + 1, jmax + 1), (imin, imax + 1)): # i = 1:32; j = 2:32
        u_here = 0.25 * (u[i, j - 1] + u[i, j] + u[i + 1, j - 1] + u[i + 1, j])
        v_star[i, j] = (
            v[i, j] + dt *
            (mu[i, j] * (v[i - 1, j] - 2 * v[i, j] + v[i + 1, j]) * dxi**2
             + mu[i, j] * (v[i, j - 1] - 2 * v[i, j] + v[i, j + 1]) * dyi**2
             - u_here * (v[i + 1, j] - v[i - 1, j]) * 0.5 * dxi
             - v[i, j] * (v[i, j + 1] - v[i, j - 1]) * 0.5 * dyi
             + gy)
        )


@ti.kernel
def advect_upwind():
    # Upwind scheme for advection term suggested by ltt
    # Seems almost divergent at same pace with original advect
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
def cal_div():
    for j, i in ti.ndrange((jmin, jmax + 1), (imin, imax + 1)):
        linear_id = (i - imin) + (j - jmin) * (imax + 1 - imin)
        R[linear_id] = (-rho[i, j] / dt *
                                ((u_star[i + 1, j] - u_star[i, j]) * dxi +
                                 (v_star[i, j + 1] - v_star[i, j]) * dyi))


# @ti.kernel
def cal_fgrad():  # 11/3 Checked, no out-of-range
    '''
    Calculate the Fgrad in internal area.
    '''
    #for i, j in ti.ndrange((imin, imax+1), (jmin, jmax+1)):
    #    dfdx = 0.5 * (F[i + 1, j] - F[i - 1, j]) / dx
    #    dfdy = 0.5 * (F[i, j + 1] - F[i, j - 1]) / dy
    #    Fgrad[i, j] = ti.Vector([dfdx, dfdy])
    Fnp = F.to_numpy()
    Fgrad_np = np.gradient(Fnp)
    # Fgrad.from_numpy(Fgrad_np)
    # Now Fgrad is not calculated (all 0); lead to a f_ad = f_d scheme for most cells.


@ti.kernel
def solve_p_jacobi(n:ti.i32):
    for s in range(n):
        ti.loop_config(serialize=True)
        for i, j in ti.ndrange((imin, imax+1), (jmin, jmax+1)):
            assert rho[i, j] <= rho_water and rho[i, j] >= rho_air
            R = (-rho[i, j] / dt *
                                ((u_star[i + 1, j] - u_star[i, j]) * dxi +
                                 (v_star[i, j + 1] - v_star[i, j]) * dyi))
            ae = - 1.0 * dxi ** 2 if i != imax else 0.0
            aw = - 1.0 * dxi ** 2 if i != imin else 0.0
            an = - 1.0 * dyi ** 2 if j != jmax else 0.0
            a_s = - 1.0 * dyi ** 2 if j != jmin else 0.0
            ap = -1.0 * (ae + aw + an + a_s)
            pt[i, j] = (R - ae * p[i+1,j] - aw * p[i-1,j] - an * p[i,j+1] - a_s * p[i,j-1]) / ap
            assert ti.abs(pt[i, j]) < 1e6, f'>>> Pressure exploded at p[i,j] = {p[i,j]}'
        for i, j in ti.ndrange((imin, imax+1), (jmin, jmax+1)):
            p[i, j] = pt[i, j]

    
@ti.kernel
def update_puv():
    # TODO: Check the pressure correction equation.
    # Pressure at the surface should be specified.
    # TODO: Why u[i,j] * dt explodes when > 64x64 grid ?
    # Seems because pressure calculation explodes
    for j, i in ti.ndrange((jmin, jmax + 1), (imin, imax + 1)):
        linear_id = (i - imin) + (j - 1) * (imax + 1 - imin)
        p[i, j] = pv[linear_id]
    for j, i in ti.ndrange((jmin, jmax + 1), (imin + 1, imax + 1)):
        u[i, j] = u_star[i, j] - dt / rho[i, j] * (p[i, j] - p[i - 1, j]) * dxi
        assert u[i, j] * dt < 0.25 * dx, f'U velocity courant number > 1, u[{i},{j}] = {u[i,j]}'
    for j, i in ti.ndrange((jmin + 1, jmax + 1), (imin, imax + 1)):
        v[i, j] = v_star[i, j] - dt / rho[i, j] * (p[i, j] - p[i, j - 1]) * dyi
        assert v[i, j] * dt < 0.25 * dy, f'V velocity courant number > 1, v[{i},{j}] = {v[i,j]}'


@ti.kernel
def update_uv():
    for j, i in ti.ndrange((jmin, jmax + 1), (imin + 1, imax + 1)):
        u[i, j] = u_star[i, j] - dt / rho[i, j] * (p[i, j] - p[i - 1, j]) * dxi
        assert u[i, j] * dt < 0.25 * dx, f'U velocity courant number > 1, u[{i},{j}] = {u[i,j]}, p[{i},{j}]={p[i,j]}'
    for j, i in ti.ndrange((jmin + 1, jmax + 1), (imin, imax + 1)):
        v[i, j] = v_star[i, j] - dt / rho[i, j] * (p[i, j] - p[i, j - 1]) * dyi
        assert v[i, j] * dt < 0.25 * dy, f'V velocity courant number > 1, v[{i},{j}] = {v[i,j]}, p[{i},{j}]={p[i,j]}'


@ti.kernel
def cal_vdiv()->float:
    d = 0.0
    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        d += rho[i, j] * (u[i+1,j] - u[i,j] + v[i,j+1] - v[i,j])
    return d


@ti.kernel
def solve_F():  # Method used by ZCL
    for j, i in ti.ndrange((jmin, jmax + 1), (imin, imax + 1)):
        u_loc = 3 / 8 * (u[i, j] + u[i + 1, j]) + 1 / 16 * \
                (u[i, j + 1] + u[i + 1, j + 1] + u[i, j - 1] + u[i + 1, j - 1])
        v_loc = 3 / 8 * (v[i, j] + v[i, j + 1]) + 1 / 16 * \
                (v[i - 1, j] + v[i - 1, j + 1] + v[i + 1, j] + v[i + 1, j + 1])
        F[i, j] = (F[i, j] - dt *
                   (u_loc * (F[i + 1, j] - F[i - 1, j]) * dxi / 2 + v_loc *
                    (F[i, j + 1] - F[i, j - 1]) * dyi / 2))
        F[i, j] = var(0, 1, F[i, j])


@ti.kernel
def solve_VOF():
    # TODO: Simplify the code
    # Method described in original VOF paper
    eps = 1e-6
    ti.loop_config(serialize=True)
    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        f_a, f_d, f_ad, f_up = 0.0, 0.0, 0.0, 0.0
        # Flux left
        if u[i, j] > 0:
            f_d, f_a, f_up = F[i-1, j], F[i, j], F[ max(0,i-2) , j]
        else:
            f_a, f_d, f_up = F[i-1, j], F[i, j], F[i+1, j]
        if abs(Fgrad[i, j][0]) > abs(Fgrad[i,j][1]):  # Surface orientation is vertical
            f_ad = f_a
        elif f_a < eps or f_up < eps:
            f_ad = f_a
        else:  # Surface orientation is horizontal
            f_ad = f_d
        V = u[i, j] * dt
        CF = max((1.0 - f_ad) * abs(V) - (1.0 - f_d) * dx, 0.0)
        flux_l = min(f_ad * abs(V) / dx + CF / dx, f_d) * (u[i,j]) / (abs(u[i,j]) + 1e-12)
        
        # Flux right
        if u[i+1, j] > 0:
            f_d, f_a, f_up = F[i, j], F[i+1, j], F[i-1, j]
        else:
            f_a, f_d, f_up = F[i, j], F[i+1, j], F[ min(i+2, imax+1) , j]
        if abs(Fgrad[i, j][0]) >  abs(Fgrad[i,j][1]):  # Surface orientation is vertical
            f_ad = f_a
        elif f_a < eps or f_up < eps:
            f_ad = f_a
        else:  # Surface orientation is horizontal
            f_ad = f_d
        V = u[i+1, j] * dt
        CF = max((1.0 - f_ad) * abs(V) - (1.0 - f_d) * dx, 0.0)
        flux_r = min(f_ad * abs(V) / dx + CF / dx, f_d) * (u[i+1, j]) / (abs(u[i+1, j]) + 1e-12)
        if i == imax:
            flux_r = 0.0
        
        # Flux top
        if v[i, j + 1] > 0:
            f_d, f_a, f_up = F[i, j], F[i, j + 1], F[i, j-1]
        else:
            f_a, f_d, f_up = F[i, j], F[i, j + 1], F[i, min(j+2, jmax+1) ]
        if abs(Fgrad[i, j][0]) > abs(Fgrad[i,j][1]):  # Surface orientation is vertical
            f_ad = f_a
        elif f_a < eps or f_up < eps:
            f_ad = f_a
        else:  # Surface orientation is horizontal
            f_ad = f_d
        V = v[i, j + 1] * dt
        CF = max((1.0 - f_ad) * abs(V) - (1.0 - f_d) * dx, 0.0)
        flux_t = min(f_ad * abs(V) / dx + CF / dx, f_d) * (v[i,j+1]) / (abs(v[i, j+1]) + 1e-12)
        
        # Flux bottom
        if v[i, j] > 0:
            f_d, f_a, f_up = F[i, j-1], F[i, j], F[i, max(0, j-2)]
        else:
            f_a, f_d, f_up = F[i, j-1], F[i, j], F[i, j+1]
        if abs(Fgrad[i, j][0]) > abs(Fgrad[i,j][1]):  # Surface orientation is vertical
            f_ad = f_a
        elif f_a < eps or f_up < eps:
            f_ad = f_a
        else:  # Surface orientation is horizontal
            f_ad = f_d
        V = v[i, j] * dt
        CF = max((1.0 - f_ad) * abs(V) - (1.0 - f_d) * dx, 0.0)
        flux_b = min(f_ad * abs(V) / dx + CF /dx, f_d) * (v[i,j]) / (abs(v[i,j]) + 1e-12)
        
        F[i, j] += (flux_l - flux_r - flux_t + flux_b)
        F[i, j] = var(0, 1, F[i, j])
        

# Calculate the coordinates of the staggered point
grid_staggered()
# Set initial volume fraction
set_init_F()
# Create Laplace operator
Laplace_operator(L, Ati)
A = L.build()
Anp = Ati.to_numpy()
np.savetxt(f'output/A.csv', Anp, delimiter=',')

istep = 0
istep_max = 50000
nstep = 100
check_mass = np.zeros(istep_max // nstep)  # Check mass
os.makedirs('output', exist_ok=True)  # Make dir for output

while istep < istep_max:
    istep += 1  # time step +1    
    # Update rho, mu by F
    cal_mu_rho()
    # Solving Pressure Poisson Equation Using Projection Method
    # advect()
    advect_upwind()
    
    '''
    cal_div()
    solver = ti.linalg.SparseSolver(solver_type="LU")
    solver.analyze_pattern(A)
    solver.factorize(A)
    pv_np = solver.solve(R)
    pv.from_numpy(pv_np)
    isSuccess = solver.info()
    update_puv()    
    '''
    solve_p_jacobi(100)
    update_uv()

    # solve_F()
    cal_fgrad()  # Calculate surface orientation based on current F
    solve_VOF()
    
    set_BC()  # set boundary conditions
    if (istep % nstep) == 0:  # Output data every 100 steps
        Fnp = F.to_numpy()
        count = istep // nstep - 1
        check_mass[count] = np.sum(abs(Fnp[imin:-1, jmin:-1]))
        div = cal_vdiv()
        print(f'>>> Number of iterations:{istep:<5d}, sum of VOF:{check_mass[count]:6.2f}; Current velocity divergence: {div:6.2e}')
        
        if SAVE_FIG:
            xm1 = xm.to_numpy()
            ym1 = ym.to_numpy()
            plt.figure(figsize=(5, 5))  # Initialize the output image        
            plt.contour(xm1[imin:-1], ym1[jmin:-1], Fnp[imin:-1, jmin:-1].T, [0.5], cmap=plt.cm.jet)
            # plt.contourf(xm1[imin:-1], ym1[jmin:-1], Fnp[imin:-1, jmin:-1].T, cmap=plt.cm.jet)  # Plot filled-contour
            plt.savefig(f'output/{istep:05d}.png')
            plt.close()
        if SAVE_DAT:
            np.savetxt(f'output/{istep:05d}-F.csv', Fnp, delimiter=',')
