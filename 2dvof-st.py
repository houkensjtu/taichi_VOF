import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

# ti.init(debug=True, arch=ti.x64, default_fp=ti.f64)
ti.init(arch=ti.x64, default_fp=ti.f64)

nx = 32  # Number of grid points in the x direction
ny = 32  # Number of grid points in the y direction
res = 32
Lx = 1.0  # The length of the domain
Ly = 1.0  # The width of the domain
sigma = 0.005  # surface tension coefficient

# Physical parameters
rho_water = 1.0
rho_air = 0.5
nu_water = 0.001  # coefficient of kinematic viscosity
nu_air = 0.0005

# Direction and magnitude of volume force
gx = 0
gy = -1.0

# The initial volume fraction of the domain
x1 = 0.3
x2 = 0.7
y1 = 0.0
y2 = 0.2

# Solution parameters
dt = 0.001

imin = 1
imax = imin + nx - 1
jmin = 1
jmax = jmin + ny - 1

F = ti.field(float, shape=(imax + 2, jmax + 2))

# Variables of interface reconstruction
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
# calculate alpha
xval = ti.field(float, shape=(imax+2, jmax+2))
yval = ti.field(float, shape=(imax+2, jmax+2))
mxval = ti.field(float, shape=(imax+2, jmax+2))
myval = ti.field(float, shape=(imax+2, jmax+2))
alpha_actual = ti.field(float, shape=(imax+2, jmax+2))
area_actual = ti.field(float, shape=(imax+2, jmax+2))
xright = ti.field(float, shape=(imax+2, jmax+2))
xleft = ti.field(float, shape=(imax+2, jmax+2))
yright = ti.field(float, shape=(imax+2, jmax+2))
yleft = ti.field(float, shape=(imax+2, jmax+2))
slope = ti.field(float, shape=(imax+2, jmax+2))
lowlim = ti.field(float, shape=(imax+2, jmax+2))
highlim = ti.field(float, shape=(imax+2, jmax+2))
alpha_calc = ti.field(float, shape=(imax+2, jmax+2))
n_ite = 1000  # Number of iteration
Area = ti.field(float, shape=(imax+2, jmax+2))
error = ti.field(float, shape=(imax+2, jmax+2))
Cr = ti.field(float, shape=(imax+2, jmax+2))
# --------------------------------------------------------------
u = ti.field(float, shape=(imax + 2, jmax + 2))
v = ti.field(float, shape=(imax + 2, jmax + 2))
p = ti.field(float, shape=(imax + 1, jmax + 1))
Fn = ti.field(float, shape=(imax + 2, jmax + 2))
x = ti.field(float, shape=imax + 2)
y = ti.field(float, shape=imax + 2)
xm, ym = ti.field(float, shape=imax + 1), ti.field(float, shape=imax + 1)

x.from_numpy(np.hstack((0, np.linspace(0, Lx, nx + 1))))
y.from_numpy(np.hstack((0, np.linspace(0, Ly, ny + 1))))
N = nx * ny
L = ti.linalg.SparseMatrixBuilder(N, N, max_num_triplets=N * 6)
dx = x[imin + 1] - x[imin]
dy = y[jmin + 1] - y[jmin]
dxi = 1 / dx
dyi = 1 / dy

# The variables in the Poisson solver
u_star = ti.field(float, shape=(imax + 2, jmax + 2))
v_star = ti.field(float, shape=(imax + 2, jmax + 2))
R = ti.field(float, shape=((imax - imin + 1) * (imax - imin + 1)))
pv = ti.field(float, shape=((imax - imin + 1) * (imax - imin + 1)))


@ti.kernel
def grid_staggered():
    for i in ti.ndrange(imin, imax + 1):
        xm[i] = 0.5 * (x[i] + x[i + 1])
    for j in ti.ndrange(jmin, jmax + 1):
        ym[j] = 0.5 * (y[j] + y[j + 1])


@ti.kernel
def set_init_F():
    # Sets the initial volume fraction
    for j, i in ti.ndrange(jmax + 1, imax + 1):
        if (xm[i] >= x1) and (xm[i] <= x2) and (ym[j] >= y1) and (ym[j] <= y2):
            F[i, j] = 1


@ti.kernel
def Laplace_operator(A: ti.types.sparse_matrix_builder()):
    for i, j in ti.ndrange(res, res):
        row = i * res + j
        if row != 0:
            center = 0.0
            if j != 0:
                A[row, row - 1] += -1.0 * dxi ** 2
                center += 1.0
            if j != res - 1:
                A[row, row + 1] += -1.0 * dxi ** 2
                center += 1.0
            if i != 0:
                A[row, row - res] += -1.0 * dxi ** 2
                center += 1.0
            if i != res - 1:
                A[row, row + res] += -1.0 * dxi ** 2
                center += 1.0
            A[row, row] += center * dxi ** 2
    for i in ti.ndrange(res * res):
        if i == 0:
            A[0, i] += 1
        else:
            A[0, i] += 0


@ti.kernel
def set_BC():
    # Set the wall as impenetrable and slip
    for i in ti.ndrange(imax + 2):
        # bottom
        u[i, jmin - 1] = u[i, jmin]
        v[i, jmin - 1] = v[i, jmin]
        F[i, jmin - 1] = F[i, jmin]
        # top
        u[i, jmax + 1] = u[i, jmax]
        v[i, jmax + 1] = v[i, jmax]
        F[i, jmax + 1] = F[i, jmax]
    for j in ti.ndrange(jmax + 2):
        # left
        u[imin - 1, j] = u[imin, j]
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
def cal_mu_rho():
    # Calculate density rho and kinematic viscosity Mu
    for j, i in ti.ndrange((jmin - 1, jmax + 2), (imin - 1, imax + 2)):
        rho[i, j] = rho_air * (1 - var(0, 1, F[i, j])) + rho_water * var(0, 1, F[i, j])
        mu[i, j] = (nu_water * rho_water * var(0, 1, F[i, j]) + nu_air * rho_air * (1 - var(0, 1, F[i, j]))) / \
                   rho[i, j]


@ti.kernel
def cal_karpa():
    """
    Calculate interface curvature
    """
    for j, i in ti.ndrange((jmin, jmax + 1), (imin, imax + 1)):
        karpa[i, j] = -(1 / dx / 2 * (mx[i + 1, j] - mx[i - 1, j]) + 1 / dy / 2 * (my[i, j + 1] - my[i, j - 1]))


@ti.kernel
def M_Possion():
    # Solving Pressure Poisson Equation Using Projection Method
    for j, i in ti.ndrange((jmin, jmax + 1), (imin + 1, imax + 1)):
        v_here = 0.25 * (v[i - 1, j] + v[i - 1, j + 1] + v[i, j] + v[i, j + 1])
        fx_karpa = - sigma * (F[i, j] - F[i - 1, j]) * (karpa[i, j] + karpa[i - 1, j]) / 2 / dx
        u_star[i, j] = (u[i, j] + dt *
                        (mu[i, j] * (u[i - 1, j] - 2 * u[i, j] + u[i + 1, j]) * dxi ** 2
                         + mu[i, j] * (u[i, j - 1] - 2 * u[i, j] + u[i, j + 1]) * dyi ** 2
                         - u[i, j] * (u[i + 1, j] - u[i - 1, j]) * 0.5 * dxi
                         - v_here * (u[i, j + 1] - u[i, j - 1]) * 0.5 * dyi
                         + gx + fx_karpa / rho[i, j]))
    for j, i in ti.ndrange((jmin + 1, jmax + 1), (imin, imax + 1)):
        u_here = 0.25 * (u[i, j - 1] + u[i, j] + u[i + 1, j - 1] + u[i + 1, j])
        fy_karpa = - sigma * (F[i, j] - F[i, j - 1]) * (karpa[i, j] + karpa[i, j - 1]) / 2 / dy
        v_star[i, j] = (v[i, j] + dt *
                        (mu[i, j] * (v[i - 1, j] - 2 * v[i, j] + v[i + 1, j]) * dxi ** 2
                         + mu[i, j] * (v[i, j - 1] - 2 * v[i, j] + v[i, j + 1]) * dyi ** 2
                         - u_here * (v[i + 1, j] - v[i - 1, j]) * 0.5 * dxi
                         - v[i, j] * (v[i, j + 1] - v[i, j - 1]) * 0.5 * dyi
                         + gy + fy_karpa / rho[i, j]))
    for j, i in ti.ndrange((jmin, jmax + 1), (imin, imax + 1)):
        R[i - imin + (j - 1) * (imax + 1 - imin)] = (-rho[i, j] / dt * (
                (u_star[i + 1, j] - u_star[i, j]) * dxi + (v_star[i, j + 1] - v_star[i, j]) * dyi))


@ti.kernel
def update():
    for j, i in ti.ndrange((jmin, jmax + 1), (imin, imax + 1)):
        p[i, j] = pv[i - imin + (j - 1) * (imax + 1 - imin)]
    for j, i in ti.ndrange((jmin, jmax + 1), (imin + 1, imax + 1)):
        u[i, j] = u_star[i, j] - dt / rho[i, j] * (p[i, j] - p[i - 1, j]) * dxi
    for j, i in ti.ndrange((jmin + 1, jmax + 1), (imin, imax + 1)):
        v[i, j] = v_star[i, j] - dt / rho[i, j] * (p[i, j] - p[i, j - 1]) * dyi


@ti.kernel
def solve_F():
    for j, i in ti.ndrange((jmin, jmax + 1), (imin, imax + 1)):
        u_loc = 3 / 8 * (u[i, j] + u[i + 1, j]) + 1 / 16 * \
                (u[i, j + 1] + u[i + 1, j + 1] + u[i, j - 1] + u[i + 1, j - 1])
        v_loc = 3 / 8 * (v[i, j] + v[i, j + 1]) + 1 / 16 * \
                (v[i - 1, j] + v[i - 1, j + 1] + v[i + 1, j] + v[i + 1, j + 1])
        F[i, j] = (F[i, j] - dt *
                   (u_loc * (F[i + 1, j] - F[i - 1, j]) * dxi / 2 +
                    v_loc * (F[i, j + 1] - F[i, j - 1]) * dyi / 2))
        F[i, j] = var(0, 1, F[i, j])


@ti.kernel
def get_normal_young():
    """
    This performs Youngs Finite Difference, shown in pg. 99 of Tryggvason et.al, Direct Numerical Simulations of Gas-Liquid Multiphase Flows
    It outputs the mx and my values for a given color function F
    """
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


@ti.kernel
def get_alpha():
    """
    This function determines the alpha value based on a given color function on section 5.2.3 (pg 104-105)
    of Tryggvason et.Direct Numerical Simulations of Gas-Liquid Multiphase Flows

    Based on the geometry, the possible alpha values have limits,
    where alpha values outside the limits would provide intersecting lines outside of the geometry of the cell.
    Because these limits are crutial to the areafinder function,
    we were unable to develop a proper iterative function based on slope-of-error analysis that would work for all points.
    calculate the area using many alpha values and choose the alpha with the lowest area.
    An example of an alpha limit is: For a m(+,-) (#4), alpha/mx cannot be both negative if alpha/my is greater than h,
    since the line does not intersect the cell.
    """
    for j, i in ti.ndrange((jmin, jmax + 1), (imin, imax + 1)):
        # Listing necessary parameters for area finding method
        xval[i, j] = x[i]
        yval[i, j] = y[j]
        mxval[i, j] = mx[i, j]
        myval[i, j] = my[i, j]
        # testing specific values
        if F[i, j] > 1.2:
            adfs = 1
        # For Color function values of 1 or 0, we do not need to get alpha
        if mxval[i, j] == 0 and myval[i, j] == 0 and F[i, j] == 0:
            #  Check if mx and my are both 0 for F of 0 (not filled),
            # which yields area of 0
            alpha_actual[i, j] = 0
            area_actual[i, j] = 0
            xright[i, j] = xval[i, j]
            xleft[i, j] = xval[i, j]
            yright[i, j] = yval[i, j]
            yleft[i, j] = yval[i, j]
            Cr[i, j] = area_actual[i, j] / dx / dy
        elif mxval[i, j] == 0 and myval[i, j] == 0 and F[i, j] == 1:
            # Check if mx and my are both 0 for C of 1 (filled),
            # which yields area of 1
            alpha_actual[i, j] = 0  # changed from 1 to 0
            area_actual[i, j] = 1 * dx * dy
            xright[i, j] = xval[i, j]
            xleft[i, j] = xval[i, j]
            yright[i, j] = yval[i, j]
            yleft[i, j] = yval[i, j]
            Cr[i, j] = area_actual[i, j] / dx / dy
        elif F[i, j] == 0:
            alpha_actual[i, j] = 0
            area_actual[i, j] = 0
            xright[i, j] = xval[i, j]
            xleft[i, j] = xval[i, j]
            yright[i, j] = yval[i, j]
            yleft[i, j] = yval[i, j]
            Cr[i, j] = area_actual[i, j] / dx / dy
        elif F[i, j] >= 0.99990:  # We were seeing values with machine error
            alpha_actual[i, j] = 0
            area_actual[i, j] = F[i, j] * dx * dy
            xright[i, j] = xval[i, j]
            xleft[i, j] = xval[i, j]
            yright[i, j] = yval[i, j]
            yleft[i, j] = yval[i, j]
            Cr[i, j] = area_actual[i, j] / dx / dy
        else:
            # Perform for loop with many alpha values and choosing the value
            # for alpha that produces the least error
            slope[i, j] = -1 / (myval[i, j] / mxval[i, j])
            # Producing alpha vector that is within constraints of geometric cell
            if mxval[i, j] >= 0 and myval[i, j] >= 0:
                # mx and my are positive (+,+) (1)
                lowlim[i, j] = 0
                highlim[i, j] = (-dx / slope[i, j] + dx) * mxval[i, j]
            elif mxval[i, j] <= 0 and myval[i, j] <= 0:
                # mx and my are negative (-,-) (3)
                lowlim[i, j] = (-dx / slope[i, j] + dx) * mxval[i, j]
                highlim[i, j] = 0
            elif mxval[i, j] <= 0 and myval[i, j] >= 0:
                # mx is negative and my is positive (-,+) (2)
                # hmx < alpha < hmy
                lowlim[i, j] = dx * mxval[i, j]
                highlim[i, j] = dy * myval[i, j]
            elif mxval[i, j] >= 0 and myval[i, j] <= 0:
                # mx is positive and my is negative (+,-) 4
                # hmy < alpha < hmx
                lowlim[i, j] = dy * myval[i, j]
                highlim[i, j] = dx * mxval[i, j]
            # Area_match = Area[i, j]
            error_min = 100.0
            for k in range(n_ite):
                alpha_calc[i, j] = lowlim[i, j] + k * (highlim[i, j] - lowlim[i, j]) / (n_ite - 1)
                # Area[k]=areafinder()
                Area[i, j], xl, xr, yl, yr = areafinder(xval[i, j], yval[i, j], mxval[i, j], myval[i, j], dx,
                                                        alpha_calc[i, j])
                error[i, j] = abs(Area[i, j] - F[i, j] * dx * dy)
                if error[i, j] < error_min:
                    # want to pull out xright,xleft,yright,yleft for (i,j) as well
                    error_min = error[i, j]
                    # Area_match=Area[i, j]
                    alpha_actual[i, j] = alpha_calc[i, j]
                    xright[i, j] = xr
                    xleft[i, j] = xl
                    yright[i, j] = yr
                    yleft[i, j] = yl
                    area_actual[i, j] = Area[i, j]
            # If the areafinder produces a large error with the orginal colorfunction value,
            # we keep the colorfunction area. This occurs for very small values,
            # so it is safe to include this.
            if (abs(area_actual[i, j] - F[i, j] * dx * dy)) / (F[i, j] * dx * dy) > 0.05:  # 5 percent error
                area_actual[i, j] = F[i, j] * dx * dy
                alpha_actual[i, j] = 0
            Cr[i, j] = area_actual[i, j] / dx / dy


@ti.func
def areafinder(x, y, mx, my, h, alpha):
    """
    This function calculated the area based on the alpha and the normal vector.
    It does this for the four possible line intersections for positive and negative slope.
    All calculate the area to the left of the intercepting line,
    so we have a condition that flips the area if needed.
    if cells counted clockwise sides (starting on left) 1, 2, 3, 4
    return: area,xleft,xright,yleft,yright
    """
    slope = -1 / (my / mx)
    area = 0.0
    xleft = x
    yleft = y
    xright = x
    yright = y
    # fudge factors for exactly horizontal and vertical lines
    if mx == 0 and my == 0:
        area = 0
        flag = 0
        # return
    elif my == 0 and mx != 0:
        my = 1e-10
    elif my != 0 and mx == 0:
        mx = 1e-10
    if slope < 0:
        if alpha / mx > h and alpha / my > h:
            # line passes through 2,3
            # counter clockwise
            # origin, right corner, right triangle, left triangle, left corner
            points = ti.Matrix(
                [[x, y], [x + h, y], [x + h, y + (alpha / mx - h) * -slope], [x + (alpha / my - h) / -slope, y + h],
                 [x, y + h]])
            area = polyarea(points)
            xright = x + h
            xleft = x + (alpha / my - h) / -slope
            yright = y + (alpha / mx - h) * -slope
        elif alpha / mx > h > alpha / my:
            # line passes through 1,3
            # counter clockwise
            # origin, right corner, right up, left up
            points = ti.Matrix(
                [[x, y], [x + h, y], [x + h, y + (alpha / mx - h) * -slope], [x, y + alpha / my]])
            area = polyarea(points)
            xleft = x
            xright = x + h
            yleft = y + (alpha / my)
            yright = y + (alpha / mx - h) * -slope
        elif alpha / mx < h < alpha / my:
            # line passes through 2,4
            # counter clockwise
            # origin, right, left up, left corner
            points = ti.Matrix(
                [[x, y], [x + alpha / mx, y], [x + (alpha / my - h) / -slope, y + h], [x, y + h]])
            area = polyarea(points)
            xleft = x + (alpha / my - h) / -slope
            xright = x + alpha / mx
            yleft = y + h
            yright = y
        elif alpha / mx < h and alpha / my < h:
            # line passes through 1,4
            # counter clockwise
            # origin, right, left up
            points = ti.Matrix([[x, y], [x + alpha / mx, y], [x, y + (alpha / my)]])
            area = polyarea(points)
            xleft = x
            xright = x + (alpha / mx)
            yleft = y + (alpha / my)
            yright = y
    if slope > 0:
        if alpha / mx > 0 and (h - (alpha / mx)) * slope > h:
            # line passes through 4,2
            # counter clockwise
            # origin, right, right up, top right corner, left corner
            points = ti.Matrix(
                [[x, y], [x + alpha / mx, y], [x + (h / slope) + (alpha / mx), y + h], [x, y + h]])
            area = polyarea(points)
            xleft = x + (alpha / mx)
            xright = x + (h / slope) + (alpha / mx)
            yleft = y
            yright = y + h
        elif alpha / mx < 0 and slope * (h) + alpha / my < h:
            # line passes through 1,3
            # counter clockwise:
            # bottom, right, topr corner, topleft corner
            points = ti.Matrix(
                [[x, y + alpha / my], [x + h, y + slope * h + alpha / my], [x + h, y + h], [x, y + h]])
            area = polyarea(points)
            xleft = x
            xright = x + h
            yleft = y + alpha / my
            yright = y + slope * h + alpha / my
        elif alpha / mx < 0 and slope * h + alpha / my > h:
            # line passes through 1,2
            # counter clockwise:
            # bottom, right, topleft corner
            points = ti.Matrix(
                [[x, y + alpha / my], [x + (h - (alpha / my)) / slope, y + h], [x, y + h]])
            area = polyarea(points)
            xleft = x
            xright = x + (h - (alpha / my)) / slope
            yleft = y + alpha / my
            yright = y + h
        elif alpha / mx > 0 and (h - (alpha / mx)) * slope < h:
            # line passes through 4,3
            # counter clockwise
            # origin, right, right up, top right corner, left corner
            points = ti.Matrix(
                [[x, y], [x + (alpha / mx), y], [x + h, y + (h - alpha / mx) * slope], [x + h, y + h], [x, y + h]])
            area = polyarea(points)
            xleft = x + (alpha / mx)
            xright = x + h
            yleft = y
            yright = y + (h - alpha / mx) * slope

    # flips the area if need be
    if (mx < 0 and my > 0) or (mx < 0 and my < 0):
        area = h * h - area

    return area, xleft, xright, yleft, yright
    # return area


@ti.func
def polyarea(points):
    """
    Computes the area of a polygon, supporting non-convex cases
    :points: The vertices of the polygon have been sorted sequentially counterclockwise
    :return:Area of a polygon
    """
    area = 0.0
    q = ti.Matrix([points[points.n - 1, 0], points[points.n - 1, 1]])
    for i in ti.static(range(points.n)):
        p = [points[i, 0], points[i, 1]]
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return -area / 2


# Calculate the coordinates of the staggered point
grid_staggered()
# Set initial volume fraction
set_init_F()

# initialize rho, mu
rho = ti.field(float, shape=(imax + 2, jmax + 2))
mu = ti.field(float, shape=(imax + 2, jmax + 2))

# Create Laplace operator
Laplace_operator(L)
A = L.build()

istep = 0
istep_max = 50000

nstep = 100
R_limit = 15.0
count = -1
check_mass = np.zeros((int(istep_max / nstep), 1))  # 检查质量

while istep < istep_max:
    # set boundary conditions
    set_BC()
    istep = istep + 1
    # Update rho, mu by F
    cal_mu_rho()
    get_normal_young()
    get_alpha()
    cal_karpa()

    # Solving Pressure Poisson Equation Using Projection Method
    M_Possion()
    solver = ti.linalg.SparseSolver(solver_type="LU")
    solver.analyze_pattern(A)
    solver.factorize(A)
    pv_init = solver.solve(R)
    pv.from_numpy(pv_init)
    isSuccess = solver.info()
    update()
    solve_F()

    get_normal_young()  # compute mx and my of each cell
    get_alpha()
    cal_karpa()

    set_BC()
    istep = istep + 1  # time step +1
    if np.mod(istep, nstep) == 0:  # Output data every 100 steps
        count = count + 1
        Fn1 = F.to_numpy()
        check_mass[count] = sum(sum(abs(Fn1[imin:imax + 1, jmin: jmax + 1])))
        print('Number of iterations', str(istep), '\n check mass：', str(check_mass[count]),
              '\n')
        plt.figure(figsize=(5, 5))
        xm1 = xm.to_numpy()
        ym1 = ym.to_numpy()

        X, Y = np.meshgrid(xm1[imin:imax + 1], ym1[jmin:jmax + 1])
        plt.contour(xm1[imin:imax + 1], ym1[jmin:jmax + 1], Fn1[imin:imax + 1, jmin:jmax + 1].T, [0.5], cmap=plt.cm.jet)
        plt.savefig(str(istep) + '.png')
