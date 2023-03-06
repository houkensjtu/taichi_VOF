import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

ti.init(arch=ti.cpu)
nx = 64
x = np.arange(nx)

q = ti.field(dtype=ti.f64, shape=(nx,))
qtd = ti.field(dtype=ti.f64, shape=(nx,))
qn = ti.field(dtype=ti.f64, shape=(nx,))

f = ti.field(dtype=ti.f64, shape=(nx + 1, ))
fh = ti.field(dtype=ti.f64, shape=(nx + 1, ))
fl = ti.field(dtype=ti.f64, shape=(nx + 1, ))
a = ti.field(dtype=ti.f64, shape=(nx + 1, ))
c = ti.field(dtype=ti.f64, shape=(nx + 1, ))

rp = ti.field(dtype=ti.f64, shape=(nx, ))
rm = ti.field(dtype=ti.f64, shape=(nx, ))

# CFL = u * dt / dx = 0.1 / 0.5 < 1.0
dx = 0.5
u = 0.1
dt = 1.0

@ti.kernel
def init():
    for I in ti.grouped(q):
        q[I] = 0.0
        qn[I] = q[I]

    for i in range(3, 10):
        q[i] = 1.0
        qn[i] = q[i]        

@ti.kernel
def advect_fct_book():
    f[0] = 0
    f[nx] = 0    
    for i in range(nx - 1):
        # fh[i + 1] = (q[i] + q[i + 1]) / 2 * u  # Central (Second-order)
        # f[i + 1] = q[i]  * u  # Up-wind
        # fh[i + 1] = q[i + 1]  * u  # Down-wind
        fh[i + 1] = (7.0 / 12 * (q[i + 1] + q[i]) - 1.0 / 12 * (q[i + 2] + q[i - 1]))  * u  # Fourth-order
        fl[i + 1] = q[i] * u
        a[i + 1] = fh[i + 1] - fl[i + 1]
        
    for i in qtd:
        # q[i] = qn[i] + (f[i] - f[i + 1]) / dx * dt
        qtd[i] = qn[i] + (fl[i] - fl[i + 1]) / dx * dt

    for i in range(nx - 1):
        # c = 0.6
        # a[i] = a[i] * c
        s = 0.0
        if a[i + 1] >= 0:
            s = 1.0
        else:
            s = -1.0
        a[i + 1] = s * ti.max(0, ti.min(ti.abs(a[i + 1]), s * (qtd[i + 2] - qtd[i + 1]) * dx / dt, s * (qtd[i] - qtd[i - 1]) * dx / dt))
        
    for i in q:
        q[i] = qtd[i] + (a[i] - a[i + 1]) / dx * dt
        
    for i in qn:
        qn[i] = q[i]

@ti.kernel
def advect_fct_zalesak():
    f[0] = 0
    f[nx] = 0    
    for i in range(nx - 1):
        fh[i + 1] = (q[i] + q[i + 1]) / 2 * u  # Central (Second-order)
        # fh[i + 1] = q[i + 1]  * u  # Down-wind
        # fh[i + 1] = (7.0 / 12 * (q[i + 1] + q[i]) - 1.0 / 12 * (q[i + 2] + q[i - 1]))  * u  # Fourth-order
        fl[i + 1] = q[i] * u  # Upwind
        a[i + 1] = fh[i + 1] - fl[i + 1]
        
    for i in qtd:
        qtd[i] = qn[i] + (fl[i] - fl[i + 1]) / dx * dt

    for i in q:
        qmax = ti.max(qtd[i - 1], qtd[i], qtd[i + 1])
        qmin = ti.min(qtd[i - 1], qtd[i], qtd[i + 1])        
        pp = ti.max(0, a[i]) - ti.min(0, a[i + 1])
        qp = (qmax - qtd[i]) * dx / dt
        if pp > 0:
            rp[i] = ti.min(1, qp / pp)
        else:
            rp[i] = 0
            
        pm = ti.max(0, a[i + 1]) - ti.min(0, a[i])
        qm = (qtd[i] - qmin) * dx / dt
        if pm > 0:
            rm[i] = ti.min(1, qm / pm)
        else:
            rm[i] = 0

    for i in range(nx - 1):
        if a[i + 1] >= 0:
            c[i + 1] = ti.min(rp[i + 1], rm[i])
        else:
            c[i + 1] = ti.min(rp[i], rm[i + 1])

    for i in q:
        q[i] = qtd[i] + (a[i]*c[i] - a[i + 1]*c[i + 1]) / dx * dt
        
    for i in qn:
        qn[i] = q[i]
        

init()
max_iter = 100
for istep in range(max_iter):
    advect_fct_zalesak()
    qnp = q.to_numpy()
    # print(qnp)
    plt.plot(x, qnp)
    plt.show()
