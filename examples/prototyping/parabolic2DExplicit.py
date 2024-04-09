import jax
import jax.numpy as jnp

def build_2Dgrid(x_1, x_2, nx,
                 y_1, y_2, ny):

    grid_x = jnp.linspace(x_1, x_2, nx)
    grid_y = jnp.linspace(y_1, y_2, ny)
    x1, y1 = jnp.meshgrid(grid_y, grid_x)
    xy = jnp.stack([y1,x1], axis=2)
    return xy

class HeatModel:

    def __init__(self, alpha):

        self.alpha = alpha

    def coefficient_record(self):

        return ["a"]
    
    def a(self, t, x):

        return self.alpha

    def b(self, t, x):

        return 0.

    def c(self, t, x):

        return 0.

    def d(self, t, x):

        return 0.

class Grid2D:

    def __init__(self, model, t_1, t_2, nt,
                 x_1, x_2, nx):

        ...

        
    def build_2Dgrid(self):

        grid_x = jnp.linspace(x_1, x_2, nx)
        grid_y = jnp.linspace(y_1, y_2, ny)
        x1, y1 = jnp.meshgrid(grid_y, grid_x)
        xy = jnp.stack([y1,x1], axis=2)
        return xy

    def build_coefficients():

        def coeff_i(fun):
            f1 = jax.vmap(fnojit, in_axes=(0,None))
            f2 = jax.vmap(f1, in_axes=(None, 0))
            out = f2(x,y)
            return out

        for ci in self.model.coefficient_record:
            



x=jnp.arange(5)
y=jnp.arange(1,7,1)

f1 = jax.vmap(f, in_axes=(0,None))
f2 = jax.vmap(f1, in_axes=(None, 0))

fxy = f2(x,y)


import time
import jax
import jax.numpy as jnp

@jax.jit
def f(x, y):

    return x+y


@jax.jit
def ftensor(x, y):

    Y = jnp.tensordot(y,jnp.ones(len(x)), axes=0)
    X = jnp.tensordot(x,jnp.ones(len(y)), axes=0).T
    out = f(X, Y)
    return out

#@jax.jit
def fnojit(x, y):

    return x+y

#@jax.jit
def fvmap(x, y):

    f1 = jax.vmap(fnojit, in_axes=(0,None))
    f2 = jax.vmap(f1, in_axes=(None, 0))
    out = f2(x,y)
    return out

x=jnp.arange(5e4, dtype=float)
y=jnp.arange(1,5e4+1,1, dtype=float)

t1_ftensor = time.time()
ztensor = ftensor(x, y)
t2_ftensor = time.time()
t1_fvmap = time.time()
zvmap = fvmap(x, y)
t2_fvmap = time.time()

print(f"tensor: {t2_ftensor - t1_ftensor}")
print(f"vmap: {t2_fvmap - t1_fvmap}")

    

class Parabolic2DExplicit:


    def __init__(self,
                 a,
                 b,
                 c,
                 d,
                 dt,
                 dx):

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.dt = dt
        self.dx = dx

    def build_terms(self):

        self.A = self.dt *(self.b / 2.0 - self.a / self.dx) / self.dx
        self.B = 1.0 - self.dt * self.c + 2.0 * self.dt * self.a / self.dx ** 2
        self.C = - self.dt * (self.b / 2. + self.a / self.dx) / self.dx
        self.D = - self.dt * self.d

    def build_Ax(self):

        def f1(Ai, Bi, Ci):

            Axi = filldiag2(jnp.zeros((self.nx, self.nx)), Bi)
            Axi = filldiag2(Axi, Ai, offset = -1)
            Axi = filldiag2(Axi, Ci, offset = +1)

        vmap_f1 = jax.vmap(f1, in_axes=(0, 0, 0))
        self.Ax = vmap_f1(self.A, self.B, self.C)

    def build_ux(self):

        def f1(Ai, Ci, Di, bd_loweri, bd_upperi):

            uxi = Di
            uxi = uxi.at[0].set(uxi[0] + Ai[0] * bd_loweri)
            uxi = uxi.at[0].set(uxi[-1] + Ci[-1] * bd_upperi)

        vmap_f1 = jax.vmap(f1, in_axes=(0, 0, 0, 0, 0))
        self.ux = vmap_f1(self.A, self.B, self.C, self.bd_lower, self.bd_upper)

    def solve1(self):

        def loop(carry, x):

            i, xprevious = carry
            xnew = self.Ax[i].dot(xprevious) + self.ux[i]
            return (i+1, xnew), xnew

        carry, y = jax.lax.scan(loop, (0, self.x0), None, length=self.tn)

    def solve1_const(self):

        def loop(carry, x):

            i, xprevious = carry
            xnew = self.Ax.dot(xprevious) + self.ux[i]
            return (i+1, xnew), xnew

        carry, y = jax.lax.scan(loop, (0, self.x0), None, length=self.tn)



        
def scantry(Ai, x0):

    def toscan(carry, x):

        i, xprevious = carry
        xnew = Ai[i].dot(xprevious)
        return (i+1, xnew), xnew

    #carry, y = jax.lax.scan(toscan, (0, x0), jnp.ones(len(Ai)))
    carry, y = jax.lax.scan(toscan, (0, x0), None, length=len(Ai))
    return y



def filldiag2(matrix, value, offset=0):
    nx, ny = matrix.shape
    nx0 = abs(min(0, offset))
    ny0 = abs(max(0, offset))
    n = min(nx - nx0, ny - ny0)
    index_x = jnp.arange(nx0, nx0 + n)
    index_y = jnp.arange(ny0, ny0 + n)
    matrix = matrix.at[(index_x, index_y)].set(value)
    return matrix
