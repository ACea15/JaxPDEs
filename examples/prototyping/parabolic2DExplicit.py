import jax
import jax.numpy as jnp
from dataclasses import dataclass

@dataclass
class Grid2D:

    x1_0: float = None
    x1_1: float = None
    x2_0: float = None
    x2_1: float = None
    x1n: int = None
    x2n: int = None
    dx1: float = None
    dx2: float = None
    x1: jnp.array = None
    x2: jnp.array = None

    def __post_init__(self):

        if self.x1 is None:
            object.__setattr__(self, 'x1', jnp.linspace(self.x1_0,
                                                        self.x1_1,
                                                        self.x1n))
        if self.x2 is None:
            object.__setattr__(self, 'x2', jnp.linspace(self.x2_0,
                                                        self.x2_1,
                                                        self.x2n))

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

class GridMapper:

    def __init__(self, model, boundary):

        self.model = model
        self.boundary = boundary

    def build_grid(self):

        grid_x = jnp.linspace(x_1, x_2, nx)
        grid_y = jnp.linspace(y_1, y_2, ny)
        x1, y1 = jnp.meshgrid(grid_y, grid_x)
        xy = jnp.stack([y1,x1], axis=2)
        return xy

    def build_coefficient(self):

        def coeff_i(fun):
            f1 = jax.vmap(fun, in_axes=(0,None))
            f2 = jax.vmap(f1, in_axes=(None, 0))
            out = f2(x,y)
            return out

        for ci in self.model.coefficient_record:
            ...


class Parabolic2DExplicit:

    def __init__(self, model, grid):

        ...

    def build_coefficients(self):

        self.a, a_const = 
        self.b, b_const = 
        
    def build_terms(self):

        self.A = self.dt * (self.b / 2.0 - self.a / self.dx) / self.dx
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

class MyClass:
    def __init__(self, value):
        self.value = value

class AnotherClass:
    def __init__(self, obj):
        self.obj_ref = obj

obj1 = MyClass([5])
obj2 = AnotherClass(obj1)

print(obj1.value)  # This will print 5
obj2.obj_ref.value = [10]
print(obj1.value)  # This will print 10 because obj1 and obj2.obj_ref refer to the same object
