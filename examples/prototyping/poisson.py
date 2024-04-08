import jax
import jax.numpy as jnp

nx = 5
ny = 6
d = jnp.zeros((nx, ny))
d.diagonal()

nx = 5
ny = 5
d2 = jnp.zeros((nx, ny))
d2.diagonal()



def fill_diagonal(a, val):
  assert a.ndim >= 2
  i, j = jnp.diag_indices(min(a.shape[-2:]))
  return a.at[..., i, j].set(val)

a = jnp.zeros((2, 3, 4, 4))

# works for scalars
a1 = fill_diagonal(a, 2)

# or for batched vectors
a2 = fill_diagonal(a, jnp.arange(24).reshape(2, 3, 4))

a3 = fill_diagonal(jnp.arange(1,7).reshape(2, 3), jnp.zeros(2))


def filldiag2(matrix, value, offset=0):
    nx, ny = matrix.shape
    nx0 = abs(min(0, offset))
    ny0 = abs(max(0, offset))
    n = min(nx - nx0, ny - ny0)
    index_x = jnp.arange(nx0, nx0 + n)
    index_y = jnp.arange(ny0, ny0 + n)
    matrix = matrix.at[(index_x, index_y)].set(value)
    return matrix

b = jnp.zeros((3, 3))

# works for scalars
#b1 = fill_diagonal(b, jnp.array([1,2,3]))

def Parabolic2DExplicit():

    ...


def scantry(Ai, x0):

    def toscan(carry, x):

        i, xprevious = carry
        xnew = Ai[i].dot(xprevious)
        return (i+1, xnew), xnew

    #carry, y = jax.lax.scan(toscan, (0, x0), jnp.ones(len(Ai)))
    carry, y = jax.lax.scan(toscan, (0, x0), None, length=len(Ai))
    return y

A =jnp.arange(5*3*3).reshape((5,3,3))
y = scantry(A, jnp.ones(3))




