from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, vmap

@partial(jit, static_argnums=(1,))
def moving_window1D(a, size: int):
    starts = jnp.arange(len(a) - size + 1)
    return vmap(lambda start: jax.lax.dynamic_slice(a, (start,), (size,)))(starts)

a = jnp.arange(10)
aw = moving_window1D(a, 3)
print(aw)



def ker_heat1(window3, beta):

    out = window3[1] + beta * (window3[0] + 2 * window3[1] + window3[2])
    return out


def iterate_kernels(ker_fun, init, x, *fargs, **fkwargs):

    def f2scan(carry, xi):
        u_new = ker_fun(xi, *fargs, **fkwargs)
        return u_new, u_new

    carry, f_x = jax.lax.scan(f2scan,
                              init,
                              x)
    return f_x

fx = iterate_kernels(ker_heat1, init=0, x=aw, beta=1)



from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, vmap

@partial(jit, static_argnums=(1,))
def moving_window(matrix, window_shape):
    matrix_width = matrix.shape[1]
    matrix_height = matrix.shape[0]

    window_width = window_shape[0]
    window_height = window_shape[1]

    startsx = jnp.arange(matrix_width - window_width + 1)
    startsy = jnp.arange(matrix_height - window_height + 1)
    starts_xy = jnp.dstack(jnp.meshgrid(startsx, startsy)).reshape(-1, 2) # cartesian product => [[x,y], [x,y], ...]

    return vmap(lambda start: jax.lax.dynamic_slice(matrix, (start[1], start[0]), (window_height, window_width)))(starts_xy)

matrix = jnp.asarray([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
print(moving_window(matrix, (2, 3))) # window width = 2, window height = 3


###################################
from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, vmap

#@partial(jit, static_argnums=(1,))
def moving_window2D(matrix, window_shape):
    matrix_width = matrix.shape[1]
    matrix_height = matrix.shape[0]

    window_width = window_shape[0]
    window_height = window_shape[1]

    startsx = jnp.arange(matrix_width - window_width + 1)
    startsy = jnp.arange(matrix_height - window_height + 1)
    starts_xy = jnp.dstack(jnp.meshgrid(startsx, startsy)).reshape(-1, 2) # cartesian product => [[x,y], [x,y], ...]

    return vmap(lambda start: jax.lax.dynamic_slice(matrix, (start[1], start[0]), (window_height, window_width)))(starts_xy)

matrix = jnp.asarray([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
print(moving_window2D(matrix, (3, 3))) # window width = 2, window height = 3


@partial(jit, static_argnums=(1,))
def moving_window3D(matrix, window_shape):

    matrix_deep = matrix.shape[2]
    matrix_width = matrix.shape[1]
    matrix_height = matrix.shape[0]

    
    window_width = window_shape[0]
    window_height = window_shape[1]
    window_deep = window_shape[2]

    startsx = jnp.arange(matrix_width - window_width + 1)
    startsy = jnp.arange(matrix_height - window_height + 1)
    startsz = jnp.arange(matrix_deep - window_deep + 1)
    starts_xyz = jnp.dstack(jnp.meshgrid(startsx, startsy, startsz)).reshape(-1, 2)
    # cartesian product => [[x,y], [x,y], ...]

    return vmap(lambda start: jax.lax.dynamic_slice(matrix, (start[1], start[0]), (window_height, window_width)))(starts_xy)
