import jax.numpy as jnp
import jax
from functools import partial

def build_1Dgrid(x_1,x_2,nx):

    grid_x = jnp.linspace(x_1, x_2, nx)
    return grid_x

def build_2Dgrid(x_1, x_2, nx,
                 y_1, y_2, ny):

    grid_x = jnp.linspace(x_1, x_2, nx)
    grid_y = jnp.linspace(y_1, y_2, ny)
    x1, y1 = jnp.meshgrid(grid_y, grid_x)
    xy = jnp.stack([y1,x1], axis=2)
    return xy

def build_3Dgrid(x_1, x_2, nx,
                 y_1, y_2, ny,
                 z_1, z_2, nz):

    grid_x = jnp.linspace(x_1, x_2, nx)
    grid_y = jnp.linspace(y_1, y_2, ny)
    grid_z = jnp.linspace(z_1, z_2, nz)
    x1, y1, z1 = jnp.meshgrid(grid_x, grid_y, grid_z)
    xyz = jnp.stack([x1, y1, z1], axis=3)
    return xyz

def build_fgrid(*dimensions):

    fgrid0 = jnp.zeros(dimensions)
    return fgrid0

def setICt0(fgrid, x):

    fgrid = fgrid.at[0].set(x)
    return fgrid


def heat_eq1(t_1, x_0, x_1, nt, nx, alpha, t_0=0.):

    dx = (x_1 - x_0) / (nx - 1)
    dt = t_1 / (nt - 1)
    beta = alpha * dt / dx **2
    grid = build_2Dgrid(t_0, t_1, nt,
                        x_0, x_1, nx)

    fgrid = build_fgrid(nt, nx)
    x0 = grid[0, :][:, 1]
    fgrid = fgrid.at[0].set(jnp.sin(jnp.pi * x0))
    # fgrid[:,0] = fgrid[:,-1] = 0

    def time_loop(carry_t, x_t):

        def x_loop(carry_i, xi):

            f_km1_i, f_km1_im1 = carry_i
            f_k_i = f_km1_i + beta * (
                xi - 2 * f_km1_i + f_km1_im1 
            )
            carry_new = jnp.array([xi, f_km1_i])
            return carry_new, f_k_i

        last_carry, ut_inside = jax.lax.scan(x_loop, carry_t[:2], carry_t[2:])
        u_t = jnp.hstack([x_t[0], ut_inside, x_t[-1]])
        return u_t, u_t

    last_carry, u_tx = jax.lax.scan(time_loop, fgrid[0], fgrid[1:])
    return u_tx

#####################
#@partial(jax.jit, static_argnums=(1,))
def moving_window1D(a, size: int):
    starts = jnp.arange(len(a) - size + 1)
    return jax.vmap(lambda start: jax.lax.dynamic_slice(a, (start,), (size,)))(starts)

a = jnp.arange(10)
aw = moving_window1D(a, 3)
print(aw)

def ker_heat1(window3, beta):

    out = window3[1] + beta * (window3[0] - 2 * window3[1] + window3[2])
    return out

def iterate_kernels(ker_fun, init, x, *fargs, **fkwargs):

    def f2scan(carry, xi):
        u_new = ker_fun(xi, *fargs, **fkwargs)
        return u_new, u_new

    carry, f_x = jax.lax.scan(f2scan,
                              init,
                              x)
    return f_x

def heat_eq2(t_1, x_0, x_1, nt, nx, alpha, t_0=0.):

    dx = (x_1 - x_0) / (nx - 1)
    dt = t_1 / (nt - 1)
    beta = alpha * dt / dx **2
    grid = build_2Dgrid(t_0, t_1, nt,
                        x_0, x_1, nx)

    fgrid = build_fgrid(nt, nx)
    x0 = grid[0, :][:, 1]
    fgrid = fgrid.at[0].set(jnp.sin(jnp.pi * x0))
    # fgrid[:,0] = fgrid[:,-1] = 0

    def time_loop(carry_t, x_t):

        carry_window = moving_window1D(carry_t, size=3)
        # last_carry, ut_inside = jax.lax.scan(x_loop, carry_t[:2], carry_t[2:])
        ut_inside = iterate_kernels(ker_heat1, init=0, x=carry_window, beta=1)
        u_t = jnp.hstack([x_t[0], ut_inside, x_t[-1]])
        return u_t, u_t

    last_carry, u_tx = jax.lax.scan(time_loop, fgrid[0], fgrid[1:])
    return u_tx



fheat = heat_eq1(t_1=1., x_0=-1., x_1=1., nt=400, nx=80, alpha=1./jnp.pi**2)
fheat2 = heat_eq2(t_1=1., x_0=-1., x_1=1., nt=400, nx=80, alpha=1./jnp.pi**2)



# def time_loop(nx):

#     x_loop = jnp.arange(1, nx)
#     u_n1 = jnp.zeros(nx)
#     def x_loop(carry_i, x_x):

#         carry_i = carry_i.at[x_x].set(x_x)
#         return carry_i, carry_i

#     last_carry, u_t = jax.lax.scan(x_loop, 0, jnp.arange(1,10))
#     return u_n1

# u_tx = time_loop(11)
