import jax.numpy as jnp

def build_1Dgrid(x_1,x_2,nx):

    grid_x = jnp.linspace(x_1, x_2, nx)
    return grid_x

def build_2Dgrid(x_1, x_2, nx,
                 y_1, y_2, ny):

    grid_x = jnp.linspace(x_1, x_2, nx)
    grid_y = jnp.linspace(y_1, y_2, ny)
    x1, y1 = jnp.meshgrid(grid_x, grid_y)
    xy = jnp.stack([x1,y1], axis=2)
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


def heat_eq1(t1, x0, x1, nt, nx, alpha):

    dx = (x1 - x0) / (nx - 1)
    dt = t1 / (nt - 1)
    beta = alpha * dt / dx **2
    grid = build_2Dgrid(t_0, t_1, nt,
                        x_0, x_1, nx)

    fgrid = build_fgrid(nt, nx)
    fgrid[0] = jnp.sin()
    x_indexes = jnp.arange(1, nx)
    def time_loop(carry_t, x_t):

        u_n1 = jnp.zeros(nx)
        def x_loop(carry_i, x_x):

            u_n1 = carry_i + beta * (
                carry_t[carry_i + 1] - 2 * carry_t[carry_i] + carry_t[carry_i - 1]
            )
            carry_new = [u_n1, ]
        last_carry, u_t = jax.lax.scan(x_loop, init, carry_t)
        return u_t, u_t
    
    last_carry, u_tx = jax.lax.scan(time_loop, init, grid)

def time_loop(nx):

    x_loop = jnp.arange(1, nx)
    u_n1 = jnp.zeros(nx)
    def x_loop(carry_i, x_x):

        carry_i = carry_i.at[x_x].set(x_x)
        return carry_i, carry_i

    last_carry, u_t = jax.lax.scan(x_loop, 0, jnp.arange(1,10))
    return u_n1

u_tx = time_loop(11)
