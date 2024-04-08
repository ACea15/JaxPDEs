import jax.numpy as jnp
import jax

DK = jnp.array([[True, False],[True, True],[False, False],[False,True]])
f1 = lambda: 1
f2 = lambda: 0
cond = lambda dk: jax.lax.cond(dk,f1,f2)
vcond = jax.vmap(jax.vmap(cond))
result = vcond(DK)
print(result)


