import jax
import jax.numpy as jnp

@jax.jit
def points(*dimens):

    state = jnp.meshgrid(*dimens, indexing = 'ij')
    p = jnp.stack(state, axis = -1)
    c = p.reshape(-1, len(dimens))

    return c

@jax.jit
def line(p1, p2):
    t = jnp.linspace(0, 1, 1_000)

    l = p1 + (t * (p2 - p1))
    line = l.reshape(-1, 2)

    return line

