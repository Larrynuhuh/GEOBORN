@jax.jit
def linlen(l):
    
    diff = jnp.diff(l, axis = 0)
    lens = jnp.linalg.norm(diff, axis = 1)
    sums = jnp.sum(lens)

    return sums


def midp(p1, p2):
    mid = (p1 + p2) / 2.0
    return mid

vmidp = jax.vmap(midp)

