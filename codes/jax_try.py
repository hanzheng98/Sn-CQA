import jax

def f(x):
    return jax.numpy.matmul(x, x)

df_dx = jax.vmap(jax.grad(f))
mat = jax.random.normal(jax.random.PRNGKey(1), (3,3))
print(mat)
print(df_dx(mat)/2)