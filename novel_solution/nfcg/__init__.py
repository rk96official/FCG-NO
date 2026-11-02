import jax

jax.config.update("jax_enable_x64", True)

from nfcg.data.random_fields import register_random_fields  # noqa: E402

register_random_fields()
