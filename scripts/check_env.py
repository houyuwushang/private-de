#!/usr/bin/env python
from __future__ import annotations

import jax
import jax.numpy as jnp

print("JAX version:", jax.__version__)
print("Devices:", jax.devices())
print("Local device count:", jax.local_device_count())

x = jnp.ones((4096, 4096), dtype=jnp.float32)
y = x @ x
print("Result:", float(y[0, 0]))
