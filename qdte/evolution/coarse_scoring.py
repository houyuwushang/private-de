from __future__ import annotations

import jax
import jax.numpy as jnp

from qdte.evolution.coarse_precision import CoarsenedInteractionPrecision
from qdte.evolution.scoring import OrthogonalPrecisionScoreContext


def prepare_coarsened_precision_score_context(
    precision: CoarsenedInteractionPrecision,
) -> OrthogonalPrecisionScoreContext:
    block_specs = []
    offset = 0
    for block in precision.blocks:
        end = offset + block.rank
        block_specs.append(
            (
                block.scope[0],
                block.scope[1] if len(block.scope) == 2 else -1,
                offset,
                end,
                jax.device_put(jnp.asarray(block.left_features, dtype=jnp.float32)),
                (
                    jax.device_put(
                        jnp.asarray(block.right_features, dtype=jnp.float32)
                    )
                    if block.right_features is not None
                    else None
                ),
                jax.device_put(jnp.asarray(block.precision, dtype=jnp.float32)),
            )
        )
        offset = end

    @jax.jit
    def scorer(
        old_rows: jax.Array,
        new_rows: jax.Array,
        weighted_coefficient_residual: jax.Array,
        edit_cost: jax.Array,
        lambda_cost: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        linear = jnp.zeros(old_rows.shape[0], dtype=jnp.float32)
        quadratic = jnp.zeros(old_rows.shape[0], dtype=jnp.float32)
        for (
            left_attribute,
            right_attribute,
            start,
            end,
            left_features,
            right_features,
            block_precision,
        ) in block_specs:
            left_old = left_features[old_rows[:, left_attribute]]
            left_new = left_features[new_rows[:, left_attribute]]
            if right_features is None:
                delta = left_new - left_old
            else:
                right_old = right_features[old_rows[:, right_attribute]]
                right_new = right_features[new_rows[:, right_attribute]]
                delta = (
                    jnp.einsum("ni,nj->nij", left_new, right_new)
                    - jnp.einsum("ni,nj->nij", left_old, right_old)
                ).reshape(old_rows.shape[0], -1)
            linear = linear + delta @ weighted_coefficient_residual[start:end]
            quadratic = quadratic + jnp.einsum(
                "ni,ij,nj->n",
                delta,
                block_precision,
                delta,
                optimize=True,
            )
        quadratic = jnp.maximum(quadratic, jnp.float32(0.0))
        scores = (
            linear
            - 0.5 * quadratic
            - lambda_cost.astype(jnp.float32) * edit_cost.astype(jnp.float32)
        )
        return scores, quadratic

    return OrthogonalPrecisionScoreContext(
        scorer=scorer,
        coefficient_dimension=precision.coefficient_dimension,
        row_width=len(precision.cardinalities),
    )


__all__ = ["prepare_coarsened_precision_score_context"]
