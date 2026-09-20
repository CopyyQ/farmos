from __future__ import annotations

import numpy as np

OBJECTIVE_VERSION = "terminal_margin_advantage_weighted_bc_v1"
MARGIN_SCALE = 10000.0
MARGIN_TEMPERATURE = 5000.0


def outcome_policy_weight(final_margin):
    margin = np.asarray(final_margin, dtype=np.float32)
    sigmoid = 1.0 / (1.0 + np.exp(-margin / MARGIN_TEMPERATURE))
    return (0.25 + 2.75 * sigmoid).astype(np.float32)
