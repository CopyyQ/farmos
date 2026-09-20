from __future__ import annotations

from kaggrl.v4_hybrid_policy import FarmOSV4HybridPolicy


class V4HybridRolloutAgent:
    """CPU-only macro-first FarmOS agent."""

    def __init__(self, *, min_confidence: float = 0.80):
        self.policy = FarmOSV4HybridPolicy(
            residual=None,
            min_confidence=min_confidence,
        )

    def reset(self) -> None:
        self.policy.reset()

    def __call__(self, observation, configuration=None):
        return self.policy.act(observation, configuration)


def agent(observation, configuration=None):
    global _DEFAULT_AGENT
    try:
        instance = _DEFAULT_AGENT
    except NameError:
        instance = _DEFAULT_AGENT = V4HybridRolloutAgent()
    return instance(observation, configuration)
