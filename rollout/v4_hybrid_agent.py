from __future__ import annotations

from kaggrl.v4_hybrid_policy import FarmOSV4HybridPolicy


class V4HybridRolloutAgent:
    """CPU macro-first FarmOS agent with optional step-strategy policy."""

    def __init__(
        self,
        *,
        option_policy=None,
        min_confidence: float = 0.80,
        min_option_confidence: float = 0.65,
    ):
        self.policy = FarmOSV4HybridPolicy(
            residual=None,
            option_policy=option_policy,
            min_confidence=min_confidence,
            min_option_confidence=min_option_confidence,
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
