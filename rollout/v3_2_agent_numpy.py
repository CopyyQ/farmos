from kaggrl.v3_2_numpy_runtime import V32NumpyPolicy
from rollout.v3_agent_numpy import V3NumpyRolloutAgent


class V32NumpyRolloutAgent(V3NumpyRolloutAgent):
    POLICY_CLASS = V32NumpyPolicy
