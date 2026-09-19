from kaggrl.v3_3_numpy_runtime import V33NumpyPolicy
from rollout.v3_agent_numpy import V3NumpyRolloutAgent


class V33NumpyRolloutAgent(V3NumpyRolloutAgent):
    POLICY_CLASS = V33NumpyPolicy
