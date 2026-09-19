ARCHITECTURE_VERSION = "rl_v3_2b_strategy_hierarchical_market_nop"
FORMAT_VERSION = 4
STRATEGY_CORE_SCALE = 8.0
OPENING_ACTIVE_INTENT_SCALE = 8.0

# A CONTINUE slot can be a no-op as well as a real market order. Keeping
# NOP_SLOT in this vocabulary lets the decoder represent malformed/unexecutable
# source orders without turning them into a fake economic action.
ACTIVE_MARKET_OPS = (
    "NOP_SLOT", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL",
    "SELL", "HIRE", "BUY_LAND",
)
STOP_ID = 0
CONTINUE_ID = 1
