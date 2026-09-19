CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
UNIT_OPS = ("PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "PLACE", "DROP", "PLANT", "WATER", "HARVEST", "FERTILIZE", "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE", "DIG")
MARKET_OPS = ("NONE", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND")
ITEMS = CROPS + ANIMALS + PRODUCTS
ITEM_TO_ID = {name: i + 1 for i, name in enumerate(dict.fromkeys(ITEMS))}
ID_TO_ITEM = {v: k for k, v in ITEM_TO_ID.items()}
UNIT_TO_ID = {name: i for i, name in enumerate(UNIT_OPS)}
ID_TO_UNIT = {v: k for k, v in UNIT_TO_ID.items()}
MARKET_TO_ID = {name: i for i, name in enumerate(MARKET_OPS)}
ID_TO_MARKET = {v: k for k, v in MARKET_TO_ID.items()}
MAX_QUANTITY = 100

ITEM_CLASSES = max(ITEM_TO_ID.values()) + 1
QTY_CLASSES = MAX_QUANTITY + 1
