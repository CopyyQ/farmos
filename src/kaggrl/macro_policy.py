from __future__ import annotations

import copy


class MacroPolicy:
    def __init__(self, routes, new_shop_routes, old_shop_routes):
        self.routes = routes
        self.new_shop_routes = new_shop_routes
        self.old_shop_routes = old_shop_routes
        self.reset()

    def reset(self):
        self._route = 0
        self._day6 = False
        self._day27 = False

    def _advance_route(self, observation):
        step = int(observation["step"])
        if step >= 144 and not self._day6:
            shops = tuple((observation.get("town", {}).get("unlocked_shops", []) or [])[:2])
            if "YARN_STORE" in shops:
                self._route = self.old_shop_routes.get(shops, 0)
            else:
                self._route = self.new_shop_routes.get(shops, 100)
            self._day6 = True
        if step >= 648 and not self._day27:
            self._route = 2
            self._day27 = True

    @staticmethod
    def _sales_first(action):
        original = action.get("market", [])[:10]
        orders = [list(o) for o in original if o and (o[0] in ("HIRE", "BUY_LAND") or (len(o) >= 3 and int(o[2]) > 0))]
        for index in range(len(orders)):
            if orders[index][0] != "SELL":
                continue
            cursor = index
            while cursor > 0:
                previous = orders[cursor - 1]
                if previous[0] == "SELL":
                    break
                if previous[0] in ("BUY_PRODUCT", "BUY_ANIMAL") and previous[1] == orders[cursor][1]:
                    break
                orders[cursor - 1], orders[cursor] = orders[cursor], orders[cursor - 1]
                cursor -= 1
        action["market"] = orders
        return action

    def route_id(self, observation):
        self._advance_route(observation)
        return self._route

    def act(self, observation, configuration=None):
        step = int(observation["step"])
        self._advance_route(observation)
        action = copy.deepcopy(self.routes[self._route][step])
        market = action.get("market") or []
        if step == 0 and market == [["BUY_PRODUCT", "WHEAT", 5], ["BUY_PRODUCT", "WHEAT", 10], ["SELL", "WHEAT", 60]]:
            action["market"] = [["BUY_PRODUCT", "WHEAT", 70], ["SELL", "WHEAT", 70]]
        if step >= 144:
            action = self._sales_first(action)
        return action
