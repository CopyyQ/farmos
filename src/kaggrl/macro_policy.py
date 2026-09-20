from __future__ import annotations

import copy

from .clock import resolve_step


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

    @staticmethod
    def _step(observation, configuration=None):
        """Return the canonical absolute turn index for either seat."""
        return resolve_step(observation, configuration)

    def _advance_route(self, observation, configuration=None):
        step = self._step(observation, configuration)
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

    def route_id(self, observation, configuration=None):
        self._advance_route(observation, configuration)
        return self._route

    def compatible_route_ids(self, observation, configuration=None):
        """Routes that are strategically compatible with the current shops.

        Before shop reveal there is only the opening route.  After reveal,
        compare only the old/new strategy route for the observed shop pair.
        From day 27 onward, also allow the dedicated liquidation route.
        """
        step = self._step(observation, configuration)
        if step < 144:
            return (0,)
        shops = tuple(
            (observation.get("town", {}).get("unlocked_shops", []) or [])[:2]
        )
        candidates = []
        for mapping in (self.old_shop_routes, self.new_shop_routes):
            route_id = mapping.get(shops)
            if route_id is not None and int(route_id) in self.routes:
                candidates.append(int(route_id))
        current = self.route_id(observation, configuration)
        if int(current) in self.routes:
            candidates.append(int(current))
        if step >= 648 and 2 in self.routes:
            candidates.append(2)
        if not candidates:
            candidates.append(int(current))
        return tuple(dict.fromkeys(candidates))

    def action_for_route(
        self,
        observation,
        route_id: int,
        configuration=None,
    ):
        step = self._step(observation, configuration)
        route = self.routes.get(int(route_id))
        if route is None or not 0 <= step < len(route):
            return {"farmer": ["PASS"], "hands": [], "market": []}
        action = copy.deepcopy(route[step])
        market = action.get("market") or []
        if step == 0 and market == [["BUY_PRODUCT", "WHEAT", 5], ["BUY_PRODUCT", "WHEAT", 10], ["SELL", "WHEAT", 60]]:
            action["market"] = [["BUY_PRODUCT", "WHEAT", 70], ["SELL", "WHEAT", 70]]
        if step >= 144:
            action = self._sales_first(action)
        return action

    def act(self, observation, configuration=None):
        self._advance_route(observation, configuration)
        return self.action_for_route(
            observation,
            self._route,
            configuration,
        )
