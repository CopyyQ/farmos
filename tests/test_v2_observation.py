from copy import deepcopy

from kaggrl.v2_observation import normalize_observation, public_opponent_view


def _tiles():
    return [[None for _ in range(10)] for _ in range(10)]


def _observation():
    own_tiles = _tiles()
    rival_tiles = _tiles()
    own_tiles[6][7] = {
        "kind": "PLANT", "crop": "TOMATO", "planted_day": 3,
        "yield_units": 4, "watered_today": True, "consecutive_unwatered": 0,
        "fertilized_until_day": 19, "max_lifespan_step": 611,
    }
    own_tiles[2][3] = {
        "kind": "PASTURE", "animal": "COW", "placed_day": 2,
        "yield_units": 5, "fed_today": True, "cared_today": False,
        "consecutive_unfed": 1, "fertilizer_available": True,
        "pending_care_bonus": 2,
    }
    rival_tiles[1][1] = {"kind": "WEED"}
    hands = [[i % 10, (i + 1) % 10] for i in range(17)]
    inventories = [{"WHEAT": 999}] + [{"MILK": 1000 + i} for i in range(17)]
    return {
        "player": 0, "day": 8, "hour": 13,
        "farms": [
            {"money": 987654, "farmer": [4, 4], "hands": hands,
             "hires_today": 17, "unlocked_quadrants": ["NW", "NE"], "tiles": own_tiles},
            {"money": 456789, "farmer": [8, 8], "hands": [[9, 9]],
             "hires_today": 2, "unlocked_quadrants": ["NW"], "tiles": rival_tiles},
        ],
        "private": {"shed": {"MILK": 12345}, "seeds": {"TOMATO": 777}, "inventories": inventories},
        "market": {"inventory": {"MILK": 123456}, "prices": {"MILK": 987}},
        "town": {"unlocked_shops": ["YARN_STORE", "YARN_STORE", "BAKERY"]},
    }


def test_full_state_is_preserved_without_fixed_hand_cap_or_clipping():
    raw = _observation()
    state = normalize_observation(raw)
    assert state.step == 8 * 24 + 13
    assert len(state.own_units) == 18
    assert state.own_units[-1]["inventory"]["MILK"] == 1016
    assert state.own_grid[6][7]["fertilized_until_day"] == 19
    assert state.own_grid[6][7]["max_lifespan_step"] == 611
    assert state.own_grid[2][3]["pending_care_bonus"] == 2
    assert state.own["money"] == 987654
    assert state.private["shed"]["MILK"] == 12345
    assert state.private["seeds"]["TOMATO"] == 777
    assert state.market["inventory"]["MILK"] == 123456
    assert state.town_shops.count("YARN_STORE") == 2


def test_rival_view_contains_only_public_farm_state():
    raw = _observation()
    state = normalize_observation(raw)
    assert "private" not in state.rival
    assert state.rival["money"] == 456789
    assert state.rival_grid[1][1]["kind"] == "WEED"
    view = public_opponent_view(raw)
    assert view == state.rival
    raw["private"]["shed"]["MILK"] = 1
    assert "private" not in view


def test_normalization_is_deep_copy_and_does_not_mutate_raw_input():
    raw = _observation()
    before = deepcopy(raw)
    state = normalize_observation(raw)
    state.own_grid[6][7]["yield_units"] = 999
    assert raw == before
    assert raw["farms"][0]["tiles"][6][7]["yield_units"] == 4
