"""Reproduce the paper's storage-only arithmetic without a QEC claim."""

from yoked_storage import estimate


def main():
    case = estimate(1280, 131, compute_patches=7 * 18)
    assert case.cold == 550400
    assert case.hot == 177112
    assert case.compute == 170352
    assert case.total == 897864
    assert case.no_yoking == 2078024
    assert case.saved == 1180160
    assert estimate(0, 2).saved == 0
    for kw in ({"cold_logical": -1, "hot_logical": 2},
               {"cold_logical": True, "hot_logical": 2},
               {"cold_logical": 1, "hot_logical": 2, "distance": 23},
               {"cold_logical": 1, "hot_logical": 2, "target_error": 1e-12},
               {"cold_logical": 1, "hot_logical": 2, "physical_error": 0.002}):
        try:
            estimate(**kw)
        except ValueError:
            pass
        else:
            raise AssertionError(kw)
    print("test_yoked_storage: fixed-assumption paper footprint reproduced")


if __name__ == "__main__":
    main()
