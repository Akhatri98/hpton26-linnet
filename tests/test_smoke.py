from solver.native import Native


def test_native_solver_and_optimizer_available():
    """The C++ core builds, loads against libz3, and answers on both back ends."""
    s = Native()
    s.product("X", 1, True)
    assert s.check(s.tracked()) == "sat"
    assert len(s.dose_times("X", 1)) == 1

    o = Native(optimize=True)
    o.product("X", 1, True)
    o.minimize_times()
    assert o.check(o.tracked()) == "sat"
    assert o.dose_times("X", 1) == [0]
