import z3


def test_z3_solver_and_optimize_available():
    x = z3.Real("x")
    s = z3.Solver()
    s.add(x > 1, x < 2)
    assert s.check() == z3.sat

    o = z3.Optimize()
    o.add(x >= 0)
    o.minimize(x)
    assert o.check() == z3.sat
    assert o.model()[x] == 0
