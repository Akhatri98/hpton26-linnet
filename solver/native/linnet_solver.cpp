// LINNET solver core: the Z3 encoding of a regimen over one repeating day, in
// linear real arithmetic, written against Z3's native C++ API (z3++.h).
//
// Shape (see the spec, "The solver core"):
//   - each product gets `max_doses` slots; slot k has a real time t[k] in [0, 24)
//     and a boolean on[k]
//   - activation is monotone (on[k+1] -> on[k]), so the active slots are a prefix
//   - active slots are strictly ordered (t[k] < t[k+1]), which is also symmetry
//     breaking; inactive slots are pinned to 0 so models are deterministic
//   - every distance is circular, because the card repeats daily
//
// Structural constraints are always asserted. Each label rule, and each
// dose-count requirement, is guarded by a tracking literal named after its id:
// `lit -> formula`. A check passes the ids it wants enforced as assumptions, so
// an unsat core names rules, and conflict shrinking can re-check any subset
// without rebuilding the encoding.
//
// Exposed as a flat C API (ln_*) so Python can drive it through ctypes. Hours
// and amounts cross the boundary as exact rationals in text ("n/d") and stay
// exact: nothing is ever rounded.
//
// Every ln_* function returns -1 (or NULL) on failure; ln_error() has the text.

#include <z3++.h>

#include <initializer_list>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#define LN_API extern "C" __declspec(dllexport)
#else
#define LN_API extern "C" __attribute__((visibility("default")))
#endif

using z3::expr;

namespace {

thread_local std::string g_error;

// One Z3 context for the process, as Z3's own Python binding uses. Never
// destroyed: encodings may outlive any static teardown order.
z3::context& context() {
    static z3::context* ctx = [] {
        z3::set_param("smt.random_seed", 0);  // deterministic re-runs
        return new z3::context();
    }();
    return *ctx;
}

// An exact rational that crossed the boundary as "n/d". Kept as a value and
// turned into a Z3 numeral only where a formula uses it, so that terms are
// created in a fixed order (see the note on Enc).
struct Q {
    long long n = 0, d = 1;
    static Q parse(const char* s) {
        std::string str(s);
        size_t slash = str.find('/');
        Q q;
        q.n = std::stoll(str.substr(0, slash));
        q.d = slash == std::string::npos ? 1 : std::stoll(str.substr(slash + 1));
        if (q.d <= 0) throw std::runtime_error("bad rational '" + str + "'");
        return q.reduced();
    }
    Q reduced() const {
        long long g = std::gcd(n < 0 ? -n : n, d);
        return g > 1 ? Q{n / g, d / g} : *this;
    }
    Q operator+(const Q& o) const { return Q{n * o.d + o.n * d, d * o.d}.reduced(); }
    Q operator-(const Q& o) const { return Q{n * o.d - o.n * d, d * o.d}.reduced(); }
    bool operator<=(const Q& o) const { return static_cast<__int128>(n) * o.d <= static_cast<__int128>(o.n) * d; }
    bool operator>(const Q& o) const { return !(*this <= o); }
    std::string str() const { return std::to_string(n) + "/" + std::to_string(d); }
};
const Q ZERO{0, 1}, DAY{24, 1};

struct Product {
    std::string name;
    bool prescribed;
    std::vector<expr> t, on;
    std::vector<std::pair<std::string, Q>> ingredients;  // name -> amount per dose
};

struct Dose {
    expr t, on;
};

// One thing tier 1 may change, with the term measuring how much it changed.
struct Component {
    char kind;  // 'm' meal shift, 's' sleep shift, 'd' doses dropped
    std::string name;
    expr term;
    std::string value;  // in the model last passed to ln_block_superset
};

// The formulas below are built one statement at a time, on purpose. C++ leaves
// the evaluation order of function arguments open, and the order in which
// terms are created is visible to Z3: it numbers them, and the numbering can
// steer which of several equally good models comes back. Spelling the order
// out keeps answers the same across compilers, and the same as the Python
// encoding this was ported from.
struct Enc {
    z3::context& ctx = context();
    std::optional<z3::solver> sol;
    std::optional<z3::optimize> opt;
    std::vector<Product> products;
    std::vector<std::pair<std::string, expr>> meals;  // fixed numeral or variable
    std::vector<std::string> tracked_ids;             // in the order they were tracked
    std::map<std::string, expr> tracked;              // id -> tracking literal
    std::vector<Component> components;
    std::vector<expr> costs;
    std::optional<expr> cost;  // the tier-1 objective
    std::string cost_value;
    std::vector<expr> blocked;  // terms of the last blocking clause, kept alive until the next
    std::optional<z3::model> model, previous_model;
    std::vector<std::string> core;
    std::string reason, scratch;

    Enc(bool optimize, unsigned timeout_ms) {
        if (optimize) {
            opt.emplace(ctx);
            z3::params p(ctx);
            p.set("timeout", timeout_ms);
            opt->set(p);
        } else {
            sol.emplace(ctx);
            sol->set("timeout", timeout_ms);
        }
    }

    // --- small helpers ---------------------------------------------------------

    void add(const expr& e) {
        if (opt) opt->add(e); else sol->add(e);
    }
    expr q(const Q& x) { return ctx.real_val(x.str().c_str()); }
    expr conj(std::initializer_list<expr> parts) {
        z3::expr_vector v(ctx);
        for (const expr& p : parts) v.push_back(p);
        return z3::mk_and(v);
    }
    expr disj(std::initializer_list<expr> parts) {
        z3::expr_vector v(ctx);
        for (const expr& p : parts) v.push_back(p);
        return z3::mk_or(v);
    }
    expr any(const std::vector<expr>& parts) {
        z3::expr_vector v(ctx);
        for (const expr& p : parts) v.push_back(p);
        return z3::mk_or(v);
    }
    expr all(const std::vector<expr>& parts) {
        z3::expr_vector v(ctx);
        v.push_back(ctx.bool_val(true));
        for (const expr& p : parts) v.push_back(p);
        return z3::mk_and(v);
    }
    expr sum(const std::vector<expr>& parts) {
        z3::expr_vector v(ctx);
        for (const expr& p : parts) v.push_back(p);
        return z3::sum(v);
    }

    // Comparisons against a constant are written constant-first (`6 <= t`, not
    // `t >= 6`), as the Python encoding wrote them. The two mean the same, but
    // when an objective's best value is approached and never reached (strict
    // ordering makes that common), Z3 returns one of many near-best models, and
    // which one depends on how the formula is spelled. Keeping the spelling
    // keeps the cards people have already seen.
    static bool flip(const expr& a, const expr& b) { return b.is_numeral() && !a.is_numeral(); }
    expr ge(const expr& a, const expr& b) { return flip(a, b) ? b <= a : a >= b; }
    expr le(const expr& a, const expr& b) { return flip(a, b) ? b >= a : a <= b; }
    expr gt(const expr& a, const expr& b) { return flip(a, b) ? b < a : a > b; }
    expr ne(const expr& a, const expr& b) { return flip(a, b) ? b != a : a != b; }

    Product& product(const char* name) {
        for (Product& p : products)
            if (p.name == name) return p;
        throw std::runtime_error(std::string("unknown product '") + name + "'");
    }
    std::vector<Dose> doses(const Product& p) {
        std::vector<Dose> out;
        for (size_t k = 0; k < p.t.size(); ++k) out.push_back({p.t[k], p.on[k]});
        return out;
    }
    expr count(const Product& p) {
        if (p.on.empty()) return ctx.int_val(0);
        std::vector<expr> ones;
        for (const expr& o : p.on) {
            expr one = ctx.int_val(1);
            expr zero = ctx.int_val(0);
            ones.push_back(z3::ite(o, one, zero));
        }
        return sum(ones);
    }
    // id == nullptr asserts the formula outright (a hard, untracked constraint).
    void track(const char* id, const expr& formula) {
        if (!id) {
            add(formula);
            return;
        }
        if (tracked.count(id)) throw std::runtime_error(std::string("duplicate rule id '") + id + "'");
        expr lit = ctx.bool_const((std::string("track::") + id).c_str());
        tracked.emplace(id, lit);
        tracked_ids.push_back(id);
        add(z3::implies(lit, formula));
    }
    void track(const std::string& id, const expr& formula) { track(id.c_str(), formula); }

    // --- circular-time building blocks ------------------------------------------

    // The forward gap from a to b, ((b - a) mod 24) in [0, 24), lies in [lo, hi].
    // With diff = b - a in (-24, 24):
    //   forward = diff        when diff >= 0
    //   forward = diff + 24   when diff <  0   (strict: diff == 0 means forward == 0)
    expr forward_gap_within(const expr& a, const expr& b, const Q& lo, const Q& hi) {
        expr diff = b - a;
        expr same_day = diff >= 0;
        expr lo1 = q(lo);
        expr above = ge(diff, lo1);
        expr hi1 = q(hi);
        expr below = le(diff, hi1);
        expr direct = conj({same_day, above, below});
        expr next_day = diff < 0;
        expr wrapped = diff + 24;
        expr above_w = ge(wrapped, lo1);
        expr below_w = le(wrapped, hi1);
        expr around = conj({next_day, above_w, below_w});
        return disj({direct, around});
    }

    // (b - a) mod 24 as a term, for a, b in [0, 24).
    expr forward(const expr& a, const expr& b) {
        expr later = ge(b, a);
        expr gap = b - a;
        expr wrapped = gap + 24;
        return z3::ite(later, gap, wrapped);
    }

    // cdist(x, y) = min(|x - y|, 24 - |x - y|) >= g.
    expr circular_distance_at_least(const expr& x, const expr& y, const Q& g) {
        return forward_gap_within(x, y, g, DAY - g);
    }

    // t is at least `before` h before and `after` h after `anchor` (circular).
    expr outside_window_around(const expr& t, const expr& anchor, const Q& before, const Q& after) {
        if (before + after > DAY) return ctx.bool_val(false);
        expr f = forward_gap_within(t, anchor, before, DAY - after);
        if (!(after > ZERO)) return f;
        // A simultaneous dose is 0 h after the anchor; forward_gap_within alone
        // would accept that when before == 0.
        expr apart = ne(t, anchor);
        return conj({f, apart});
    }

    // Every pair of doses that are both taken is at least g hours apart.
    expr pairwise_apart(const std::vector<Dose>& ds, const Q& g) {
        std::vector<expr> parts;
        for (size_t i = 0; i < ds.size(); ++i)
            for (size_t j = i + 1; j < ds.size(); ++j) {
                expr both = conj({ds[i].on, ds[j].on});
                expr apart = circular_distance_at_least(ds[i].t, ds[j].t, g);
                parts.push_back(z3::implies(both, apart));
            }
        return all(parts);
    }

    std::vector<expr> meal_terms(const char* id, const char* meal) {
        std::vector<expr> out;
        if (!meal) {
            for (auto& m : meals) out.push_back(m.second);
            return out;
        }
        for (auto& m : meals)
            if (m.first == meal) return {m.second};
        throw std::runtime_error(std::string(id ? id : "(hard rule)") + ": no time given for meal '" + meal + "'");
    }

    void add_meal(const char* name, const expr& e) {
        for (auto& m : meals)
            if (m.first == name) throw std::runtime_error(std::string("duplicate meal '") + name + "'");
        meals.emplace_back(name, e);
    }

    // t lies in the awake arc [wake - early, bed + late] on the circular day.
    expr awake(const expr& t, const Q& wake, const Q& awake_len, const expr& early, const expr& late) {
        expr w = q(wake);
        expr since_wake = forward(w, t);
        expr len = q(awake_len);
        expr day_end = len + late;
        expr before_bed = le(since_wake, day_end);
        expr until_wake = forward(t, w);
        expr just_before_wake = le(until_wake, early);
        return disj({before_bed, just_before_wake});
    }

    z3::optimize& optimizer() {
        if (!opt) throw std::runtime_error("objectives need an encoding built with optimize=True");
        return *opt;
    }
    z3::model& last_model() {
        if (!model) throw std::runtime_error("no model: the last check was not sat");
        return *model;
    }
    // Exact value of a term in the last model, as "n" or "n/d".
    const char* numeral(const expr& e) {
        expr v = last_model().eval(e, true);
        if (!v.is_numeral()) throw std::runtime_error("non-rational model value " + v.to_string());
        if (v.is_real()) {
            // Read the way Z3's Python binding reads a rational: numerator, then
            // denominator, each a short-lived term. Not needed for the value;
            // it keeps Z3's term numbering, and so its choice among tied
            // models, in step with the Python encoding (see the note on Enc).
            { expr numerator = v.numerator(); }
            { expr denominator = v.denominator(); }
        }
        scratch = Z3_get_numeral_string(ctx, v);
        return scratch.c_str();
    }
};

template <class F>
int guard(F f) {
    try {
        return f();
    } catch (const z3::exception& e) {
        g_error = e.msg();
    } catch (const std::exception& e) {
        g_error = e.what();
    }
    return -1;
}

Enc& E(void* h) { return *static_cast<Enc*>(h); }

}  // namespace

// --- lifetime --------------------------------------------------------------------

LN_API const char* ln_error() { return g_error.c_str(); }

LN_API void* ln_new(int optimize, unsigned timeout_ms) {
    try {
        return new Enc(optimize != 0, timeout_ms);
    } catch (const z3::exception& e) {
        g_error = e.msg();
    } catch (const std::exception& e) {
        g_error = e.what();
    }
    return nullptr;
}

LN_API void ln_free(void* h) { delete static_cast<Enc*>(h); }

// --- structure: products, ingredients, meals -------------------------------------

// count: fixed dose count (the patient's reported usage) or -1 for none. Without
// it an as-needed count is free in [0, max_doses]. Prescribed products always
// take every slot.
LN_API int ln_product(void* h, const char* name, int max_doses, int prescribed, int count) {
    return guard([&] {
        Enc& e = E(h);
        for (const Product& p : e.products)
            if (p.name == name) throw std::runtime_error(std::string("duplicate product '") + name + "'");
        std::string n(name);
        if (prescribed && count >= 0 && count != max_doses)
            throw std::runtime_error(n + " is prescribed; its count is fixed at " + std::to_string(max_doses));
        Product p{n, prescribed != 0, {}, {}, {}};
        for (int k = 0; k < max_doses; ++k) p.t.push_back(e.ctx.real_const(("t::" + n + "::" + std::to_string(k)).c_str()));
        for (int k = 0; k < max_doses; ++k) p.on.push_back(e.ctx.bool_const(("on::" + n + "::" + std::to_string(k)).c_str()));
        for (int k = 0; k < max_doses; ++k) {
            expr from = p.t[k] >= 0;
            expr to = p.t[k] < 24;
            expr in_day = e.conj({from, to});
            e.add(z3::implies(p.on[k], in_day));
            expr off = !p.on[k];
            expr pinned = p.t[k] == 0;
            e.add(z3::implies(off, pinned));
        }
        for (int k = 0; k + 1 < max_doses; ++k) {
            e.add(z3::implies(p.on[k + 1], p.on[k]));  // activation monotonicity
            expr both = e.conj({p.on[k], p.on[k + 1]});
            expr ordered = p.t[k] < p.t[k + 1];
            e.add(z3::implies(both, ordered));  // ordering
        }
        if (p.prescribed) {
            expr taken = e.count(p);
            e.track("count::" + n, taken == max_doses);
        } else {
            expr taken = e.count(p);
            e.track("max::" + n, taken <= max_doses);
            if (count >= 0) e.track("reported::" + n, taken == count);
        }
        e.products.push_back(std::move(p));
        return 0;
    });
}

LN_API int ln_ingredient(void* h, const char* product, const char* ingredient, const char* amount) {
    return guard([&] {
        Enc& e = E(h);
        e.product(product).ingredients.emplace_back(ingredient, Q::parse(amount));
        return 0;
    });
}

LN_API int ln_meal_fixed(void* h, const char* name, const char* time) {
    return guard([&] {
        Enc& e = E(h);
        e.add_meal(name, e.q(Q::parse(time)));
        return 0;
    });
}

// The meal time is a solver variable (tier 1 may move it).
LN_API int ln_meal_var(void* h, const char* name) {
    return guard([&] {
        Enc& e = E(h);
        e.add_meal(name, e.ctx.real_const((std::string("meal::") + name).c_str()));
        return 0;
    });
}

// --- label rules. id == NULL makes the rule a hard, untracked constraint ---------

// Family 1: any two doses of `product` are at least `hours` apart.
LN_API int ln_min_interval(void* h, const char* id, const char* product, const char* hours) {
    return guard([&] {
        Enc& e = E(h);
        expr formula = e.pairwise_apart(e.doses(e.product(product)), Q::parse(hours));
        e.track(id, formula);
        return 0;
    });
}

// Family 1 across products: any two doses containing `ingredient`.
LN_API int ln_ingredient_interval(void* h, const char* id, const char* ingredient, const char* hours) {
    return guard([&] {
        Enc& e = E(h);
        std::vector<Dose> ds;
        for (const Product& p : e.products)
            for (auto& ing : p.ingredients)
                if (ing.first == ingredient) {
                    for (const Dose& d : e.doses(p)) ds.push_back(d);
                    break;
                }
        expr formula = e.pairwise_apart(ds, Q::parse(hours));
        e.track(id, formula);
        return 0;
    });
}

// Family 3: each dose of `a` at least `before` h before and `after` h after every dose of `b`.
LN_API int ln_separation(void* h, const char* id, const char* a, const char* b, const char* before, const char* after) {
    return guard([&] {
        Enc& e = E(h);
        const Product& pa = e.product(a);
        const Product& pb = e.product(b);
        Q bef = Q::parse(before), aft = Q::parse(after);
        if (pa.on.empty() || pb.on.empty()) {
            e.track(id, e.ctx.bool_val(true));
        } else if (bef + aft > DAY) {
            // No placement satisfies both gaps: the two products cannot share a day.
            expr a_taken = e.any(pa.on);
            expr b_taken = e.any(pb.on);
            expr both = e.conj({a_taken, b_taken});
            e.track(id, !both);
        } else {
            std::vector<expr> parts;
            for (const Dose& da : e.doses(pa))
                for (const Dose& db : e.doses(pb)) {
                    expr both = e.conj({da.on, db.on});
                    expr clear = e.outside_window_around(da.t, db.t, bef, aft);
                    parts.push_back(z3::implies(both, clear));
                }
            expr formula = e.all(parts);
            e.track(id, formula);
        }
        return 0;
    });
}

// Family 4: total daily amount of `ingredient`, across all products.
LN_API int ln_ingredient_ceiling(void* h, const char* id, const char* ingredient, const char* max_amount) {
    return guard([&] {
        Enc& e = E(h);
        std::vector<expr> amounts;
        for (const Product& p : e.products)
            for (auto& ing : p.ingredients)
                if (ing.first == ingredient) {
                    for (const expr& o : p.on) {
                        expr amount = e.q(ing.second);
                        expr none = e.q(ZERO);
                        amounts.push_back(z3::ite(o, amount, none));
                    }
                    break;
                }
        if (amounts.empty()) {
            e.track(id, e.ctx.bool_val(true));
            return 0;
        }
        expr total = e.sum(amounts);
        expr ceiling = e.q(Q::parse(max_amount));
        e.track(id, e.le(total, ceiling));
        return 0;
    });
}

// Doses of `product` fall inside one of n spans. A span with start > end wraps midnight.
LN_API int ln_window(void* h, const char* id, const char* product, const char** starts, const char** ends, int n) {
    return guard([&] {
        Enc& e = E(h);
        std::vector<expr> parts;
        for (const Dose& d : e.doses(e.product(product))) {
            std::vector<expr> spans;
            for (int i = 0; i < n; ++i) {
                Q s = Q::parse(starts[i]), en = Q::parse(ends[i]);
                expr from = e.q(s);
                expr after_start = e.ge(d.t, from);
                expr to = e.q(en);
                expr before_end = e.le(d.t, to);
                spans.push_back(s <= en ? e.conj({after_start, before_end}) : e.disj({after_start, before_end}));
            }
            expr inside = e.any(spans);
            parts.push_back(z3::implies(d.on, inside));
        }
        expr formula = e.all(parts);
        e.track(id, formula);
        return 0;
    });
}

// Family 2: no dose of `product` from `before` h before to `after` h after a
// meal: the named one, or every meal if `meal` is NULL.
LN_API int ln_meal_gap(void* h, const char* id, const char* product, const char* before, const char* after, const char* meal) {
    return guard([&] {
        Enc& e = E(h);
        std::vector<expr> ms = e.meal_terms(id, meal);
        Q bef = Q::parse(before), aft = Q::parse(after);
        std::vector<expr> parts;
        for (const Dose& d : e.doses(e.product(product)))
            for (const expr& m : ms) {
                expr clear = e.outside_window_around(d.t, m, bef, aft);
                parts.push_back(z3::implies(d.on, clear));
            }
        expr formula = e.all(parts);
        e.track(id, formula);
        return 0;
    });
}

// Family 2: each dose of `product` falls between `earliest` and `latest` hours before `meal`.
LN_API int ln_meal_window(void* h, const char* id, const char* product, const char* meal, const char* earliest, const char* latest) {
    return guard([&] {
        Enc& e = E(h);
        if (!meal) throw std::runtime_error("a meal window needs a named meal");
        expr m = e.meal_terms(id, meal)[0];
        Q lo = Q::parse(latest), hi = Q::parse(earliest);
        std::vector<expr> parts;
        for (const Dose& d : e.doses(e.product(product))) {
            expr inside = e.forward_gap_within(d.t, m, lo, hi);
            parts.push_back(z3::implies(d.on, inside));
        }
        expr formula = e.all(parts);
        e.track(id, formula);
        return 0;
    });
}

// --- routine and tier-1 terms ------------------------------------------------------

// Every dose falls while the patient is awake: tracked per product as
// `routine::sleep@<product>`. With max_shift == NULL the sleep window is fixed.
// Otherwise waking earlier / going to bed later become variables bounded by
// max_shift and costing cost_per_hour (readable as `sleep::wake_earlier`,
// `sleep::bed_later`).
LN_API int ln_sleep(void* h, const char* wake, const char* awake_len, const char* max_shift, const char* cost_per_hour) {
    return guard([&] {
        Enc& e = E(h);
        if ((max_shift == nullptr) != (cost_per_hour == nullptr))
            throw std::runtime_error("a movable sleep window needs both a limit and a cost");
        Q w = Q::parse(wake), len = Q::parse(awake_len);
        std::optional<expr> early, late;
        if (max_shift) {
            early = e.ctx.real_const("sleep::wake_earlier");
            late = e.ctx.real_const("sleep::bed_later");
            expr early_min = *early >= 0;
            expr late_min = *late >= 0;
            expr mx = e.q(Q::parse(max_shift));
            expr early_max = e.le(*early, mx);
            expr late_max = e.le(*late, mx);
            e.add(early_min);
            e.add(late_min);
            e.add(early_max);
            e.add(late_max);
        } else {
            early = e.q(ZERO);
            late = e.q(ZERO);
        }
        for (const Product& p : e.products) {
            std::vector<expr> parts;
            for (const Dose& d : e.doses(p)) {
                expr up = e.awake(d.t, w, len, *early, *late);
                parts.push_back(z3::implies(d.on, up));
            }
            expr formula = e.all(parts);
            e.track("routine::sleep@" + p.name, formula);
        }
        if (max_shift) {
            expr rate = e.q(Q::parse(cost_per_hour));
            expr shift = *early + *late;
            e.costs.push_back(rate * shift);
            e.components.push_back({'s', "sleep::wake_earlier", *early, ""});
            e.components.push_back({'s', "sleep::bed_later", *late, ""});
        }
        return 0;
    });
}

// A variable meal may move at most max_shift hours from `original`, at
// cost_per_hour. The size of the move is readable as `dev::<meal>`.
LN_API int ln_meal_shift(void* h, const char* meal, const char* original, const char* max_shift, const char* cost_per_hour) {
    return guard([&] {
        Enc& e = E(h);
        expr m = e.meal_terms(nullptr, meal)[0];
        expr dev = e.ctx.real_const((std::string("dev::") + meal).c_str());
        expr from = m >= 0;
        expr to = m < 24;
        expr was = e.q(Q::parse(original));
        expr later = m - was;
        expr covers_later = dev >= later;
        expr earlier = was - m;
        expr covers_earlier = dev >= earlier;
        expr limit = e.q(Q::parse(max_shift));
        expr bounded = e.le(dev, limit);
        e.add(from);
        e.add(to);
        e.add(covers_later);
        e.add(covers_earlier);
        e.add(bounded);
        expr rate = e.q(Q::parse(cost_per_hour));
        e.costs.push_back(rate * dev);
        e.components.push_back({'m', std::string("dev::") + meal, dev, ""});
        return 0;
    });
}

// Tier 1 may suggest fewer as-needed doses than reported, never more, at
// cost_per_dose each. Tracked as `reported_cap::<product>`; the number dropped
// is readable as `dropped::<product>`.
LN_API int ln_reported_cap(void* h, const char* product, int reported, const char* cost_per_dose) {
    return guard([&] {
        Enc& e = E(h);
        expr taken = e.count(e.product(product));
        expr cnt = z3::to_real(taken);
        expr capped = cnt <= reported;
        e.track(std::string("reported_cap::") + product, capped);
        expr was = e.ctx.real_val(reported);
        expr dropped = was - cnt;
        expr rate = e.q(Q::parse(cost_per_dose));
        e.costs.push_back(rate * dropped);
        e.components.push_back({'d', std::string("dropped::") + product, dropped, ""});
        return 0;
    });
}

// --- objectives (lexicographic, in the order added) ---------------------------------

// Total cost of the tier-1 changes declared so far.
LN_API int ln_minimize_cost(void* h) {
    return guard([&] {
        Enc& e = E(h);
        z3::optimize& o = e.optimizer();
        if (e.costs.empty()) e.costs.push_back(e.q(ZERO));
        expr total = e.sum(e.costs);
        o.minimize(total);
        e.cost = total;
        return 0;
    });
}

// Earliest valid dose times. Many schedules tie; this picks one
// deterministically so a re-run shows the same card.
LN_API int ln_minimize_times(void* h) {
    return guard([&] {
        Enc& e = E(h);
        z3::optimize& o = e.optimizer();
        std::vector<expr> ts;
        for (const Product& p : e.products)
            for (const Dose& d : e.doses(p)) {
                expr unused = e.q(ZERO);
                ts.push_back(z3::ite(d.on, d.t, unused));
            }
        if (ts.empty()) ts.push_back(e.q(ZERO));
        o.minimize(e.sum(ts));
        return 0;
    });
}

LN_API int ln_maximize_count(void* h, const char* product) {
    return guard([&] {
        Enc& e = E(h);
        e.optimizer().maximize(e.count(e.product(product)));
        return 0;
    });
}

// After a sat check on a tier-1 encoding: read how much each change moved
// (ln_change_value) and the total cost (ln_cost_value), then require the next
// model to leave at least one of this model's changes undone, so it is a
// genuinely different fix, not this one plus a pointless tweak. Continuous
// shifts count as "also changed" if above eps; dose drops only if at least as
// many are dropped. Returns the number of changes blocked; 0 means the model
// changed nothing and no constraint was added.
LN_API int ln_block_superset(void* h, const char* eps) {
    return guard([&] {
        Enc& e = E(h);
        if (!e.cost) throw std::runtime_error("no tier-1 objective: call ln_minimize_cost first");
        e.last_model();
        e.previous_model.reset();
        e.blocked.clear();
        Q ep = Q::parse(eps);
        std::vector<std::optional<expr>> signature;
        for (Component& c : e.components) {
            c.value = e.numeral(c.term);
            Q v = Q::parse(c.value.c_str());
            if (c.kind == 'd') {
                expr dropped = e.q(v);
                signature.push_back(e.ge(c.term, dropped));
            } else if (v > ep) {
                expr floor = e.q(ep);
                signature.push_back(e.gt(c.term, floor));
            } else {
                signature.push_back(std::nullopt);
            }
        }
        e.cost_value = e.numeral(*e.cost);
        z3::expr_vector superset(e.ctx);
        for (size_t i = 0; i < e.components.size(); ++i) {
            if (!signature[i]) continue;
            e.blocked.push_back(*signature[i]);
            if (Q::parse(e.numeral(e.components[i].term)) > ep) superset.push_back(*signature[i]);
        }
        if (superset.empty()) return 0;
        expr repeat = z3::mk_and(superset);
        e.add(!repeat);
        return static_cast<int>(superset.size());
    });
}

LN_API int ln_change_count(void* h) { return static_cast<int>(E(h).components.size()); }
LN_API const char* ln_change_name(void* h, int i) { return E(h).components.at(i).name.c_str(); }
LN_API const char* ln_change_value(void* h, int i) { return E(h).components.at(i).value.c_str(); }
LN_API const char* ln_cost_value(void* h) { return E(h).cost_value.c_str(); }

// --- solving -------------------------------------------------------------------------

LN_API int ln_tracked_count(void* h) { return static_cast<int>(E(h).tracked_ids.size()); }
LN_API const char* ln_tracked_id(void* h, int i) { return E(h).tracked_ids.at(i).c_str(); }

// Check with exactly the n given rule ids enforced. 0 sat, 1 unsat, 2 unknown.
// Never fails on timeout: that is 2, with ln_reason().
LN_API int ln_check(void* h, const char** ids, int n) {
    return guard([&] {
        Enc& e = E(h);
        // The last model stays alive through this check, as it did in Python
        // (the old result was dropped only once the new one was in hand).
        e.previous_model = std::move(e.model);
        e.model.reset();
        e.core.clear();
        e.reason.clear();
        z3::expr_vector asms(e.ctx);
        for (int i = 0; i < n; ++i) {
            auto it = e.tracked.find(ids[i]);
            if (it == e.tracked.end()) throw std::runtime_error(std::string("unknown rule id '") + ids[i] + "'");
            asms.push_back(it->second);
        }
        z3::check_result r = e.opt ? e.opt->check(asms) : e.sol->check(asms);
        if (r == z3::sat) {
            e.model.emplace(e.opt ? e.opt->get_model() : e.sol->get_model());
            return 0;
        }
        if (r == z3::unsat) {
            z3::expr_vector c = e.opt ? e.opt->unsat_core() : e.sol->unsat_core();
            for (unsigned i = 0; i < c.size(); ++i) {
                std::string name = c[i].decl().name().str();
                e.core.push_back(name.rfind("track::", 0) == 0 ? name.substr(7) : name);
            }
            e.previous_model.reset();
            return 1;
        }
        e.previous_model.reset();
        e.reason = e.opt ? Z3_optimize_get_reason_unknown(e.ctx, *e.opt) : e.sol->reason_unknown();
        if (e.reason.empty()) e.reason = "unknown";
        return 2;
    });
}

LN_API int ln_core_size(void* h) { return static_cast<int>(E(h).core.size()); }
LN_API const char* ln_core_id(void* h, int i) { return E(h).core.at(i).c_str(); }
LN_API const char* ln_reason(void* h) { return E(h).reason.c_str(); }

// Slot k of `product` in the last model: 1 and *out = its time if the dose is
// taken, 0 if the slot is unused.
LN_API int ln_dose_time(void* h, const char* product, int k, const char** out) {
    return guard([&] {
        Enc& e = E(h);
        const Product& p = e.product(product);
        if (!e.last_model().eval(p.on.at(k), true).is_true()) return 0;
        *out = e.numeral(p.t.at(k));
        return 1;
    });
}

LN_API int ln_meal_count(void* h) { return static_cast<int>(E(h).meals.size()); }
LN_API const char* ln_meal_name(void* h, int i) { return E(h).meals.at(i).first.c_str(); }

// Time of meal i in the last model, as "n" or "n/d".
LN_API const char* ln_meal_value(void* h, int i) {
    try {
        Enc& e = E(h);
        return e.numeral(e.meals.at(i).second);
    } catch (const z3::exception& ex) {
        g_error = ex.msg();
    } catch (const std::exception& ex) {
        g_error = ex.what();
    }
    return nullptr;
}

// The whole encoding as SMT-LIB text, objectives included. For debugging.
LN_API const char* ln_sexpr(void* h) {
    try {
        Enc& e = E(h);
        e.scratch = e.opt ? Z3_optimize_to_string(e.ctx, *e.opt) : Z3_solver_to_string(e.ctx, *e.sol);
        return e.scratch.c_str();
    } catch (const z3::exception& ex) {
        g_error = ex.msg();
    } catch (const std::exception& ex) {
        g_error = ex.what();
    }
    return nullptr;
}
