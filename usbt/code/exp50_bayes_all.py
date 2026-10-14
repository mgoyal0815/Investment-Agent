"""Experiment 50 (owner, 15 Oct 2026): ONE Bayesian study of every enhancement tested so far, on top of the
latest system (System E, point-in-time signals), on the panel with index/sector ETF columns.

Phase 1  sparse random design: each configuration switches 1-6 enhancements away from System E (N_DESIGN).
         Each is scored, fresh $10,000, on TRAIN starts 2012-2016 (stop 2019) and CHECK starts 2000-2008
         (stop 2011) and 2020-2025 (to today). Target: share of years beating BOTH SPY and QQQ, minus 0.5 x
         share of starts failing the drawdown rule (worst fall smaller than SPY's and QQQ's).
Phase 2  Bayesian ridge on all main effects + pairwise interactions, fitted per period; an enhancement
         'matters' only if credible on TRAIN and the same sign on both CHECK periods.
Phase 3  Bayesian optimisation (Optuna TPE) on TRAIN only, warm-started from the best design points; the top 3
         plus System E are then judged on every monthly start 2000-2025 (to today), split by era."""
import itertools, json, multiprocessing as mp, random, subprocess, sys
import numpy as np
import pandas as pd
import yardstick as Y

PATH = "../data/panel_idx.pkl"
SE = json.load(open("../system_E.json"))["cfg"]
BASE_PAUSE = ("breadth", "hyg", "large")
# every enhancement tested in exp34-49, first level = System E's own setting
F = dict(
    # entries (exp42/43)
    rsi_max=[None, 75, 80], adx_min=[None, 20], macd_pos=[None, True], hi52_min=[None, 0.9], vol_max_pct=[None, 0.8],
    bo_recent=[None, 60], score_mix=[None, "resmom", "hi52", "mom12_1", "mom12_7"],
    # holding / exits (exp39/43/44)
    trend_hold=[True, None], lock_at=[0.5, 0.3, 1.0], lock_trail=[0.25, 0.2, 0.35], lock_trend=[None, True], lock_half=[None, True],
    rank_exempt_gain=[1.0, None, 0.5], rsi_sell_floor=[None, 40], chand_k=[None, 4], rank_ladder=[None, 2], buffer=[1.5, 2.0],
    rank_min_hold=[0, 42],
    # sizing (exp39)
    N=[10, 12, 8], add_cap=[3.0, 5.0], add_gain=[0.2, 0.1], weight=[None, "invvol"],
    # market / pause (exp41/43/45/46)
    pause_needs_bear=[None, True], p_credit=[False, True], p_vix30=[False, True], p_vixrel=[False, True], p_momcrash=[False, True],
    p_spymacd=[False, True], thrust_days=[None, 63], pause_tight_exit=[True, False], reenter_fast=[None, True],
    gold_slots=[4, 2, 0], ramp=[3, 0], trend=[True, False],
    # giants / exposure (exp48)
    index_hurdle=[None, True], etf_in_universe=[None, True], mkt_vol_target=[None, 0.15, 0.2], vol_hard=[None, True],
    narrow_size_w=[None, 0.5],
    # literature (exp49)
    sector_top_k=[None, 3, 5], sector_w=[None, 0.25], dm_guard=[None, True], vol_target=[None, 0.3])
TOK = dict(p_credit="credit", p_vix30="vix30", p_vixrel="vixrel1.3", p_momcrash="momcrash", p_spymacd="spymacd")


def build(x):
    c = dict(SE)
    for k, v in x.items():
        if k in TOK:
            continue
        if k == "score_mix":
            c[k] = None if v is None else {v: 0.25 if v in ("resmom", "hi52", "mom12_7") else 0.5}
        elif k == "lock_half":
            c[k] = v
            if v:
                c["lock_trail2"] = 0.4
        else:
            c[k] = v
    c["pause_rules"] = BASE_PAUSE + tuple(TOK[k] for k in TOK if x.get(k))
    return c


def score_row(s, p):
    return s[f"{p}|beat"] - 0.5 * (1 - s[f"{p}|ddok"])


def evaluate(pool, c, per):
    out = {}
    for p, (st, e) in per.items():
        s = Y.summary(Y.evaluate(pool, c, st, e))
        out.update({f"{p}|beat": s["yrs_beat_SPY_QQQ"], f"{p}|ddok": s["dd_ok"], f"{p}|cagr": s["med_cagr"], f"{p}|smh3": s["r3_beat_SMH"], f"{p}|worst": s["worst_dd"]})
    return out


def model(design_csv, out_txt):
    from sklearn.linear_model import BayesianRidge
    d = pd.read_csv(design_csv, dtype=str, keep_default_na=False)
    fac = [c for c in d.columns if c.startswith("f:")]
    for c in d.columns:
        if not c.startswith("f:"):
            d[c] = d[c].astype(float)
    X = pd.get_dummies(d[fac], drop_first=False)
    base = {c: d[c].iloc[0] for c in fac}
    X = X[[c for c in X.columns if not any(c == f"{f}_{base[f]}" for f in fac)]].astype(float)
    X = X.loc[:, X.sum() >= 5]
    inter = {f"{a} x {b}": X[a] * X[b] for a, b in itertools.combinations(X.columns, 2)
             if a.split("_")[0] != b.split("_")[0] and (X[a] * X[b]).sum() >= 12}
    XI = pd.concat([X, pd.DataFrame(inter)], axis=1)
    res = {}
    for p in ("train", "early", "late"):
        y = d[f"{p}|beat"] - 0.5 * (1 - d[f"{p}|ddok"])
        mdl = BayesianRidge().fit(XI.values, y.values)
        res[p] = pd.DataFrame({"effect": mdl.coef_, "sd": np.sqrt(np.diag(mdl.sigma_))}, index=XI.columns)
    T = res["train"].join(res["early"], rsuffix="_early").join(res["late"], rsuffix="_late")
    T["credible_train"] = T.effect.abs() > 2 * T.sd
    T["same_sign_all"] = (np.sign(T.effect) == np.sign(T.effect_early)) & (np.sign(T.effect) == np.sign(T.effect_late))
    T["matters"] = T.credible_train & T.same_sign_all & ((T.effect_early.abs() > T.sd_early) | (T.effect_late.abs() > T.sd_late))
    T = T.sort_values("effect", ascending=False)
    T.to_csv(out_txt.replace(".txt", ".csv"))
    with open(out_txt, "w") as f:
        cols = ["effect", "sd", "effect_early", "effect_late", "credible_train", "same_sign_all", "matters"]
        f.write("MAIN EFFECTS\n" + T.loc[[i for i in T.index if " x " not in i], cols].round(3).to_string() + "\n\nINTERACTIONS THAT MATTER\n")
        f.write(T.loc[[i for i in T.index if " x " in i and T.loc[i, "matters"]], cols].round(3).to_string() + "\n")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    Y._init(PATH)
    pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    PER = {"train": (Y.starts("2012-01-01", "2016-12-31", 3), "2019-12-31"),
           "early": (Y.starts("2000-01-01", "2008-12-31", 4), "2011-12-30"),
           "late": (Y.starts("2020-01-01", "2025-10-31", 3), None)}
    rng = random.Random(50)
    base = {k: v[0] for k, v in F.items()}
    design = [dict(base)]
    for _ in range(n):
        x = dict(base)
        for k in rng.sample(list(F), rng.randint(1, 6)):
            x[k] = rng.choice(F[k][1:])
        design.append(x)
    rows = []
    for k, x in enumerate(design):
        r = evaluate(pool, build(x), PER)
        rows.append({**{f"f:{a}": json.dumps(b) for a, b in x.items()}, **r})
        if (k + 1) % 10 == 0:
            pd.DataFrame(rows).to_csv("../results/exp50_design.csv", index=False)
            print(k + 1, flush=True)
        if (k + 1) % 150 == 0:
            model("../results/exp50_design.csv", f"../results/exp50_effects_at_{k + 1}.txt")
            print("INTERIM", k + 1, flush=True)
    pd.DataFrame(rows).to_csv("../results/exp50_design.csv", index=False)
    model("../results/exp50_design.csv", "../results/exp50_effects_final.txt")
    print("PHASE1_DONE", flush=True)
    # ---- phase 3: Optuna TPE on TRAIN only
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    keys = list(F); enc = {k: [json.dumps(v) for v in F[k]] for k in keys}
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=50, multivariate=True))
    d = pd.read_csv("../results/exp50_design.csv", dtype={f"f:{k}": str for k in keys}, keep_default_na=False)
    d["obj"] = d["train|beat"] - 0.5 * (1 - d["train|ddok"])
    for _, r in d.sort_values("obj", ascending=False).head(15).iterrows():
        st.enqueue_trial({k: r[f"f:{k}"] for k in keys})
    log = []
    TR = {"train": PER["train"]}

    def objective(t):
        x = {k: json.loads(t.suggest_categorical(k, enc[k])) for k in keys}
        s = evaluate(pool, build(x), TR)
        o = score_row(s, "train")
        log.append(dict(obj=o, beat=s["train|beat"], ddok=s["train|ddok"], cagr=s["train|cagr"], x=json.dumps(x)))
        return o
    st.optimize(objective, n_trials=200)
    L = pd.DataFrame(log).sort_values("obj", ascending=False); L.to_csv("../results/exp50_optuna.csv", index=False)
    print("PHASE3_SEARCH_DONE", flush=True)
    fin, seen = [("System E (latest)", dict(base))], set()
    for _, r in L.iterrows():
        if r.x not in seen and json.loads(r.x) != base:
            seen.add(r.x); fin.append((f"Bayesian pick {len(fin)}", json.loads(r.x)))
        if len(fin) == 4:
            break
    rows = []
    for lab, x in fin:
        T = Y.evaluate(pool, build(x), Y.starts())
        rr = {"label": lab, "changes": json.dumps({k: v for k, v in x.items() if v != base[k]})}
        for era, (a, b) in {"choose 2012-16": (2012, 2016), "2000-11": (2000, 2011), "2017-19": (2017, 2019), "2020-25": (2020, 2025), "all": (2000, 2025)}.items():
            s = Y.summary(T[(T.start.dt.year >= a) & (T.start.dt.year <= b)])
            rr.update({f"{era}|beat": s["yrs_beat_SPY_QQQ"], f"{era}|ddok": s["dd_ok"], f"{era}|cagr": s["med_cagr"]})
        s = Y.summary(T); rr.update(cal=s["cal_years_beat"], smh3=s["r3_beat_SMH"], worst=s["worst_dd"])
        rows.append(rr); print("final", lab, flush=True)
    pd.DataFrame(rows).to_csv("../results/exp50_final.csv", index=False)
    print("PHASE3_DONE", flush=True)
