"""Phase 3 of exp43: Bayesian optimisation (Optuna TPE) of the enhancement mix on TRAIN starts only
(2012-2016, stop 2019), warm-started from the best factorial points; then the top 3 by TRAIN are judged on
2000-2008 starts, 2020-2025 starts and every monthly start 2000-2025 (to today)."""
import json, multiprocessing as mp, sys
import optuna
import pandas as pd
import yardstick as Y
import exp43_bayes as B

optuna.logging.set_verbosity(optuna.logging.WARNING)


def obj_train(pool, x):
    s = Y.summary(Y.evaluate(pool, B.build(x), *B.PER["train"]))
    return s["yrs_beat_SPY_QQQ"] - 0.5 * (1 - s["dd_ok"]), s


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    Y._init("../data/panel.pkl")
    pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=("../data/panel.pkl",))
    B.PER = {"train": (Y.starts("2012-01-01", "2016-12-31", 3), "2019-12-31")}
    keys = list(B.F)
    enc = {k: [json.dumps(v) for v in B.F[k]] for k in keys}
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=43, multivariate=True))
    d = pd.read_csv("../results/exp43_design.csv")
    d["obj"] = d["train|beat"] - 0.5 * (1 - d["train|ddok"])
    for _, r in d.sort_values("obj", ascending=False).head(15).iterrows():
        study.enqueue_trial({k: (r[f"f:{k}"] if r[f"f:{k}"] in enc[k] else enc[k][0]) for k in keys})
    log = []

    def objective(t):
        x = {k: json.loads(t.suggest_categorical(k, enc[k])) for k in keys}
        o, s = obj_train(pool, x)
        log.append(dict(trial=t.number, obj=o, beat=s["yrs_beat_SPY_QQQ"], ddok=s["dd_ok"], cagr=s["med_cagr"], x=json.dumps(x)))
        if t.number % 10 == 0:
            pd.DataFrame(log).to_csv("../results/exp43_optuna.csv", index=False)
            print(t.number, "best", round(study.best_value if study.best_trial else o, 3), flush=True)
        return o
    study.optimize(objective, n_trials=n)
    L = pd.DataFrame(log).sort_values("obj", ascending=False)
    L.to_csv("../results/exp43_optuna.csv", index=False)
    PER = {"train 2012-16": (Y.starts("2012-01-01", "2016-12-31", 2), "2019-12-31"),
           "check 2000-08": (Y.starts("2000-01-01", "2008-12-31", 2), "2011-12-30"),
           "check 2020-25": (Y.starts("2020-01-01", "2025-10-31", 1), None), "every start": (Y.starts(), None)}
    fin = [("System D", {k: v[0] for k, v in B.F.items()})]
    seen = set()
    for _, r in L.iterrows():
        if r.x not in seen:
            seen.add(r.x); fin.append((f"optuna #{len(fin)}", json.loads(r.x)))
        if len(fin) == 4:
            break
    rows = []
    for lab, x in fin:
        for p, (st, e) in PER.items():
            s = Y.summary(Y.evaluate(pool, B.build(x), st, e))
            rows.append(dict(label=lab, period=p, beat=s["yrs_beat_SPY_QQQ"], cal=s["cal_years_beat"], smh3=s["r3_beat_SMH"], ddok=s["dd_ok"],
                             cagr=s["med_cagr"], worst=s["worst_dd"], changes=json.dumps({k: v for k, v in x.items() if v != B.F[k][0]})))
        print("final", lab, flush=True)
    pd.DataFrame(rows).to_csv("../results/exp43_final.csv", index=False)
    print("PHASE3_DONE", flush=True)
