"""One round of the iterative Bayesian protocol around any base system (owner, 15 Oct 2026).
  python3 round.py ../system_X.json TAG [N] -> results/TAG_*; candidate written to ../TAG_candidate.json
Phase 1: sparse random design (1-6 changes from the base) over every enhancement in exp50; Phase 2: Bayesian
ridge effects; Phase 3: Optuna on 2012-2016 starts only; the best non-base pick is the round's candidate.
The candidate is then put through accept.py against the base."""
import json, multiprocessing as mp, random, sys
import pandas as pd
import yardstick as Y
import exp50_bayes_all as B

PATH = "../data/panel_idx.pkl"
base_file, TAG = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 400
BASE = json.load(open(base_file))["cfg"]
B.SE = BASE
toks = set(BASE.get("pause_rules") or B.BASE_PAUSE)
B.BASE_PAUSE = tuple(t for t in (BASE.get("pause_rules") or B.BASE_PAUSE) if t not in B.TOK.values())


def base_level(k, vals):
    if k in B.TOK:
        return B.TOK[k] in toks
    v = BASE.get(k)
    if k == "score_mix" and isinstance(v, dict):
        v = next(iter(v))
    if k == "lock_half":
        v = v or None
    return v


F = {}
for k, vals in B.F.items():
    b = base_level(k, vals)
    F[k] = [b] + [v for v in vals if json.dumps(v) != json.dumps(b)]

if __name__ == "__main__":
    Y._init(PATH)
    pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    PER = {"train": (Y.starts("2012-01-01", "2016-12-31", 3), "2019-12-31"),
           "early": (Y.starts("2000-01-01", "2008-12-31", 4), "2011-12-30"),
           "late": (Y.starts("2020-01-01", "2025-10-31", 3), None)}
    rng = random.Random(hash(TAG) % 10 ** 6)
    base = {k: v[0] for k, v in F.items()}
    design = [dict(base)]
    for _ in range(n):
        x = dict(base)
        for k in rng.sample(list(F), rng.randint(1, 6)):
            x[k] = rng.choice(F[k][1:])
        design.append(x)
    rows = []
    for k, x in enumerate(design):
        rows.append({**{f"f:{a}": json.dumps(b) for a, b in x.items()}, **B.evaluate(pool, B.build(x), PER)})
        if (k + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv(f"../results/{TAG}_design.csv", index=False); print(k + 1, flush=True)
    pd.DataFrame(rows).to_csv(f"../results/{TAG}_design.csv", index=False)
    B.model(f"../results/{TAG}_design.csv", f"../results/{TAG}_effects.txt")
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    keys = list(F); enc = {k: [json.dumps(v) for v in F[k]] for k in keys}
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=7, multivariate=True))
    d = pd.read_csv(f"../results/{TAG}_design.csv", dtype={f"f:{k}": str for k in keys}, keep_default_na=False)
    d["obj"] = d["train|beat"] - 0.5 * (1 - d["train|ddok"])
    for _, r in d.sort_values("obj", ascending=False).head(15).iterrows():
        st.enqueue_trial({k: r[f"f:{k}"] for k in keys})
    log = []

    def objective(t):
        x = {k: json.loads(t.suggest_categorical(k, enc[k])) for k in keys}
        s = B.evaluate(pool, B.build(x), {"train": PER["train"]})
        o = B.score_row(s, "train"); log.append(dict(obj=o, x=json.dumps(x))); return o
    st.optimize(objective, n_trials=150)
    L = pd.DataFrame(log).sort_values("obj", ascending=False); L.to_csv(f"../results/{TAG}_optuna.csv", index=False)
    best = next(json.loads(r.x) for _, r in L.iterrows() if json.loads(r.x) != base)
    cand = B.build(best)
    json.dump(dict(name=f"{TAG} candidate", changes={k: v for k, v in best.items() if v != base[k]}, cfg=cand),
              open(f"../{TAG}_candidate.json", "w"), indent=1, default=list)
    print("CANDIDATE", json.dumps({k: v for k, v in best.items() if v != base[k]}), flush=True)
    import accept
    ok, checks, tab = accept.decide(pool, BASE, cand)
    tab.to_csv(f"../results/{TAG}_accept.csv"); print(checks, flush=True)
    print("ROUND_RESULT", "ACCEPT" if ok else "REJECT", flush=True)
