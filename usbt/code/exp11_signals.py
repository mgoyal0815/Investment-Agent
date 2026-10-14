"""Experiment 11: new signal families and volatility-managed exposure on the current best.
  rp_metrics: the two ranks inside RobustPair. v5: 6-month return vs SPY + distance above SMA200.
              New: residual momentum (12-1 month, net of SPY beta), nearness to the 52-week high.
  vol_target: cap the stock book at vol_target / its own 63-session realised volatility
              (Barroso & Santa-Clara 2015); vol_hard also trims holdings to the cap.
  size_mult:  1 or 1.5 x the v5 position budgets.
Every configuration is run on all three windows; the pick is the best runner.objective2 on
2012-2019 only. Writes results/exp11_all.csv and exp11_log.txt."""
import itertools, json
import pandas as pd
import runner as R
from exp2_search import materialise
from exp8_more import NEW

MET = [("rs126", "dist200"), ("resmom", "dist200"), ("hi52", "rs126"), ("resmom", "hi52"), ("rs126", "hi52")]
VOL = [(None, False)] + [(v, h) for v in (0.20, 0.25, 0.30, 0.40) for h in (False, True)]
if __name__ == "__main__":
    cfgs = []
    for met, (vt, vh), sm in itertools.product(MET, VOL, (1.0, 1.5)):
        lab = f"{met[0]}+{met[1]}, vol {vt or 'off'}{' hard' if vh else ''}, size x{sm}"
        cfgs.append((lab, dict(NEW, rp_metrics=met, vol_target=vt, vol_hard=vh, size_mult=sm)))
    rn = R.Runner()
    res = {}
    for per in ("train", "test", "early"):
        res[per] = rn.eval_many([materialise(c) for _, c in cfgs], per)
        print(per, "done", flush=True)
    rows = []
    for k, (lab, c) in enumerate(cfgs):
        r = {"label": lab, "cfg": json.dumps(c, default=list)}
        for per in ("train", "test", "early"):
            a = res[per][k]
            r.update({f"{per}_obj2": R.objective2(a), f"{per}_cagr": a["cagr"], f"{per}_dd": a["maxdd"], f"{per}_beat": a["beat_all"]})
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("train_obj2", ascending=False)
    df.to_csv("../results/exp11_all.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["label"] + [f"{p}_{x}" for p in ("train", "test", "early") for x in ("cagr", "dd", "beat")]
    print("TOP 12 BY TRAINING SCORE\n" + df[cols].head(12).round(3).to_string(index=False))
    print("CURRENT BEST\n" + df[df.label == "rs126+dist200, vol off, size x1.0"][cols].round(3).to_string(index=False))
