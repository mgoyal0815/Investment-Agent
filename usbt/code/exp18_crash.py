"""Experiment 18: the crash bleed (Aug 2011, May 2010, May 2011): the pause stops buying but keeps
every holding, and the 88-session floor blocks their stops.
  mh_profit_only / mh_loser_hold  framework 19.8: a position below cost may exit after 5 or 21 sessions
  hard_stop      sell a position that closes this far below its cost: none / 15% / 20%
  pause_cut      while paused, sell positions more than pause_cut_loss below cost: off / 0% / 5%
All windows; pick by runner.objective2 on 2012-2019. Writes results/exp18_all.csv."""
import itertools, json
import pandas as pd
import runner as R
from exp2_search import materialise

A = json.load(open("../system_A.json"))["cfg"]
if __name__ == "__main__":
    cfgs = []
    for mh, hs, pc in itertools.product((None, 5, 21), (None, 0.15, 0.20), (None, 0.0, 0.05)):
        c = dict(A, hard_stop=hs)
        if mh:
            c.update(mh_profit_only=True, mh_loser_hold=mh)
        if pc is not None:
            c.update(pause_cut=True, pause_cut_loss=pc)
        cfgs.append((f"loser hold {mh or 'off'}, hard stop {hs or 'off'}, pause cut {('off' if pc is None else f'{pc:.0%}')}", c))
    rn = R.Runner()
    res = {p: rn.eval_many([materialise(c) for _, c in cfgs], p) for p in ("train", "test", "early")}
    rows = []
    for k, (lab, c) in enumerate(cfgs):
        r = {"label": lab, "cfg": json.dumps(c, default=list)}
        for p in res:
            a = res[p][k]
            r.update({f"{p}_obj2": R.objective2(a), f"{p}_cagr": a["cagr"], f"{p}_dd": a["maxdd"], f"{p}_beat": a["beat_all"]})
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("train_obj2", ascending=False)
    df.to_csv("../results/exp18_all.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["label"] + [f"{p}_{x}" for p in ("train", "test", "early") for x in ("cagr", "dd", "beat")]
    print(df[cols].round(3).to_string(index=False))
