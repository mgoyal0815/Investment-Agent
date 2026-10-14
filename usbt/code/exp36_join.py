"""Experiment 36: 'start any time' WITHOUT a cold start. The system runs continuously from 2000 using only
past data; an investor who starts on date s simply buys what the model portfolio holds that day and
follows it. So each start = the continuous curve from s onward (no 3-month ramp, no empty book).
Also: fresh starts with the ramp switched off. Same yardstick as yardstick.py."""
import json, multiprocessing as mp
import pandas as pd
import engine as X, yardstick as Y
from exp24_timing import mat

if __name__ == "__main__":
    Y._init("../data/panel.pkl")
    m = Y.MM
    out = []
    for name in ("system_C", "system_B"):
        cfg = json.load(open(f"../{name}.json"))["cfg"]
        cv = X.run(m, mat(cfg), start="2000-01-03")["curve"]
        T = pd.DataFrame([Y.judge(cv.loc[s:], m) for s in Y.starts()])
        out.append(Y.summary(T, f"{name}: join the running portfolio"))
        pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=("../data/panel.pkl",))
        T2 = Y.evaluate(pool, dict(cfg, ramp=0), Y.starts())
        out.append(Y.summary(T2, f"{name}: fresh start, no ramp"))
        pool.close()
    df = pd.DataFrame(out)
    df.to_csv("../results/exp36_join.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).T.to_string())
