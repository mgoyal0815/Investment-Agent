import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, exp50_bayes_all as B
PATH="../data/panel_ndx.pkl"; Y._init(PATH)
pool=mp.get_context("fork").Pool(4,initializer=Y._init,initargs=(PATH,))
E=json.load(open("../system_L.json"))["cfg"]
PER={"choose 2012-16":(Y.starts("2012-01-01","2016-12-31",3),"2019-12-31"),"check 2000-08":(Y.starts("2000-01-01","2008-12-31",4),"2011-12-30"),"check 2020-25":(Y.starts("2020-01-01","2025-10-31",3),None)}
G=[("System L",E)]
G+=[(f"Hold {n} stocks",dict(E,N=n)) for n in (8,9,11)]
G+=[(f"ADX entry filter >= {a}",dict(E,adx_min=a)) for a in (20,30)]+[("No ADX entry filter",dict(E,adx_min=None))]
G+=[(f"OBV trend >= {v}",dict(E,obv_min=v)) for v in (-0.5,0.5)]
G+=[(f"Profit lock from +{int(a*100)}%, trail {int(t*100)}%",dict(E,lock_at=a,lock_trail=t)) for a,t in ((0.5,0.30),(0.75,0.25),(0.4,0.25))]
G+=[(f"Sell below rank {b}x holdings",dict(E,buffer=b)) for b in (1.3,1.7)]
G+=[(f"Cyclicals/defensives restart at {n}-day average",dict(E,unpause_on=["coppock",f"ra_cd{n}"])) for n in (125,)]
G+=[(f"Restart window {d} sessions",dict(E,unpause_days=d)) for d in (35,50)]
G+=[(f"Crash trigger RSI <= {r}",dict(E,pause_dip_index_rsi=r)) for r in (22,27)]
G+=[(f"Index hurdle margin {int(m*100)} pts",dict(E,index_hurdle_margin=m)) for m in (0.05,)]
rows=[]
for lab,c in G:
    r=B.evaluate(pool,c,PER); rows.append(dict(rule=lab,**{k:v for k,v in r.items() if k.endswith(("|beat","|ddok","|cagr"))})); print(lab,flush=True)
d=pd.DataFrame(rows); d.to_csv("../results/tuneL_quick.csv",index=False)
pd.set_option("display.width",250); print(d.round(3).to_string(index=False))
