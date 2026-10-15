import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, exp50_bayes_all as B
PATH="../data/panel.pkl"; Y._init(PATH)
pool=mp.get_context("fork").Pool(4,initializer=Y._init,initargs=(PATH,))
E=json.load(open("../system_G.json"))["cfg"]
PER={"choose 2012-16":(Y.starts("2012-01-01","2016-12-31",3),"2019-12-31"),"check 2000-08":(Y.starts("2000-01-01","2008-12-31",4),"2011-12-30"),"check 2020-25":(Y.starts("2020-01-01","2025-10-31",3),None)}
G=[("System G",E)]
G+=[(f"A. Paused: keep idle money in leading index while it is above its {n}d",dict(E,park_sma=n)) for n in (50,100,200)]
G+=[(f"B. Top {round((1-x)*100)}% by 6-month strength skip the ADX gate",dict(E,adx_exempt_rs=x)) for x in (0.8,0.9,0.95)]
G+=[(f"C. Profit lock trail {t:.0%} (lock at +50%)",dict(E,lock_trail=t)) for t in (0.30,0.35,0.40)]
G+=[(f"C. Profit lock from +{a:.0%} (trail 25%)",dict(E,lock_at=a)) for a in (0.75,1.0,9.0)]
rows=[]
for lab,c in G:
    r=B.evaluate(pool,c,PER); rows.append(dict(rule=lab,**{k:v for k,v in r.items() if k.endswith(("|beat","|ddok","|cagr"))})); print(lab,flush=True)
d=pd.DataFrame(rows); d.to_csv("../results/fixes_quick.csv",index=False)
pd.set_option("display.width",250); print(d.round(3).to_string(index=False))
