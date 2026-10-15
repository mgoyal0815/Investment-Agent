import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, exp50_bayes_all as B
PATH="../data/panel.pkl"; Y._init(PATH)
pool=mp.get_context("fork").Pool(4,initializer=Y._init,initargs=(PATH,))
E=json.load(open("../system_G.json"))["cfg"]
PER={"choose 2012-16":(Y.starts("2012-01-01","2016-12-31",3),"2019-12-31"),"check 2000-08":(Y.starts("2000-01-01","2008-12-31",4),"2011-12-30"),"check 2020-25":(Y.starts("2020-01-01","2025-10-31",3),None)}
CN={"spy50":"SPY above 50d","b50_40":"40%+ of stocks above 50d","b50_50":"50%+ of stocks above 50d","mcc":"McClellan > 0","hblv50":"high-beta/low-vol above 50d","hblv100":"high-beta/low-vol above 100d","vixrel1":"VIX below its 50d average"}
R=lambda tok, **k: dict(E, unpause_on=["coppock", tok], **k)
G=[("System G",E)]
G+=[(f"Cyc/def crosses {n}d, no confirmation",R(f"ra_cd{n}")) for n in (75,100,150)]
G+=[(f"Cyc/def crosses 100d + {v}",R("ra_cd100",unpause_confirm=[k])) for k,v in CN.items()]
G+=[("Cyc/def crosses 100d, 42-day window",R("ra_cd100",unpause_days=42))]
G+=[(f"Disc/staples crosses 50d + {CN[k]}",R("ra_dp50",unpause_confirm=[k])) for k in ("spy50","mcc","b50_50")]
rows=[]
for lab,c in G:
    r=B.evaluate(pool,c,PER); rows.append(dict(rule=lab,**{k:v for k,v in r.items() if k.endswith(("|beat","|ddok","|cagr"))})); print(lab,flush=True)
d=pd.DataFrame(rows); d.to_csv("../results/restart2_quick.csv",index=False)
pd.set_option("display.width",250); print(d.round(3).to_string(index=False))
