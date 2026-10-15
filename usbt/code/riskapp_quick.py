import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, exp50_bayes_all as B
PATH="../data/panel.pkl"; Y._init(PATH)
pool=mp.get_context("fork").Pool(4,initializer=Y._init,initargs=(PATH,))
E=json.load(open("../system_G.json"))["cfg"]
PER={"choose 2012-16":(Y.starts("2012-01-01","2016-12-31",3),"2019-12-31"),"check 2000-08":(Y.starts("2000-01-01","2008-12-31",4),"2011-12-30"),"check 2020-25":(Y.starts("2020-01-01","2025-10-31",3),None)}
PR=["breadth","hyg","large"]; R={"dp":"discretionary/staples","cd":"cyclicals/defensives","hblv":"high-beta/low-vol"}
G=[("System G",E)]
G+=[(f"Pause vote: {R[r]} below {n}d",dict(E,pause_rules=PR+[f"ra_{r}{n}"])) for r in R for n in (50,100,200)]
G+=[(f"Restart: {R[r]} crosses above {n}d",dict(E,unpause_on=["coppock",f"ra_{r}{n}"])) for r in R for n in (50,100)]
G+=[(f"Veto pause: {R[r]} above {n}d and rising",dict(E,ra_veto=f"{r}{n}")) for r in R for n in (50,100,200)]
rows=[]
for lab,c in G:
    r=B.evaluate(pool,c,PER); rows.append(dict(rule=lab,**{k:v for k,v in r.items() if k.endswith(("|beat","|ddok","|cagr"))})); print(lab,flush=True)
d=pd.DataFrame(rows); d.to_csv("../results/riskapp_quick.csv",index=False)
pd.set_option("display.width",250); print(d.round(3).to_string(index=False))
