import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, exp50_bayes_all as B
PATH="../data/panel.pkl"; Y._init(PATH)
pool=mp.get_context("fork").Pool(4,initializer=Y._init,initargs=(PATH,))
E=json.load(open("../system_J.json"))["cfg"]
PER={"choose 2012-16":(Y.starts("2012-01-01","2016-12-31",3),"2019-12-31"),"check 2000-08":(Y.starts("2000-01-01","2008-12-31",4),"2011-12-30"),"check 2020-25":(Y.starts("2020-01-01","2025-10-31",3),None)}
G=[("System J",E)]
G+=[(f"A. {k} most-traded skip the ADX and OBV gates",dict(E,giant_gate_exempt=k)) for k in (10,20,50)]
G+=[(f"B. {k} most-traded get +{int(b*100)} rank points",dict(E,giant_bonus=(k,b))) for k,b in ((20,0.05),(20,0.10),(20,0.20),(50,0.10))]
G+=[(f"C. Ranking always tilted to the most-traded, weight {w}",dict(E,size_w=w)) for w in (0.25,0.5)]
G+=[("A+B. 20 most-traded skip gates and get +10 points",dict(E,giant_gate_exempt=20,giant_bonus=(20,0.10)))]
rows=[]
for lab,c in G:
    r=B.evaluate(pool,c,PER); rows.append(dict(rule=lab,**{k:v for k,v in r.items() if k.endswith(("|beat","|ddok","|cagr"))})); print(lab,flush=True)
d=pd.DataFrame(rows); d.to_csv("../results/giants2_quick.csv",index=False)
pd.set_option("display.width",250); print(d.round(3).to_string(index=False))
