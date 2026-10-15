import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, exp50_bayes_all as B
PATH="../data/panel.pkl"; Y._init(PATH)
pool=mp.get_context("fork").Pool(4,initializer=Y._init,initargs=(PATH,))
E=json.load(open("../system_J.json"))["cfg"]
PER={"choose 2012-16":(Y.starts("2012-01-01","2016-12-31",3),"2019-12-31"),"check 2000-08":(Y.starts("2000-01-01","2008-12-31",4),"2011-12-30"),"check 2020-25":(Y.starts("2020-01-01","2025-10-31",3),None)}
G=[("System J",E)]
NR=dict(pause_dip_index_rsi=None)
G+=[(f"Steep fall instead of RSI: SPY {int(p*100)}% below its {n}-day high",dict(E,dip_fall=(p,n),**NR)) for p,n in ((0.10,10),(0.10,20),(0.15,20),(0.20,40))]
G+=[(f"RSI 25 or steep fall: SPY {int(p*100)}% below its {n}-day high",dict(E,dip_fall=(p,n))) for p,n in ((0.10,10),(0.15,20))]
G+=[(f"VIX spike instead of RSI: VIX peaked {v}+ and is {int(f*100)}% off its peak",dict(E,dip_vix=(v,f),**NR)) for v,f in ((30,0.10),(35,0.15),(40,0.15))]
G+=[("RSI 25 or VIX spike (35+, 15% off peak)",dict(E,dip_vix=(35,0.15)))]
rows=[]
for lab,c in G:
    r=B.evaluate(pool,c,PER); rows.append(dict(rule=lab,**{k:v for k,v in r.items() if k.endswith(("|beat","|ddok","|cagr"))})); print(lab,flush=True)
d=pd.DataFrame(rows); d.to_csv("../results/panic2_quick.csv",index=False)
pd.set_option("display.width",250); print(d.round(3).to_string(index=False))
