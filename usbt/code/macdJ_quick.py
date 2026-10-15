import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, exp50_bayes_all as B
PATH="../data/panel.pkl"; Y._init(PATH)
pool=mp.get_context("fork").Pool(4,initializer=Y._init,initargs=(PATH,))
E=json.load(open("../system_J.json"))["cfg"]
PER={"choose 2012-16":(Y.starts("2012-01-01","2016-12-31",3),"2019-12-31"),"check 2000-08":(Y.starts("2000-01-01","2008-12-31",4),"2011-12-30"),"check 2020-25":(Y.starts("2020-01-01","2025-10-31",3),None)}
G=[("System J",E)]
PR=["breadth","hyg","large"]
G+=[("Market: SPY MACD down-cross is an extra pause vote",dict(E,pause_rules=PR+["spymacd"]))]
G+=[("Market: SPY MACD up-cross restarts buying (42 sessions)",dict(E,unpause_on=list(E["unpause_on"])+["spymacdup"]))]
G+=[("Market: QQQ MACD up-cross restarts buying",dict(E,unpause_on=list(E["unpause_on"])+["qqqmacdup"]))]
G+=[("Market: SPY MACD pause vote + up-cross restart",dict(E,pause_rules=PR+["spymacd"],unpause_on=list(E["unpause_on"])+["spymacdup"]))]
G+=[(f"Stock: sell on MACD down-cross (gain >= {int(g*100)}%), rebuy only with MACD up",dict(E,macd_exit=0.0,macd_exit_min_gain=g,macd_pos=True)) for g in (0.0,0.1,0.3)]
G+=[("Stock: sell half on MACD down-cross, rebuy with MACD up",dict(E,macd_exit=0.0,macd_exit_min_gain=0.0,macd_exit_frac=0.5,macd_pos=True))]
rows=[]
for lab,c in G:
    r=B.evaluate(pool,c,PER); rows.append(dict(rule=lab,**{k:v for k,v in r.items() if k.endswith(("|beat","|ddok","|cagr"))})); print(lab,flush=True)
d=pd.DataFrame(rows); d.to_csv("../results/macdJ_quick.csv",index=False)
pd.set_option("display.width",250); print(d.round(3).to_string(index=False))
