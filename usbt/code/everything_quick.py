import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, exp50_bayes_all as B
PATH="../data/panel.pkl"; Y._init(PATH)
pool=mp.get_context("fork").Pool(4,initializer=Y._init,initargs=(PATH,))
E=json.load(open("../system_K.json"))["cfg"]
PER={"choose 2012-16":(Y.starts("2012-01-01","2016-12-31",3),"2019-12-31"),"check 2000-08":(Y.starts("2000-01-01","2008-12-31",4),"2011-12-30"),"check 2020-25":(Y.starts("2020-01-01","2025-10-31",3),None)}
G=[("System K",E)]
PR=["breadth","hyg","large"]
G+=[(f"1. Hold {n} stocks",dict(E,N=n)) for n in (8,12,15)]
G+=[(f"1. Sell below rank {b}x holdings",dict(E,buffer=b)) for b in (1.2,2.0)]
G+=[("1. Review monthly (one review)",dict(E,stagger=[-1])),("1. Review about weekly (4 staggered reviews)",dict(E,stagger=[-1,7,14,21]))]
G+=[(f"1. Ranking: {int(w*100)}% 6m strength / {int(100-w*100)}% distance above 200d",dict(E,robust_w=w)) for w in (0.3,0.7)]
G+=[(f"1. Ranking blended with {k} (weight 0.3)",dict(E,score_mix={k:0.3})) for k in ("mom12_1","mom63")]
G+=[(f"2. At most {k} holdings per industry",dict(E,max_per_sector=k)) for k in (2,3)]
G+=[(f"2. Skip names correlated above {c} with a holding",dict(E,corr_max=c)) for c in (0.7,0.8)]
G+=[("3. Buy only stocks that dipped over the last 5 days",dict(E,buy_dip5=True)),("3. Buy only while QQQ MACD is falling",dict(E,buy_qqq_macd_down=True))]
G+=[("3. Sell in May: May-Oct is a pause vote",dict(E,pause_rules=PR+["maytooct"])),("3. Midterm years Jan-Sep are a pause vote",dict(E,pause_rules=PR+["midterm"]))]
G+=[("4. Dollar rising (UUP above 63d) is a pause vote",dict(E,pause_rules=PR+["uupup63"])),("4. Commodities falling (DBC below 50d) is a pause vote",dict(E,pause_rules=PR+["dbc50"]))]
G+=[("4. Yield curve inverted is a pause vote",dict(E,pause_rules=PR+["curve_inv"])),("4. Yield curve un-inverting is a pause vote",dict(E,pause_rules=PR+["curve_uninv"]))]
rows=[]
for lab,c in G:
    r=B.evaluate(pool,c,PER); rows.append(dict(rule=lab,**{k:v for k,v in r.items() if k.endswith(("|beat","|ddok","|cagr"))})); print(lab,flush=True)
d=pd.DataFrame(rows); d.to_csv("../results/everything_quick.csv",index=False)
pd.set_option("display.width",250); print(d.round(3).to_string(index=False))
