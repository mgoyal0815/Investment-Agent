"""Builds ../page/system_e.html (close-out page) from results/final_E_* and the experiment summaries."""
import json, math
import pandas as pd

R = "../results/"
cur = pd.read_csv(R + "final_E_curves.csv", index_col=0, parse_dates=True)
yrs = pd.read_csv(R + "final_E_years.csv", index_col=0)
sc = pd.read_csv(R + "final_E_scorecard.csv")
sm = json.load(open(R + "final_E_summary.json"))
SER = [("System E", "e"), ("live v5", "v5"), ("SPY", "spy"), ("QQQ", "qqq"), ("SMH", "smh")]

# ---- growth chart (log scale), weekly points
W, H, L, Rt, T, B = 920, 400, 64, 118, 18, 34
c = cur.resample("W-FRI").last()
lo, hi = 1500, 1_200_000
x0, x1 = c.index[0], c.index[-1]
fx = lambda t: L + (t - x0).days / (x1 - x0).days * (W - L - Rt)
fy = lambda v: T + (math.log10(hi) - math.log10(v)) / (math.log10(hi) - math.log10(lo)) * (H - T - B)
g = []
for v, lab in [(2000, "$2k"), (5000, "$5k"), (10000, "$10k"), (20000, "$20k"), (50000, "$50k"), (100000, "$100k"), (200000, "$200k"), (500000, "$500k"), (1000000, "$1M")]:
    y = fy(v); g.append(f'<line class="grid" x1="{L}" x2="{W-Rt}" y1="{y:.1f}" y2="{y:.1f}"/><text class="ax" x="{L-8}" y="{y+4:.1f}" text-anchor="end">{lab}</text>')
for yr in range(2000, 2027, 2):
    x = fx(pd.Timestamp(f"{yr}-01-01")); g.append(f'<text class="ax" x="{x:.1f}" y="{H-10}" text-anchor="middle">{yr}</text>')
paths, ends = [], []
for name, cls in SER[::-1]:
    s = c[name].dropna()
    pts = " ".join(f"{fx(t):.1f},{fy(max(v, lo)):.1f}" for t, v in s.items())
    paths.append(f'<polyline class="ln {cls}" points="{pts}"/>')
    ends.append((fy(s.iloc[-1]), name, cls, s.iloc[-1]))
ends.sort()
lab, last = [], -99
for y, name, cls, v in ends:
    y = max(y, last + 15); last = y
    lab.append(f'<text class="end {cls}" x="{W-Rt+8}" y="{y+4:.1f}">{name} ${v/1000:,.0f}k</text>')
chart = f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Growth of $10,000 from January 2000, log scale">{"".join(g)}{"".join(paths)}{"".join(lab)}</svg>'

# ---- year table
rows = []
full = yrs.loc[[y for y in yrs.index if y <= 2025]]
wins = 0
for y, r in yrs.iterrows():
    best = max(r["SPY"], r["QQQ"])
    win = r["System E"] > best
    if y <= 2025:
        wins += win
    cell = lambda v, k: f'<td class="num{" hi" if k == "System E" else ""}{" neg" if v < 0 else ""}">{v:+.1f}</td>'
    tag = '<span class="pill w">beat both</span>' if win else '<span class="pill l">behind</span>'
    yl = f"{y}" + (" <small>to 12 Oct</small>" if y == 2026 else "")
    rows.append(f'<tr><th scope="row">{yl}</th>{cell(r["System E"],"System E")}{cell(r["SPY"],"")}{cell(r["QQQ"],"")}{cell(r["SMH"],"")}{cell(r["live v5"],"")}<td>{tag}</td></tr>')
year_table = "\n".join(rows)

# ---- scorecard
def pct(v): return f"{v*100:.0f}%"
sc_rows = []
for era in ("every start", "starts 2000-11", "starts 2012-19", "starts 2020-25"):
    e = sc[(sc.system == "System E") & (sc.era == era)].iloc[0]; v = sc[(sc.system == "live v5") & (sc.era == era)].iloc[0]
    sc_rows.append(f'<tr><th scope="row">{era.replace("starts ", "Starts ").replace("every start", "Every monthly start")}</th>'
                   f'<td class="num hi">{pct(e.beat)}</td><td class="num">{pct(v.beat)}</td><td class="num hi">{pct(e.cagr)}</td><td class="num">{pct(v.cagr)}</td>'
                   f'<td class="num hi">{pct(e.ddok)}</td><td class="num hi">{e.worst*100:.0f}%</td><td class="num hi">{pct(e.smh3)}</td></tr>')
sc_table = "\n".join(sc_rows)
E, S, Q, M, V = (sm[k] for k in ("System E", "SPY", "QQQ", "SMH", "live v5"))
every = sc[(sc.system == "System E") & (sc.era == "every start")].iloc[0]
html = open("page_template_E.html").read()
for k, v in {"{{CHART}}": chart, "{{YEARS}}": year_table, "{{SCORE}}": sc_table, "{{WINS}}": str(int(wins)),
             "{{E_CAGR}}": f"{E['cagr']*100:.1f}%", "{{E_END}}": f"${E['end']/1000:,.0f}k", "{{E_DD}}": f"{E['maxdd']*100:.0f}%",
             "{{S_CAGR}}": f"{S['cagr']*100:.1f}%", "{{Q_CAGR}}": f"{Q['cagr']*100:.1f}%", "{{S_DD}}": f"{S['maxdd']*100:.0f}%", "{{Q_DD}}": f"{Q['maxdd']*100:.0f}%",
             "{{S_END}}": f"${S['end']/1000:,.0f}k", "{{Q_END}}": f"${Q['end']/1000:,.0f}k", "{{V_CAGR}}": f"{V['cagr']*100:.1f}%", "{{V_END}}": f"${V['end']/1000:,.0f}k",
             "{{BEAT}}": pct(every.beat)}.items():
    html = html.replace(k, v)
open("../page/system_e.html", "w").write(html)
print("written", len(html))
