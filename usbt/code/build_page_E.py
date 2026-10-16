"""Builds ../page/system_e.html (close-out page) from results/final_E_* and the experiment summaries."""
import json, math, os
import pandas as pd
X = os.environ.get("SYS", "E")
NAME = f"System {X}"

R = "../results/"
cur = pd.read_csv(R + f"final_{X}_curves.csv", index_col=0, parse_dates=True)
yrs = pd.read_csv(R + f"final_{X}_years.csv", index_col=0)
sc = pd.read_csv(R + f"final_{X}_scorecard.csv")
sm = json.load(open(R + f"final_{X}_summary.json"))
SER = [(NAME, "e"), ("live v5", "v5"), ("SPY", "spy"), ("QQQ", "qqq"), ("SMH", "smh")]

# ---- growth chart (log scale), weekly points
W, H, L, Rt, T, B = 920, 400, 64, 118, 18, 34
c = cur.resample("W-FRI").last()
lo, hi = 1500, max(1_200_000, cur.max().max() * 1.15)
x0, x1 = c.index[0], c.index[-1]
fx = lambda t: L + (t - x0).days / (x1 - x0).days * (W - L - Rt)
fy = lambda v: T + (math.log10(hi) - math.log10(v)) / (math.log10(hi) - math.log10(lo)) * (H - T - B)
g = []
for v, lab in [(2000, "$2k"), (5000, "$5k"), (10000, "$10k"), (20000, "$20k"), (50000, "$50k"), (100000, "$100k"), (200000, "$200k"), (500000, "$500k"), (1000000, "$1M"), (2000000, "$2M")]:
    if v > hi: continue
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

# ---- year-by-year line chart (calendar-year returns, %)
YW, YH, YL, YR, YT, YB = 920, 380, 56, 20, 30, 34
yv = yrs.copy(); yv.index = yv.index.astype(int)
ymin = math.floor(min(-60, yv.min().min()) / 20) * 20; ymax = math.ceil(max(80, yv.max().max()) / 20) * 20
yrs_i = list(yv.index); fxy = lambda y: YL + (y - yrs_i[0]) / (yrs_i[-1] - yrs_i[0]) * (YW - YL - YR)
fyy = lambda v: YT + (ymax - v) / (ymax - ymin) * (YH - YT - YB)
yg = []
for v in range(int(ymin), int(ymax) + 1, 20):
    yy = fyy(v); yg.append(f'<line class="grid{" zero" if v == 0 else ""}" x1="{YL}" x2="{YW-YR}" y1="{yy:.1f}" y2="{yy:.1f}"/><text class="ax" x="{YL-6}" y="{yy+4:.1f}" text-anchor="end">{v:+d}%</text>')
for y in yrs_i:
    if y % 2 == 0 or y == yrs_i[-1]:
        yg.append(f'<text class="ax" x="{fxy(y):.1f}" y="{YH-12}" text-anchor="middle">{"’26*" if y == 2026 else y}</text>')
for y, r in yv.iterrows():                      # shade years where the system beat both SPY and QQQ
    if r[NAME] > max(r["SPY"], r["QQQ"]):
        w_ = (YW - YL - YR) / (len(yrs_i) - 1)
        yg.append(f'<rect class="winband" x="{fxy(y)-w_/2:.1f}" y="{YT}" width="{w_:.1f}" height="{YH-YT-YB}"/>')
ylines = []
for name, cls in SER[::-1]:
    s_ = yv[name].dropna()
    ylines.append(f'<polyline class="ln {cls}" points="{" ".join(f"{fxy(y):.1f},{fyy(v):.1f}" for y, v in s_.items())}"/>')
    ylines.append("".join(f'<circle class="dot {cls}" cx="{fxy(y):.1f}" cy="{fyy(v):.1f}" r="{3.2 if cls == "e" else 2.2}"><title>{y} {name}: {v:+.1f}%</title></circle>' for y, v in s_.items()))
leg = "".join(f'<span class="lg {cls}"><i></i>{name}</span>' for name, cls in SER)
ychart = f'<div class="legend">{leg}<span class="lg band"><i></i>{NAME} beat SPY and QQQ</span></div><svg viewBox="0 0 {YW} {YH}" role="img" aria-label="Calendar-year returns, {NAME} vs SPY, QQQ, SMH and live v5">{"".join(yg)}{"".join(ylines)}</svg>'

# ---- year table
rows = []
full = yrs.loc[[y for y in yrs.index if y <= 2025]]
wins = 0
for y, r in yrs.iterrows():
    best = max(r["SPY"], r["QQQ"])
    win = r[NAME] > best
    if y <= 2025:
        wins += win
    cell = lambda v, k: f'<td class="num{" hi" if k == NAME else ""}{" neg" if v < 0 else ""}">{v:+.1f}</td>'
    tag = '<span class="pill w">beat both</span>' if win else '<span class="pill l">behind</span>'
    yl = f"{y}" + (" <small>to 12 Oct</small>" if y == 2026 else "")
    rows.append(f'<tr><th scope="row">{yl}</th>{cell(r[NAME],NAME)}{cell(r["SPY"],"")}{cell(r["QQQ"],"")}{cell(r["SMH"],"")}{cell(r["live v5"],"")}<td>{tag}</td></tr>')
year_table = "\n".join(rows)

# ---- scorecard
def pct(v): return f"{v*100:.0f}%"
sc_rows = []
for era in ("every start", "starts 2000-11", "starts 2012-19", "starts 2020-25"):
    e = sc[(sc.system == NAME) & (sc.era == era)].iloc[0]; v = sc[(sc.system == "live v5") & (sc.era == era)].iloc[0]
    sc_rows.append(f'<tr><th scope="row">{era.replace("starts ", "Starts ").replace("every start", "Every monthly start")}</th>'
                   f'<td class="num hi">{pct(e.beat)}</td><td class="num">{pct(v.beat)}</td><td class="num hi">{pct(e.cagr)}</td><td class="num">{pct(v.cagr)}</td>'
                   f'<td class="num hi">{pct(e.ddok)}</td><td class="num hi">{e.worst*100:.0f}%</td><td class="num hi">{pct(e.smh3)}</td></tr>')
sc_table = "\n".join(sc_rows)
E, S, Q, M, V = (sm[k] for k in (NAME, "SPY", "QQQ", "SMH", "live v5"))
every = sc[(sc.system == NAME) & (sc.era == "every start")].iloc[0]
html = open("page_template_E.html").read()
for k, v in {"{{CHART}}": chart, "{{YCHART}}": ychart, "{{YEARS}}": year_table, "{{SCORE}}": sc_table, "{{WINS}}": str(int(wins)),
             "{{E_CAGR}}": f"{E['cagr']*100:.1f}%", "{{E_END}}": f"${E['end']/1000:,.0f}k", "{{E_DD}}": f"{E['maxdd']*100:.0f}%",
             "{{S_CAGR}}": f"{S['cagr']*100:.1f}%", "{{Q_CAGR}}": f"{Q['cagr']*100:.1f}%", "{{S_DD}}": f"{S['maxdd']*100:.0f}%", "{{Q_DD}}": f"{Q['maxdd']*100:.0f}%",
             "{{S_END}}": f"${S['end']/1000:,.0f}k", "{{Q_END}}": f"${Q['end']/1000:,.0f}k", "{{V_CAGR}}": f"{V['cagr']*100:.1f}%", "{{V_END}}": f"${V['end']/1000:,.0f}k",
             "{{BEAT}}": pct(every.beat), "{{SYS}}": NAME}.items():
    html = html.replace(k, v)
open("../page/system_e.html", "w").write(html)
print("written", len(html))
