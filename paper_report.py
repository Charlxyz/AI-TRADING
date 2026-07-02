import os
"""
paper_report.py — Génère un rapport HTML depuis paper_trades.json

Usage :
    python paper_report.py
    python paper_report.py --log paper_trades.json --output rapport.html
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>FVG Paper Trading — Rapport</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --green: #3fb950; --red: #f85149; --yellow: #d29922;
    --blue: #58a6ff; --purple: #bc8cff; --text: #c9d1d9;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Consolas', monospace; padding: 24px; }}
  h1 {{ color: var(--blue); font-size: 1.4rem; margin-bottom: 4px; }}
  .sub {{ color: #8b949e; font-size: 0.85rem; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }}
  .card .label {{ font-size: 0.75rem; color: #8b949e; margin-bottom: 4px; }}
  .card .value {{ font-size: 1.3rem; font-weight: bold; }}
  .green {{ color: var(--green); }} .red {{ color: var(--red); }}
  .yellow {{ color: var(--yellow); }} .blue {{ color: var(--blue); }}
  .purple {{ color: var(--purple); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-top: 16px; }}
  th {{ background: var(--surface); color: #8b949e; padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #21262d; }}
  tr:hover td {{ background: #1c2128; }}
  .badge {{ padding: 2px 7px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; }}
  .badge-tp {{ background: #1a4731; color: var(--green); }}
  .badge-sl {{ background: #3d1a1a; color: var(--red); }}
  .badge-ok {{ background: #1a2e4a; color: var(--blue); }}
  .badge-warn {{ background: #3d2e1a; color: var(--yellow); }}
  .badge-bad {{ background: #3d1a1a; color: var(--red); }}
  canvas {{ display: block; margin-top: 24px; }}
  .section-title {{ font-size: 0.9rem; color: #8b949e; margin: 20px 0 10px; text-transform: uppercase; letter-spacing: 0.08em; }}
</style>
</head>
<body>
<h1>⚡ FVG Base Hits — Rapport Paper Trading</h1>
<p class="sub">Généré le {generated_at} · {n_trades} trades · Capital initial : {initial_balance:.2f}$</p>

<div class="grid">
  <div class="card"><div class="label">Balance finale</div><div class="value {bal_color}">{balance:.2f}$</div></div>
  <div class="card"><div class="label">PnL total</div><div class="value {pnl_color}">{pnl:+.2f}$</div></div>
  <div class="card"><div class="label">Win rate</div><div class="value {wr_color}">{win_rate:.1%}</div></div>
  <div class="card"><div class="label">Trades</div><div class="value blue">{n_trades}</div></div>
  <div class="card"><div class="label">Max Drawdown</div><div class="value {dd_color}">{max_dd:.2%}</div></div>
  <div class="card"><div class="label">Sharpe ratio</div><div class="value {sharpe_color}">{sharpe:.3f}</div></div>
  <div class="card"><div class="label">Conformité FVG+LTF</div><div class="value purple">{conformity:.1%}</div></div>
  <div class="card"><div class="label">Moy. gain / perte</div><div class="value">{avg_win:+.2f}$ / {avg_loss:+.2f}$</div></div>
</div>

<p class="section-title">Courbe d'équité</p>
<canvas id="chart" height="80"></canvas>

<p class="section-title">Historique des trades</p>
<table>
  <thead>
    <tr>
      <th>#</th><th>Direction</th><th>Levier</th><th>Entrée</th><th>Sortie</th>
      <th>PnL ($)</th><th>PnL (%)</th><th>Notionnel</th><th>Statut</th>
      <th>Conformité</th><th>Heure entrée</th>
    </tr>
  </thead>
  <tbody>
  {rows}
  </tbody>
</table>

<script>
const eq = {equity_json};
const canvas = document.getElementById('chart');
canvas.width = document.body.clientWidth - 48;
const ctx = canvas.getContext('2d');
const mn = Math.min(...eq), mx = Math.max(...eq), spread = mx - mn || 1;
const w = canvas.width, h = canvas.height;
ctx.strokeStyle = eq[eq.length-1] >= eq[0] ? '#3fb950' : '#f85149';
ctx.lineWidth = 1.5;
ctx.beginPath();
eq.forEach((v, i) => {{
  const x = (i / (eq.length - 1)) * w;
  const y = h - ((v - mn) / spread) * (h - 4) - 2;
  i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
}});
ctx.stroke();
// Ligne de base (capital initial)
const baseY = h - (({initial_balance} - mn) / spread) * (h - 4) - 2;
ctx.setLineDash([4, 4]);
ctx.strokeStyle = '#30363d';
ctx.lineWidth = 1;
ctx.beginPath();
ctx.moveTo(0, baseY);
ctx.lineTo(w, baseY);
ctx.stroke();
</script>
</body>
</html>"""


def build_report(log_path: str, output_path: str):
    data = json.loads(Path(log_path).read_text(encoding="utf-8"))
    trades  = data.get("trades", [])
    equity  = data.get("equity_curve", [data.get("initial_balance", 10000)])
    initial = data.get("initial_balance", 10000)
    balance = data.get("balance", initial)

    pnls   = [t["pnl_usdt"] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n      = len(trades)

    win_rate  = len(wins) / n if n else 0
    total_pnl = sum(pnls)
    avg_win   = sum(wins) / len(wins) if wins else 0
    avg_loss  = sum(losses) / len(losses) if losses else 0

    import numpy as np
    eq = np.array(equity)
    running_max = np.maximum.accumulate(eq)
    max_dd = float(((eq - running_max) / np.maximum(running_max, 1)).min()) if len(eq) > 1 else 0
    sharpe = float(np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(252)) if len(pnls) > 1 else 0

    conf_trades = [t for t in trades if t.get("had_fvg") and t.get("had_ltf_model")]
    conformity  = len(conf_trades) / n if n else 0

    # Couleurs
    def col(v, pos="green", neg="red"): return pos if v >= 0 else neg

    rows_html = ""
    for t in reversed(trades):
        pnl    = t["pnl_usdt"]
        status = {"tp_hit": '<span class="badge badge-tp">✅ TP</span>',
                  "sl_hit": '<span class="badge badge-sl">❌ SL</span>',
                  "manual": '<span class="badge badge-warn">✋ Manuel</span>'}.get(t["status"], t["status"])
        conf   = ('<span class="badge badge-ok">✅ FVG+LTF</span>' if (t.get("had_fvg") and t.get("had_ltf_model"))
                  else ('<span class="badge badge-warn">⚠️ FVG seul</span>' if t.get("had_fvg")
                        else '<span class="badge badge-bad">❌ Hors stratégie</span>'))
        dir_sym = "▲ BUY" if t["direction"] == "buy" else "▼ SELL"
        pnl_color = "green" if pnl > 0 else "red"
        rows_html += (
            f"<tr>"
            f"<td>{t['id']}</td>"
            f"<td><b>{dir_sym}</b></td>"
            f"<td>{t['entry_price']:.2f}</td>"
            f"<td>{t['exit_price']:.2f}</td>"
            f"<td class='{pnl_color}'><b>{pnl:+.2f}$</b></td>"
            f"<td class='{pnl_color}'>{t['pnl_pct']:+.2f}%</td>"
            f"<td>{status}</td>"
            f"<td>{conf}</td>"
            f"<td>{t['entry_time']}</td>"
            f"</tr>\n"
        )

    html = TEMPLATE.format(
        generated_at    = datetime.now().strftime("%d/%m/%Y %H:%M"),
        n_trades        = n,
        initial_balance = initial,
        balance         = balance,
        bal_color       = col(balance - initial),
        pnl             = total_pnl,
        pnl_color       = col(total_pnl),
        win_rate        = win_rate,
        wr_color        = "green" if win_rate >= 0.5 else "red",
        max_dd          = max_dd,
        dd_color        = "red" if max_dd < -0.05 else "yellow",
        sharpe          = sharpe,
        sharpe_color    = "green" if sharpe > 1 else ("yellow" if sharpe > 0 else "red"),
        conformity      = conformity,
        avg_win         = avg_win,
        avg_loss        = avg_loss,
        rows            = rows_html if rows_html else "<tr><td colspan='9'>Aucun trade clôturé</td></tr>",
        equity_json     = json.dumps(equity[-500:]),
    )

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"✅ Rapport généré : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None,
                        help="Symbole (ex: BTCUSDT) — déduit automatiquement le chemin du log")
    parser.add_argument("--log",    type=str, default=None,
                        help="Chemin direct vers le fichier JSON (override --symbol)")
    parser.add_argument("--output", type=str, default=None,
                        help="Chemin du rapport HTML (défaut: paper_sessions/SYMBOL/rapport.html)")
    args = parser.parse_args()

    # Résolution du chemin log
    if args.log:
        log_path = args.log
        out_path = args.output or "paper_report.html"
    elif args.symbol:
        sym = args.symbol.upper().replace("/", "")
        log_path = os.path.join("paper_sessions", sym, f"{sym}.json")
        out_path = args.output or os.path.join("paper_sessions", sym, "rapport.html")
    else:
        # Cherche tous les fichiers JSON dans paper_sessions/
        import glob
        found = glob.glob("paper_sessions/**/*.json", recursive=True)
        if not found:
            print("Aucun fichier de session trouvé. Lance d'abord paper_trading.py.")
            print("Usage : python paper_report.py --symbol BTCUSDT")
            exit(1)
        if len(found) == 1:
            log_path = found[0]
            out_path = args.output or log_path.replace(".json", "_rapport.html")
            print(f"Session trouvée : {log_path}")
        else:
            print("Plusieurs sessions trouvées. Précise --symbol :")
            for f in found:
                print(f"  {f}")
            print("\nEx : python paper_report.py --symbol BTCUSDT")
            exit(1)

    build_report(log_path, out_path)