import type { ReactNode } from "react";
import { Bar, BarChart, CartesianGrid, Legend as RLegend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ImpactResult } from "../lib/impact";
import { fmtSigned } from "../lib/shock";
import { CHART, tooltipStyle } from "./chartTheme";

const SHORT: Record<string, string> = {
  kerala: "Kerala",
  konkan: "Konkan",
  gujarat: "Gujarat",
  vidarbha: "Vidarbha",
  delhi: "Delhi",
  gangetic: "Gangetic",
  odisha: "Odisha",
  tamilnadu: "Tamil Nadu",
};

function pctCell(p: number, digits = 0) {
  const cls = Math.abs(p) < 0.5 ? "chg chg--zero" : p > 0 ? "chg chg--up" : "chg chg--down";
  return <span className={cls}>{fmtSigned(p, digits)}%</span>;
}

function MetricChart({ title, data, unit }: { title: string; data: Array<{ name: string; normal: number; shocked: number }>; unit: string }) {
  return (
    <div className="impact__chart">
      <div className="chart-title">{title}</div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 6, right: 6, bottom: 0, left: -10 }} barGap={1}>
          <CartesianGrid stroke={CHART.grid} vertical={false} />
          <XAxis dataKey="name" tick={{ fill: CHART.tick, fontSize: 10 }} stroke={CHART.axis} interval={0} angle={-30} textAnchor="end" height={46} />
          <YAxis tick={{ fill: CHART.tick, fontSize: 11 }} stroke={CHART.axis} width={44} />
          <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "rgba(148,163,184,0.08)" }} formatter={(v) => `${Number(v).toFixed(1)} ${unit}`} />
          <RLegend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="normal" name="Normal" fill={CHART.normal} isAnimationActive={false} />
          <Bar dataKey="shocked" name="Shocked" fill={CHART.shocked} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function secondary(impact: ImpactResult): string {
  const parts: string[] = [];
  const flagged = impact.rows.filter((r) => r.rainFlag).map((r) => r.region.name);
  parts.push(flagged.length ? `Water-stress flag (rain down > 25%): ${flagged.join(", ")}.` : "No region loses more than 25% of its rain.");
  const heat = impact.rows.filter((r) => r.heatS !== r.heatN).sort((a, b) => Math.abs(b.heatS - b.heatN) - Math.abs(a.heatS - a.heatN));
  parts.push(
    heat.length
      ? `Heat days above ${impact.heatThresholdC} °C change in ${heat.map((r) => `${r.region.name} (${r.heatN} → ${r.heatS})`).join(", ")}.`
      : `Heat days above ${impact.heatThresholdC} °C unchanged.`,
  );
  const wind = [...impact.rows].sort((a, b) => Math.abs(b.windPct) - Math.abs(a.windPct))[0];
  parts.push(wind && Math.abs(wind.windPct) >= 1 ? `Wind power changes most in ${wind.region.name} (${fmtSigned(wind.windPct)}%).` : "Wind power within 1% everywhere.");
  return parts.join(" ");
}

export function ImpactPanel({
  impact,
  threshold,
  setThreshold,
  headline,
}: {
  impact: ImpactResult | null;
  threshold: number;
  setThreshold: (n: number) => void;
  headline: ReactNode;
}) {
  return (
    <section className="panel">
      <header className="panel__head">
        <div>
          <h2>Impact on people</h2>
          <p className="muted">Computed in the browser from the live normal and shocked 10-day runs. Regions are simple lat/lon boxes (land cells).</p>
        </div>
        <label className="small muted thresh">
          Heat threshold{" "}
          <input type="number" className="num" min={30} max={48} step={1} value={threshold} onChange={(e) => setThreshold(Math.max(30, Math.min(48, Number(e.target.value) || 40)))} /> °C
          <span className="thresh__note">daily-mean t2m</span>
        </label>
      </header>
      {!impact ? (
        <div className="empty">Run a what-if above to see crop water stress, heat stress and wind power for 8 regions.</div>
      ) : (
        <>
          <div className="impact__headline">{headline}</div>
          <p className="impact__sub">{secondary(impact)}</p>
          <div className="kpis">
            <div className="kpi">
              <span className="kpi__k">All-India land rain, 10 days</span>
              <span className="kpi__v">
                {impact.india.rainN.toFixed(1)} → {impact.india.rainS.toFixed(1)} mm {pctCell(impact.india.rainPct, 1)}
              </span>
            </div>
            <div className="kpi">
              <span className="kpi__k">Hot land cell-days (&gt; {impact.heatThresholdC} °C)</span>
              <span className="kpi__v">
                {impact.india.hotCellDaysN} → {impact.india.hotCellDaysS}
              </span>
            </div>
            <div className="kpi">
              <span className="kpi__k">Wind power proxy, India land + coast</span>
              <span className="kpi__v">
                {impact.india.windN.toFixed(0)} → {impact.india.windS.toFixed(0)} m³/s³ {pctCell(impact.india.windPct, 1)}
              </span>
            </div>
          </div>
          <div className="impact">
            <table className="table">
              <thead>
                <tr>
                  <th rowSpan={2}>Region</th>
                  <th colSpan={4} className="th-group">
                    Kharif water: 10-day rain (mm)
                  </th>
                  <th colSpan={2} className="th-group">
                    Heat days &gt; {impact.heatThresholdC} °C
                  </th>
                  <th colSpan={3} className="th-group">
                    Wind power, mean |w|³ (m³/s³)
                  </th>
                </tr>
                <tr>
                  <th>Normal</th>
                  <th>Shocked</th>
                  <th>Change</th>
                  <th>Stress</th>
                  <th>Normal</th>
                  <th>Shocked</th>
                  <th>Normal</th>
                  <th>Shocked</th>
                  <th>Change</th>
                </tr>
              </thead>
              <tbody>
                {impact.rows.map((r) => (
                  <tr key={r.region.id} className={r.rainFlag ? "is-flag" : ""}>
                    <td className="td-name">
                      {r.region.name}
                      <span className="td-cells">{r.region.cells.length} cells</span>
                    </td>
                    <td>{r.rainN.toFixed(1)}</td>
                    <td>{r.rainS.toFixed(1)}</td>
                    <td>{pctCell(r.rainPct)}</td>
                    <td>{r.rainFlag ? <span className="flag">Water stress</span> : <span className="muted">—</span>}</td>
                    <td>{r.heatN}</td>
                    <td className={r.heatS > r.heatN ? "hot" : ""}>{r.heatS}</td>
                    <td>{r.windN.toFixed(0)}</td>
                    <td>{r.windS.toFixed(0)}</td>
                    <td>{pctCell(r.windPct)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="small muted">
              Water stress = 10-day rain drops more than 25% vs normal. Heat days = days on which any land cell in the region has daily-mean t2m above the threshold. Wind power ∝ |wind|³,
              averaged over land and coastal cells.
            </p>
          </div>
          <div className="impact__charts">
            <MetricChart title="10-day rain (mm)" unit="mm" data={impact.rows.map((r) => ({ name: SHORT[r.region.id], normal: +r.rainN.toFixed(2), shocked: +r.rainS.toFixed(2) }))} />
            <MetricChart title={`Heat days (> ${impact.heatThresholdC} °C)`} unit="days" data={impact.rows.map((r) => ({ name: SHORT[r.region.id], normal: r.heatN, shocked: r.heatS }))} />
            <MetricChart title="Wind power proxy (m³/s³)" unit="m³/s³" data={impact.rows.map((r) => ({ name: SHORT[r.region.id], normal: +r.windN.toFixed(1), shocked: +r.windS.toFixed(1) }))} />
          </div>
        </>
      )}
    </section>
  );
}
