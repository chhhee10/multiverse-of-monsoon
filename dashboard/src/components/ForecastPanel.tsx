import { useMemo, useState } from "react";
import { CartesianGrid, Legend as RLegend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CH_INDEX, type DemoWindow, Stack } from "../lib/data";
import type { Outline } from "../lib/geo";
import { rmse } from "../lib/timeline";
import { VIEWS, type ViewKey, getField } from "../lib/views";
import { DayControl, ViewTabs } from "./Controls";
import { GridMap } from "./GridMap";
import { Legend } from "./Legend";
import { CHART, tooltipStyle } from "./chartTheme";

export function ForecastPanel({
  win,
  normalTL,
  truthTL,
  outline,
  loading,
  info,
}: {
  win: DemoWindow;
  normalTL: Stack | null;
  truthTL: Stack;
  outline: Outline | null;
  loading: boolean;
  info: string;
}) {
  const [view, setView] = useState<ViewKey>("tp");
  const [day, setDay] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [arrows, setArrows] = useState(false);
  const spec = VIEWS[view];

  const twin = normalTL ? getField(normalTL, day, view) : null;
  const truth = getField(truthTL, day, view);
  const arrowsFor = (s: Stack | null) => (arrows && s ? { u: s.field(day, CH_INDEX.u10), v: s.field(day, CH_INDEX.v10) } : null);

  const skill = useMemo(() => {
    if (!normalTL) return [];
    const pers = getField(truthTL, 0, view);
    return Array.from({ length: 10 }, (_, k) => {
      const d = k + 1;
      const t = getField(truthTL, d, view);
      return { day: d, twin: +rmse(getField(normalTL, d, view), t, view).toFixed(3), persistence: +rmse(pers, t, view).toFixed(3) };
    });
  }, [normalTL, truthTL, view]);

  const today = skill[day - 1];

  return (
    <section className="panel">
      <header className="panel__head">
        <div>
          <h2>Forecast vs reality</h2>
          <p className="muted">
            Left: the twin's 10-day forecast from a live <code>POST /simulate</code>. Right: what ERA5 recorded on the same days.
          </p>
        </div>
        <div className="panel__meta">{loading ? <span className="pill pill--busy">Running forecast…</span> : <span className="pill">{info}</span>}</div>
      </header>
      <div className="toolbar">
        <ViewTabs value={view} onChange={setView} />
        <label className="check">
          <input type="checkbox" checked={arrows} onChange={(e) => setArrows(e.target.checked)} /> Wind arrows
        </label>
        <DayControl day={day} setDay={setDay} max={10} playing={playing} setPlaying={setPlaying} baseDate={win.last_input_day} />
      </div>
      <div className="fvr">
        <GridMap title="Twin forecast" badge={day === 0 ? "input" : "live model"} field={twin} spec={spec} outline={outline} arrows={arrowsFor(normalTL)} placeholder={loading ? "Running…" : "No forecast yet"} />
        <GridMap title="ERA5 reality" badge={day === 0 ? "input" : "observed"} field={truth} spec={spec} outline={outline} arrows={arrowsFor(truthTL)} />
        <div className="fvr__side">
          <div className="stat-row">
            <div className="stat">
              <span className="stat__k">Twin RMSE, day {day === 0 ? "—" : `+${day}`}</span>
              <span className="stat__v">
                {today ? today.twin.toFixed(2) : "—"} <small>{spec.diffUnit}</small>
              </span>
            </div>
            <div className="stat">
              <span className="stat__k">Persistence RMSE</span>
              <span className="stat__v stat__v--muted">
                {today ? today.persistence.toFixed(2) : "—"} <small>{spec.diffUnit}</small>
              </span>
            </div>
          </div>
          <div className="chart-title">Error vs ERA5 by lead day ({spec.short.toLowerCase()}, lat-weighted RMSE)</div>
          <div className="chart">
            <ResponsiveContainer width="100%" height={190}>
              <LineChart data={skill} margin={{ top: 6, right: 8, bottom: 0, left: -12 }}>
                <CartesianGrid stroke={CHART.grid} vertical={false} />
                <XAxis dataKey="day" tick={{ fill: CHART.tick, fontSize: 11 }} tickFormatter={(d) => `+${d}`} stroke={CHART.axis} />
                <YAxis tick={{ fill: CHART.tick, fontSize: 11 }} stroke={CHART.axis} width={44} />
                <Tooltip contentStyle={tooltipStyle} labelFormatter={(d) => `Day +${d}`} formatter={(v) => [`${Number(v).toFixed(2)} ${spec.diffUnit}`]} />
                <RLegend wrapperStyle={{ fontSize: 11, color: CHART.tick }} />
                <Line type="monotone" dataKey="twin" name="Twin" stroke={CHART.normal} strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
                <Line type="monotone" dataKey="persistence" name="Persistence (repeat day 0)" stroke={CHART.muted} strokeDasharray="4 3" strokeWidth={1.5} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="muted small">
            Persistence = assume tomorrow looks like today. A useful twin should sit below the dashed line as the days go by.
          </p>
        </div>
      </div>
      <div className="legend-row legend-row--2">
        <Legend spec={spec} />
      </div>
    </section>
  );
}
