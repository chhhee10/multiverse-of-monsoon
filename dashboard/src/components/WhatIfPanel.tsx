import { type ReactNode, useEffect, useMemo, useState } from "react";
import { CartesianGrid, Legend as RLegend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CH_INDEX, type DemoWindow, N, Stack, W, cellLat, cellLon } from "../lib/data";
import type { Outline } from "../lib/geo";
import type { Region } from "../lib/impact";
import { diffField, maxAbsDiff } from "../lib/timeline";
import { VIEWS, type ViewKey, getField, niceRange } from "../lib/views";
import { CHART, tooltipStyle } from "./chartTheme";
import { DayControl, ViewTabs } from "./Controls";
import { type EditTool, GridMap } from "./GridMap";
import { Legend } from "./Legend";

const DAYS_1_10 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

export function WhatIfPanel({
  win,
  normalTL,
  shockedTL,
  truthTL,
  outline,
  editCells,
  ranCells,
  setCells,
  tool,
  brushRadius,
  view,
  setView,
  rippleKey,
  summary,
  editor,
  selectedCell,
  setSelectedCell,
  regions,
}: {
  win: DemoWindow;
  normalTL: Stack | null;
  shockedTL: Stack | null;
  truthTL: Stack;
  outline: Outline | null;
  editCells: Set<number>;
  ranCells: Set<number> | null;
  setCells: (s: Set<number>) => void;
  tool: EditTool | null;
  brushRadius: number;
  view: ViewKey;
  setView: (v: ViewKey) => void;
  rippleKey: number;
  summary: ReactNode;
  editor: ReactNode;
  selectedCell: number | null;
  setSelectedCell: (i: number) => void;
  regions: Region[];
}) {
  const [day, setDay] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [arrows, setArrows] = useState(false);
  const [showRegions, setShowRegions] = useState(false);
  const spec = VIEWS[view];

  // Each fresh what-if result replays the ripple from day 0 (the shock itself) through day 10.
  useEffect(() => {
    if (!rippleKey) return;
    setDay(0);
    setPlaying(true);
  }, [rippleKey]);

  const range = useMemo(() => {
    if (!normalTL || !shockedTL) return niceRange(0, spec.diffFloor);
    return niceRange(maxAbsDiff(normalTL, shockedTL, view, DAYS_1_10), spec.diffFloor);
  }, [normalTL, shockedTL, view, spec.diffFloor]);

  const normal = normalTL ? getField(normalTL, day, view) : null;
  const shocked = shockedTL ? getField(shockedTL, day, view) : null;
  const diff = normalTL && shockedTL ? diffField(normalTL, shockedTL, day, view) : null;
  const arrowsFor = (s: Stack | null) => (arrows && s ? { u: s.field(day, CH_INDEX.u10), v: s.field(day, CH_INDEX.v10) } : null);

  const footprint = useMemo(() => {
    if (!normalTL || !shockedTL) return [];
    const inside = ranCells ?? new Set<number>();
    return Array.from({ length: 11 }, (_, d) => {
      const f = diffField(normalTL, shockedTL, d, view);
      let si = 0,
        so = 0,
        ni = 0,
        no = 0,
        reach = 0;
      const thr = range * 0.1;
      for (let i = 0; i < N; i++) {
        const v = Math.abs(spec.diffToDisplay(f[i]));
        if (inside.has(i)) {
          si += v;
          ni++;
        } else {
          so += v;
          no++;
          if (v > thr) reach++;
        }
      }
      return { day: d, inside: +(si / Math.max(1, ni)).toFixed(4), outside: +(so / Math.max(1, no)).toFixed(4), reach };
    });
  }, [normalTL, shockedTL, ranCells, view, spec, range]);

  const series = useMemo(() => {
    if (selectedCell == null || !normalTL) return [];
    return Array.from({ length: 11 }, (_, d) => ({
      day: d,
      normal: +spec.toDisplay(getField(normalTL, d, view)[selectedCell]).toFixed(2),
      shocked: shockedTL ? +spec.toDisplay(getField(shockedTL, d, view)[selectedCell]).toFixed(2) : undefined,
      truth: +spec.toDisplay(getField(truthTL, d, view)[selectedCell]).toFixed(2),
    }));
  }, [selectedCell, normalTL, shockedTL, truthTL, view, spec]);

  const mapCommon = {
    spec,
    outline,
    shockCells: tool ? editCells : (ranCells ?? editCells),
    edit: tool,
    brushRadius,
    onPaint: setCells,
    selectedCell,
    onCellClick: setSelectedCell,
    regions: showRegions ? regions : null,
  };

  return (
    <section className="panel panel--hero">
      <header className="panel__head">
        <div>
          <div className="eyebrow">The multiverse</div>
          <h2>What if the weather started differently?</h2>
          <p className="muted">Nudge the last input day inside the amber box, and the twin carries the change forward for 10 days. Everything below is a live run.</p>
        </div>
      </header>
      <div className="summary" aria-live="polite">
        {summary}
      </div>
      <div className="whatif">
        {editor}
        <div className="whatif__main">
          <div className="toolbar">
            <ViewTabs value={view} onChange={setView} />
            <label className="check">
              <input type="checkbox" checked={arrows} onChange={(e) => setArrows(e.target.checked)} /> Wind arrows
            </label>
            <label className="check" title="Outline the 8 regions used by the impact module">
              <input type="checkbox" checked={showRegions} onChange={(e) => setShowRegions(e.target.checked)} /> Impact regions
            </label>
            <DayControl day={day} setDay={setDay} max={10} playing={playing} setPlaying={setPlaying} baseDate={win.last_input_day} playLabel="Play ripple" intervalMs={700} />
          </div>
          <div className="maps3">
            <GridMap {...mapCommon} title="Normal" badge="no shock" field={normal} arrows={arrowsFor(normalTL)} placeholder="Waiting for the normal run" />
            <GridMap {...mapCommon} title="Shocked" badge={shockedTL ? "with shock" : "not run yet"} field={shocked} arrows={arrowsFor(shockedTL)} placeholder="Pick a preset or draw a box, then Run" />
            <GridMap {...mapCommon} title="Difference" badge="shocked − normal" field={diff} diffRange={range} placeholder="Difference appears after a run" />
          </div>
          <div className="legend-row legend-row--3">
            <div className="legend-span2">
              <Legend spec={spec} />
            </div>
            <Legend spec={spec} diffRange={range} />
          </div>
          <div className="whatif__charts">
            <div>
              <div className="chart-title">
                How the shock spreads: mean |Δ {spec.short.toLowerCase()}| inside vs outside the box ({spec.diffUnit})
              </div>
              <div className="chart">
                {footprint.length ? (
                  <ResponsiveContainer width="100%" height={170}>
                    <LineChart data={footprint} margin={{ top: 6, right: 8, bottom: 0, left: -12 }}>
                      <CartesianGrid stroke={CHART.grid} vertical={false} />
                      <XAxis dataKey="day" tick={{ fill: CHART.tick, fontSize: 11 }} tickFormatter={(d) => (d ? `+${d}` : "0")} stroke={CHART.axis} />
                      <YAxis tick={{ fill: CHART.tick, fontSize: 11 }} stroke={CHART.axis} width={44} />
                      <Tooltip contentStyle={tooltipStyle} labelFormatter={(d) => `Day ${d ? "+" + d : 0}`} />
                      <RLegend wrapperStyle={{ fontSize: 11 }} />
                      <Line dataKey="inside" name="Inside box" stroke={CHART.shocked} strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
                      <Line dataKey="outside" name="Outside box" stroke={CHART.normal} strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="chart__empty">Run a what-if to see the ripple over time.</div>
                )}
              </div>
            </div>
            <div>
              <div className="chart-title">
                {selectedCell != null
                  ? `One cell, ${cellLat(Math.floor(selectedCell / W)).toFixed(1)}°N ${cellLon(selectedCell % W).toFixed(1)}°E: ${spec.short.toLowerCase()} (${spec.unit})`
                  : "One cell: pick Inspect, then click any map"}
              </div>
              <div className="chart">
                {series.length ? (
                  <ResponsiveContainer width="100%" height={170}>
                    <LineChart data={series} margin={{ top: 6, right: 8, bottom: 0, left: -12 }}>
                      <CartesianGrid stroke={CHART.grid} vertical={false} />
                      <XAxis dataKey="day" tick={{ fill: CHART.tick, fontSize: 11 }} tickFormatter={(d) => (d ? `+${d}` : "0")} stroke={CHART.axis} />
                      <YAxis tick={{ fill: CHART.tick, fontSize: 11 }} stroke={CHART.axis} width={50} domain={["auto", "auto"]} tickFormatter={(v: number) => String(+v.toFixed(2))} />
                      <Tooltip contentStyle={tooltipStyle} labelFormatter={(d) => `Day ${d ? "+" + d : 0}`} />
                      <RLegend wrapperStyle={{ fontSize: 11 }} />
                      <Line dataKey="truth" name="ERA5" stroke={CHART.truth} strokeDasharray="4 3" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                      <Line dataKey="normal" name="Normal" stroke={CHART.normal} strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
                      {shockedTL && <Line dataKey="shocked" name="Shocked" stroke={CHART.shocked} strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />}
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="chart__empty">Normal vs shocked vs ERA5 for a single grid cell.</div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
