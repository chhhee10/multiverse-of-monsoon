import { PRESETS, type Preset, SHOCK_VARS, SHOCK_VAR_ORDER, type ShockComponent, type ShockVar, cellsCentroid, clampComponent, defaultMode, rangeFor } from "../lib/shock";
import { useEffect, useState } from "react";
import type { EditTool } from "./GridMap";

/** Number box that lets people type freely ("-", "-6") and only commits real numbers (clamped by the caller). */
function NumInput({ value, min, max, step, onCommit }: { value: number; min: number; max: number; step: number; onCommit: (v: number) => void }) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  return (
    <input
      type="number"
      className="num"
      min={min}
      max={max}
      step={step}
      value={text}
      aria-label="Magnitude value"
      onChange={(e) => {
        setText(e.target.value);
        const v = parseFloat(e.target.value);
        if (Number.isFinite(v) && v >= min && v <= max) onCommit(v);
      }}
      onBlur={() => {
        const v = parseFloat(text);
        onCommit(Number.isFinite(v) ? Math.max(min, Math.min(max, v)) : 0);
        setText(String(Number.isFinite(v) ? Math.max(min, Math.min(max, v)) : 0));
      }}
      onKeyDown={(e) => {
        if (e.key === "Enter") (e.target as HTMLInputElement).blur();
      }}
    />
  );
}

export function ShockEditor({
  cells,
  setCells,
  comps,
  setComps,
  activePreset,
  onPreset,
  tool,
  setTool,
  brushRadius,
  setBrushRadius,
  onRun,
  running,
  canRun,
  dirty,
  runInfo,
}: {
  cells: Set<number>;
  setCells: (s: Set<number>) => void;
  comps: ShockComponent[];
  setComps: (c: ShockComponent[]) => void;
  activePreset: string | null;
  onPreset: (p: Preset) => void;
  tool: EditTool | null;
  setTool: (t: EditTool | null) => void;
  brushRadius: number;
  setBrushRadius: (n: number) => void;
  onRun: () => void;
  running: boolean;
  canRun: boolean;
  dirty: boolean;
  runInfo: string | null;
}) {
  const update = (i: number, patch: Partial<ShockComponent>) => {
    const next = comps.map((c, j) => {
      if (j !== i) return c;
      const merged = { ...c, ...patch };
      if (patch.v && patch.v !== c.v) {
        merged.mode = defaultMode(patch.v);
        merged.value = 0;
      }
      return clampComponent(merged);
    });
    setComps(next);
  };
  const addComp = () => {
    const used = new Set(comps.map((c) => c.v));
    const v = SHOCK_VAR_ORDER.find((k) => !used.has(k)) ?? "t2m";
    setComps([...comps, { v, mode: defaultMode(v), value: 0 }]);
  };
  const centroid = cells.size ? cellsCentroid(cells) : null;

  return (
    <aside className="shock">
      <div className="shock__group">
        <div className="shock__label">Presets (judges' tests)</div>
        <div className="presets">
          {PRESETS.map((p) => (
            <button key={p.id} className={`preset ${activePreset === p.id ? "is-on" : ""} ${p.id === "zero" ? "preset--zero" : ""}`} onClick={() => onPreset(p)} disabled={running}>
              <span className="preset__name">{p.name}</span>
              <span className="preset__desc">{p.description}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="shock__group">
        <div className="shock__label">
          Custom shock: where
          <span className="shock__hint">{cells.size ? `${cells.size} cells · centre ${centroid!.lat.toFixed(1)}°N ${centroid!.lon.toFixed(1)}°E` : "draw on any map"}</span>
        </div>
        <div className="seg seg--full">
          {(
            [
              ["box", "Box"],
              ["brush", "Brush"],
              ["erase", "Erase"],
              [null, "Inspect"],
            ] as Array<[EditTool | null, string]>
          ).map(([t, label]) => (
            <button key={label} className={`seg__btn ${tool === t ? "is-on" : ""}`} onClick={() => setTool(t)} title={t ? `${label} cells on the map` : "Click a cell to see its time series"}>
              {label}
            </button>
          ))}
        </div>
        <div className="shock__row">
          {(tool === "brush" || tool === "erase") && (
            <label className="small muted">
              Brush{" "}
              <select value={brushRadius} onChange={(e) => setBrushRadius(Number(e.target.value))}>
                <option value={0}>1 cell</option>
                <option value={1}>3 × 3</option>
                <option value={2}>5 × 5</option>
              </select>
            </label>
          )}
          <button className="btn btn--ghost btn--sm" onClick={() => setCells(new Set())} disabled={!cells.size || running}>
            Clear area
          </button>
        </div>
      </div>

      <div className="shock__group">
        <div className="shock__label">Custom shock: what</div>
        {comps.map((c, i) => {
          const spec = SHOCK_VARS[c.v];
          const r = rangeFor(c);
          return (
            <div className="comp" key={i}>
              <div className="comp__top">
                <select value={c.v} onChange={(e) => update(i, { v: e.target.value as ShockVar })} aria-label="Variable">
                  {SHOCK_VAR_ORDER.map((k) => (
                    <option key={k} value={k}>
                      {SHOCK_VARS[k].label}
                    </option>
                  ))}
                </select>
                {spec.abs && spec.pct ? (
                  <div className="seg seg--sm">
                    <button className={`seg__btn ${c.mode === "abs" ? "is-on" : ""}`} onClick={() => update(i, { mode: "abs", value: 0 })}>
                      {spec.abs.unit}
                    </button>
                    <button className={`seg__btn ${c.mode === "pct" ? "is-on" : ""}`} onClick={() => update(i, { mode: "pct", value: 0 })}>
                      %
                    </button>
                  </div>
                ) : (
                  <span className="comp__mode">{c.mode === "abs" ? "absolute" : "% of current"}</span>
                )}
                <button className="icon-btn" onClick={() => setComps(comps.filter((_, j) => j !== i))} aria-label="Remove" title="Remove" disabled={comps.length <= 1}>
                  ×
                </button>
              </div>
              <div className="comp__val">
                <input type="range" min={r.min} max={r.max} step={r.step} value={c.value} onChange={(e) => update(i, { value: Number(e.target.value) })} aria-label="Magnitude" />
                <NumInput value={c.value} min={r.min} max={r.max} step={r.step} onCommit={(v) => update(i, { value: v })} />
                <span className="comp__unit">{r.unit}</span>
              </div>
              <div className="comp__range">
                clamp {r.min} … +{r.max} {r.unit}
              </div>
            </div>
          );
        })}
        <button className="btn btn--ghost btn--sm" onClick={addComp} disabled={comps.length >= 4}>
          + Add variable
        </button>
      </div>

      <div className="shock__run">
        <button className="btn btn--primary" onClick={onRun} disabled={!canRun || running}>
          {running ? "Simulating…" : dirty ? "Run what-if" : "Re-run what-if"}
        </button>
        <div className="small muted">{runInfo ?? "Sends the shock as a live perturbation to /simulate."}</div>
      </div>
    </aside>
  );
}
