import { RdBu_r, type ViewSpec, colorAt, rampPos } from "../lib/views";

/** Colour bar in real units. For a difference map pass `diffRange` (symmetric, centred on 0). */
export function Legend({ spec, diffRange, label }: { spec: ViewSpec; diffRange?: number | null; label?: string }) {
  const isDiff = diffRange != null;
  const stops = isDiff ? RdBu_r : spec.stops;
  const n = 24;
  const grad = Array.from({ length: n + 1 }, (_, k) => `${colorAt(stops, k / n)} ${((k / n) * 100).toFixed(1)}%`).join(",");
  const R = diffRange ?? 1;
  const ticks = isDiff ? [-R, -R / 2, 0, R / 2, R] : spec.ticks;
  const pos = (v: number) => (isDiff ? v / (2 * R) + 0.5 : rampPos(spec, v));
  const fmt = (v: number) => {
    const s = Math.abs(v) >= 10 || Number.isInteger(v) ? v.toFixed(0) : v.toFixed(Math.abs(v) < 1 ? 2 : 1);
    return isDiff && v > 0 ? `+${s}` : s.replace("-", "−");
  };
  return (
    <div className="legend">
      <div className="legend__label">
        {label ?? (isDiff ? `Shocked − normal, ${spec.short.toLowerCase()}` : spec.label)} <span className="legend__unit">({isDiff ? spec.diffUnit : spec.unit})</span>
      </div>
      <div className="legend__bar" style={{ background: `linear-gradient(to right, ${grad})` }} />
      <div className="legend__ticks">
        {ticks.map((t) => (
          <span key={t} style={{ left: `${pos(t) * 100}%` }}>
            {fmt(t)}
          </span>
        ))}
      </div>
    </div>
  );
}
