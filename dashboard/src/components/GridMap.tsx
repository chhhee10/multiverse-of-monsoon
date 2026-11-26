import { useEffect, useMemo, useRef, useState } from "react";
import { H, N, W, cellLat, cellLon, cellsInBox } from "../lib/data";
import { type Outline, PX, VB_H, VB_W, cellsOutlinePath, xOf, yOf } from "../lib/geo";
import { RdBu_r, type ViewSpec, lut, rampPos } from "../lib/views";

export type EditTool = "box" | "brush" | "erase";

export interface GridMapProps {
  title: string;
  badge?: string;
  field: Float32Array | null; // physical units, row 0 = south
  spec: ViewSpec;
  diffRange?: number | null; // set => difference map, symmetric range in display units
  outline: Outline | null;
  shockCells?: Set<number> | null;
  arrows?: { u: Float32Array; v: Float32Array } | null;
  selectedCell?: number | null;
  regions?: Array<{ id: string; name: string; box: [number, number, number, number] }> | null;
  onCellClick?: (i: number) => void;
  edit?: EditTool | null;
  brushRadius?: number;
  onPaint?: (cells: Set<number>) => void;
  placeholder?: string;
  dim?: boolean;
}

const GRATICULE = (() => {
  let d = "";
  for (let lon = 70; lon <= 100; lon += 5) d += `M${xOf(lon)},0V${VB_H}`;
  for (let lat = 10; lat <= 35; lat += 5) d += `M0,${yOf(lat)}H${VB_W}`;
  return d;
})();

function arrowsPath(u: Float32Array, v: Float32Array): string {
  let d = "";
  for (let r = 0; r < H; r += 2)
    for (let c = 0; c < W; c += 2) {
      const i = r * W + c;
      const sp = Math.hypot(u[i], v[i]);
      if (!Number.isFinite(sp) || sp < 0.3) continue;
      const L = Math.min(sp / 12, 1) * PX * 1.7 + 2;
      const ux = u[i] / sp,
        uy = -v[i] / sp; // svg y points down
      const cx = (c + 1) * PX,
        cy = (H - 1 - r) * PX; // centre of the 2x2 block
      const x0 = cx - (ux * L) / 2,
        y0 = cy - (uy * L) / 2,
        x1 = cx + (ux * L) / 2,
        y1 = cy + (uy * L) / 2;
      const hx = ux * 3.2,
        hy = uy * 3.2;
      d += `M${x0.toFixed(1)},${y0.toFixed(1)}L${x1.toFixed(1)},${y1.toFixed(1)}`;
      d += `M${(x1 - hx - hy * 0.6).toFixed(1)},${(y1 - hy + hx * 0.6).toFixed(1)}L${x1.toFixed(1)},${y1.toFixed(1)}L${(x1 - hx + hy * 0.6).toFixed(1)},${(y1 - hy - hx * 0.6).toFixed(1)}`;
    }
  return d;
}

export function GridMap(p: GridMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [preview, setPreview] = useState<Set<number> | null>(null);
  const drag = useRef<{ start: number; working: Set<number> } | null>(null);

  // Paint the 36x32 field into a 36x32 canvas; CSS scales it up with crisp (pixelated) cells.
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(W, H);
    if (p.field) {
      const isDiff = p.diffRange != null;
      const L = lut(isDiff ? RdBu_r : p.spec.stops);
      const R = p.diffRange ?? 1;
      for (let r = 0; r < H; r++)
        for (let c = 0; c < W; c++) {
          const raw = p.field[r * W + c];
          const o = ((H - 1 - r) * W + c) * 4; // flip: row 0 (south) is drawn at the bottom
          if (!Number.isFinite(raw)) {
            img.data[o + 3] = 0;
            continue;
          }
          const t = isDiff ? Math.max(0, Math.min(1, p.spec.diffToDisplay(raw) / (2 * R) + 0.5)) : rampPos(p.spec, p.spec.toDisplay(raw));
          const k = Math.round(t * 255) * 3;
          img.data[o] = L[k];
          img.data[o + 1] = L[k + 1];
          img.data[o + 2] = L[k + 2];
          img.data[o + 3] = 255;
        }
    }
    ctx.putImageData(img, 0, 0);
  }, [p.field, p.spec, p.diffRange]);

  const shown = preview ?? p.shockCells ?? null;
  const shockPath = useMemo(() => (shown && shown.size ? cellsOutlinePath(shown) : ""), [shown]);
  const regionShapes = useMemo(
    () =>
      (p.regions ?? []).map((r) => {
        const cells = cellsInBox(...r.box);
        let top = Infinity,
          left = Infinity;
        for (const i of cells) {
          top = Math.min(top, (H - 1 - Math.floor(i / W)) * PX);
          left = Math.min(left, (i % W) * PX);
        }
        return { id: r.id, name: r.name, d: cellsOutlinePath(cells), x: left, y: top };
      }),
    [p.regions],
  );
  const arrowD = useMemo(() => (p.arrows ? arrowsPath(p.arrows.u, p.arrows.v) : ""), [p.arrows]);

  const cellAt = (e: React.PointerEvent): number | null => {
    const svg = svgRef.current;
    if (!svg) return null;
    const b = svg.getBoundingClientRect();
    const c = Math.floor(((e.clientX - b.left) / b.width) * W);
    const rFromTop = Math.floor(((e.clientY - b.top) / b.height) * H);
    if (c < 0 || c >= W || rFromTop < 0 || rFromTop >= H) return null;
    return (H - 1 - rFromTop) * W + c;
  };

  const brushCells = (i: number): number[] => {
    const rad = p.brushRadius ?? 1;
    const r0 = Math.floor(i / W),
      c0 = i % W;
    const out: number[] = [];
    for (let dr = -rad; dr <= rad; dr++)
      for (let dc = -rad; dc <= rad; dc++) {
        const r = r0 + dr,
          c = c0 + dc;
        if (r >= 0 && r < H && c >= 0 && c < W) out.push(r * W + c);
      }
    return out;
  };

  const rectCells = (a: number, b: number): Set<number> => {
    const r0 = Math.min(Math.floor(a / W), Math.floor(b / W)),
      r1 = Math.max(Math.floor(a / W), Math.floor(b / W));
    const c0 = Math.min(a % W, b % W),
      c1 = Math.max(a % W, b % W);
    const s = new Set<number>();
    for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) s.add(r * W + c);
    return s;
  };

  const onDown = (e: React.PointerEvent) => {
    const i = cellAt(e);
    if (i == null) return;
    if (!p.edit) {
      p.onCellClick?.(i);
      return;
    }
    (e.target as Element).setPointerCapture?.(e.pointerId);
    if (p.edit === "box") {
      drag.current = { start: i, working: rectCells(i, i) };
      setPreview(drag.current.working);
    } else {
      const working = new Set(p.shockCells ?? []);
      for (const j of brushCells(i)) p.edit === "erase" ? working.delete(j) : working.add(j);
      drag.current = { start: i, working };
      p.onPaint?.(new Set(working));
    }
  };
  const onMove = (e: React.PointerEvent) => {
    const i = cellAt(e);
    setHover(i);
    if (!drag.current || i == null || !p.edit) return;
    if (p.edit === "box") {
      drag.current.working = rectCells(drag.current.start, i);
      setPreview(drag.current.working);
    } else {
      let changed = false;
      for (const j of brushCells(i)) {
        if (p.edit === "erase" ? drag.current.working.delete(j) : !drag.current.working.has(j) && !!drag.current.working.add(j)) changed = true;
      }
      if (changed) p.onPaint?.(new Set(drag.current.working));
    }
  };
  const onUp = () => {
    if (drag.current && p.edit === "box") p.onPaint?.(drag.current.working);
    drag.current = null;
    setPreview(null);
  };

  let readout = " ";
  if (hover != null && p.field) {
    const r = Math.floor(hover / W),
      c = hover % W;
    const raw = p.field[hover];
    const isDiff = p.diffRange != null;
    const v = isDiff ? p.spec.diffToDisplay(raw) : p.spec.toDisplay(raw);
    const unit = isDiff ? p.spec.diffUnit : p.spec.unit;
    readout = `${cellLat(r).toFixed(1)}°N ${cellLon(c).toFixed(1)}°E · ${isDiff && v > 0 ? "+" : ""}${v.toFixed(p.spec.decimals)} ${unit}`;
  }

  const sel = p.selectedCell != null && p.selectedCell >= 0 && p.selectedCell < N ? p.selectedCell : null;

  return (
    <figure className={`map ${p.dim ? "map--dim" : ""}`}>
      <figcaption className="map__head">
        <span className="map__title">{p.title}</span>
        {p.badge && <span className="map__badge">{p.badge}</span>}
      </figcaption>
      <div className={`map__frame ${p.edit ? "map__frame--edit" : ""}`}>
        <canvas ref={canvasRef} width={W} height={H} className="map__canvas" />
        <svg
          ref={svgRef}
          className="map__svg"
          viewBox={`0 0 ${VB_W} ${VB_H}`}
          preserveAspectRatio="none"
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerCancel={onUp}
          onPointerLeave={() => setHover(null)}
        >
          <path d={GRATICULE} className="map__grat" />
          {p.outline && (
            <>
              <path d={p.outline.others} className="map__others" />
              <path d={p.outline.india} className="map__india-halo" />
              <path d={p.outline.india} className="map__india" />
            </>
          )}
          {arrowD && (
            <>
              <path d={arrowD} className="map__arrows-halo" />
              <path d={arrowD} className="map__arrows" />
            </>
          )}
          {shockPath && (
            <>
              <path d={shockPath} className="map__shock-halo" />
              <path d={shockPath} className={`map__shock ${preview ? "map__shock--preview" : ""}`} />
            </>
          )}
          {regionShapes.map((r) => (
            <g key={r.id} className="map__region">
              <path d={r.d} />
              <text x={r.x + 1.5} y={r.y - 1.5}>
                {r.name}
              </text>
            </g>
          ))}
          {sel != null && (
            <rect x={(sel % W) * PX} y={(H - 1 - Math.floor(sel / W)) * PX} width={PX} height={PX} className="map__sel" />
          )}
          {[10, 20, 30].map((lat) => (
            <text key={lat} x={3} y={yOf(lat) - 2} className="map__tick">
              {lat}°N
            </text>
          ))}
          {[70, 80, 90].map((lon) => (
            <text key={lon} x={xOf(lon) + 2} y={VB_H - 3} className="map__tick">
              {lon}°E
            </text>
          ))}
        </svg>
        {!p.field && p.placeholder && (
          <div className="map__placeholder">
            <span>{p.placeholder}</span>
          </div>
        )}
      </div>
      <div className="map__readout">{readout}</div>
    </figure>
  );
}
