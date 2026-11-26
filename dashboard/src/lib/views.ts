// What can be drawn on a map: the six model channels plus derived wind speed, each with a FIXED colour scale in
// display units so frames never flicker. Colour maps are standard perceptual / ColorBrewer ramps.
import { CH_INDEX, N, Stack } from "./data";

export type ViewKey = "t2m" | "tp" | "ws" | "u10" | "v10" | "msl" | "tcwv";

const RdYlBu_r = ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"];
const Blues = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"];
const PuOr_r = ["#2d004b", "#542788", "#8073ac", "#b2abd2", "#d8daeb", "#f7f7f7", "#fee0b6", "#fdb863", "#e08214", "#b35806", "#7f3b08"];
const Viridis = ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"];
const YlGnBu = ["#ffffd9", "#edf8b1", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8", "#253494", "#081d58"];
const Magma = ["#000004", "#1c1044", "#4f127b", "#812581", "#b5367a", "#e55064", "#fb8761", "#fec287", "#fcfdbf"];
export const RdBu_r = ["#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"];

export interface ViewSpec {
  key: ViewKey;
  label: string;
  short: string;
  unit: string;
  diffUnit: string;
  toDisplay: (v: number) => number; // physical -> display units
  diffToDisplay: (v: number) => number; // physical delta -> display delta
  min: number;
  max: number;
  ticks: number[];
  stops: string[];
  /** display value -> [0,1] position on the ramp (defaults to linear) */
  pos?: (v: number) => number;
  diffFloor: number; // smallest symmetric range for difference maps (display units)
  decimals: number;
}

const id = (v: number) => v;
const sqrtPos = (max: number) => (v: number) => Math.sqrt(Math.max(0, v) / max);

export const VIEWS: Record<ViewKey, ViewSpec> = {
  t2m: {
    key: "t2m", label: "2 m temperature", short: "Temp", unit: "°C", diffUnit: "K",
    toDisplay: (v) => v - 273.15, diffToDisplay: id,
    min: -10, max: 45, ticks: [-10, 0, 10, 20, 30, 40], stops: RdYlBu_r, diffFloor: 0.5, decimals: 1,
  },
  tp: {
    key: "tp", label: "Rainfall", short: "Rain", unit: "mm/day", diffUnit: "mm/day",
    toDisplay: id, diffToDisplay: id,
    min: 0, max: 100, ticks: [0, 1, 5, 10, 25, 50, 100], stops: Blues, pos: sqrtPos(100), diffFloor: 2, decimals: 1,
  },
  ws: {
    key: "ws", label: "10 m wind speed", short: "Wind", unit: "m/s", diffUnit: "m/s",
    toDisplay: id, diffToDisplay: id,
    min: 0, max: 15, ticks: [0, 3, 6, 9, 12, 15], stops: Magma, diffFloor: 0.5, decimals: 1,
  },
  u10: {
    key: "u10", label: "10 m east wind (u)", short: "u10", unit: "m/s", diffUnit: "m/s",
    toDisplay: id, diffToDisplay: id,
    min: -15, max: 15, ticks: [-15, -10, -5, 0, 5, 10, 15], stops: PuOr_r, diffFloor: 0.5, decimals: 1,
  },
  v10: {
    key: "v10", label: "10 m north wind (v)", short: "v10", unit: "m/s", diffUnit: "m/s",
    toDisplay: id, diffToDisplay: id,
    min: -15, max: 15, ticks: [-15, -10, -5, 0, 5, 10, 15], stops: PuOr_r, diffFloor: 0.5, decimals: 1,
  },
  msl: {
    key: "msl", label: "Sea-level pressure", short: "Pressure", unit: "hPa", diffUnit: "hPa",
    toDisplay: (v) => v / 100, diffToDisplay: (v) => v / 100,
    min: 990, max: 1030, ticks: [990, 1000, 1010, 1020, 1030], stops: Viridis, diffFloor: 0.5, decimals: 1,
  },
  tcwv: {
    key: "tcwv", label: "Column water vapour", short: "Moisture", unit: "kg/m²", diffUnit: "kg/m²",
    toDisplay: id, diffToDisplay: id,
    min: 0, max: 70, ticks: [0, 10, 20, 30, 40, 50, 60, 70], stops: YlGnBu, diffFloor: 1, decimals: 1,
  },
};

export const VIEW_ORDER: ViewKey[] = ["t2m", "tp", "ws", "u10", "v10", "msl", "tcwv"];

/** Physical field for one day of a stack. */
export function getField(s: Stack, day: number, view: ViewKey): Float32Array {
  if (view === "ws") {
    const u = s.field(day, CH_INDEX.u10),
      v = s.field(day, CH_INDEX.v10);
    const out = new Float32Array(N);
    for (let i = 0; i < N; i++) out[i] = Math.hypot(u[i], v[i]);
    return out;
  }
  return s.field(day, CH_INDEX[view]);
}

// ---------- colour lookup ----------
function hexToRgb(h: string): [number, number, number] {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export type Lut = Uint8ClampedArray; // 256 * 3

const lutCache = new Map<string[], Lut>();
export function lut(stops: string[]): Lut {
  const hit = lutCache.get(stops);
  if (hit) return hit;
  const rgb = stops.map(hexToRgb);
  const out = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (rgb.length - 1);
    const k = Math.min(rgb.length - 2, Math.floor(t));
    const f = t - k;
    for (let j = 0; j < 3; j++) out[i * 3 + j] = rgb[k][j] + (rgb[k + 1][j] - rgb[k][j]) * f;
  }
  lutCache.set(stops, out);
  return out;
}

export function colorAt(stops: string[], t: number): string {
  const L = lut(stops);
  const i = Math.round(Math.max(0, Math.min(1, t)) * 255) * 3;
  return `rgb(${L[i]},${L[i + 1]},${L[i + 2]})`;
}

/** Position on the ramp for a display value. */
export function rampPos(spec: ViewSpec, v: number): number {
  if (spec.pos) return Math.max(0, Math.min(1, spec.pos(v)));
  return Math.max(0, Math.min(1, (v - spec.min) / (spec.max - spec.min)));
}

/** Nice symmetric range for a difference map. */
export function niceRange(maxAbs: number, floor: number): number {
  const m = Math.max(maxAbs, floor);
  const p = Math.pow(10, Math.floor(Math.log10(m)));
  for (const k of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (k * p >= m * 0.999) return k * p;
  return 10 * p;
}
