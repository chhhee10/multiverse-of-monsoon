// What-if shocks: a set of grid cells plus one or more variable changes, turned into the [6][32][36] delta that
// /simulate adds to the last input day. Zeros everywhere outside the shocked cells.
import { CH_INDEX, N, NC, Stack, W, cellLat, cellLon, cellsInBox } from "./data";

export type ShockVar = "t2m" | "tp" | "wind" | "u10" | "v10" | "msl" | "tcwv";
export type ShockMode = "abs" | "pct";

export interface ShockComponent {
  v: ShockVar;
  mode: ShockMode;
  value: number; // abs: K, mm/day, hPa (msl); pct: % of the current value
}

export interface Range {
  min: number;
  max: number;
  step: number;
  unit: string;
}

export interface ShockVarSpec {
  label: string;
  noun: string; // used in the plain-English summary
  channels: number[];
  abs?: Range & { toPhysical: number };
  pct?: Range;
}

// UI clamps from the spec: temperature ±10 K, wind ±80 %, moisture ±50 %, rain ≥ −100 %, pressure ±2000 Pa.
export const SHOCK_VARS: Record<ShockVar, ShockVarSpec> = {
  t2m: { label: "Temperature (t2m)", noun: "temperature", channels: [CH_INDEX.t2m], abs: { min: -10, max: 10, step: 0.5, unit: "K", toPhysical: 1 } },
  tcwv: { label: "Moisture (tcwv)", noun: "moisture", channels: [CH_INDEX.tcwv], pct: { min: -50, max: 50, step: 1, unit: "%" } },
  wind: { label: "Wind (u10 + v10)", noun: "wind", channels: [CH_INDEX.u10, CH_INDEX.v10], pct: { min: -80, max: 80, step: 5, unit: "%" } },
  u10: { label: "East wind (u10)", noun: "east-west wind", channels: [CH_INDEX.u10], pct: { min: -80, max: 80, step: 5, unit: "%" } },
  v10: { label: "North wind (v10)", noun: "north-south wind", channels: [CH_INDEX.v10], pct: { min: -80, max: 80, step: 5, unit: "%" } },
  tp: {
    label: "Rain (tp)", noun: "rain", channels: [CH_INDEX.tp],
    pct: { min: -100, max: 200, step: 5, unit: "%" },
    abs: { min: -50, max: 100, step: 1, unit: "mm/day", toPhysical: 1 },
  },
  msl: { label: "Pressure (msl)", noun: "pressure", channels: [CH_INDEX.msl], abs: { min: -20, max: 20, step: 0.5, unit: "hPa", toPhysical: 100 } },
};

export const SHOCK_VAR_ORDER: ShockVar[] = ["t2m", "tcwv", "wind", "u10", "v10", "tp", "msl"];

export function rangeFor(c: Pick<ShockComponent, "v" | "mode">): Range {
  const s = SHOCK_VARS[c.v];
  return (c.mode === "abs" ? s.abs : s.pct) ?? (s.abs ?? s.pct)!;
}

export function defaultMode(v: ShockVar): ShockMode {
  return SHOCK_VARS[v].abs ? "abs" : "pct";
}

export function clampComponent(c: ShockComponent): ShockComponent {
  const s = SHOCK_VARS[c.v];
  const mode: ShockMode = c.mode === "abs" ? (s.abs ? "abs" : "pct") : s.pct ? "pct" : "abs";
  const r = rangeFor({ v: c.v, mode });
  const value = Number.isFinite(c.value) ? Math.max(r.min, Math.min(r.max, c.value)) : 0;
  return { v: c.v, mode, value };
}

export interface Box {
  lat0: number;
  lat1: number;
  lon0: number;
  lon1: number;
}

export interface Preset {
  id: string;
  name: string;
  place: string;
  description: string;
  box: Box;
  components: ShockComponent[];
}

export const PRESETS: Preset[] = [
  {
    id: "warm_bob",
    name: "Warm Bay of Bengal",
    place: "the Bay of Bengal",
    description: "+3 K on t2m and +15% of the current tcwv over 8–20°N, 82–94°E",
    box: { lat0: 8, lat1: 20, lon0: 82, lon1: 94 },
    components: [
      { v: "t2m", mode: "abs", value: 3 },
      { v: "tcwv", mode: "pct", value: 15 },
    ],
  },
  {
    id: "monsoon_stall",
    name: "Monsoon stall",
    place: "the Arabian Sea",
    description: "−40% of the current u10 and v10 over 8–20°N, 65–75°E",
    box: { lat0: 8, lat1: 20, lon0: 65, lon1: 75 },
    components: [
      { v: "u10", mode: "pct", value: -40 },
      { v: "v10", mode: "pct", value: -40 },
    ],
  },
  {
    id: "zero",
    name: "Zero shock",
    place: "the Bay of Bengal",
    description: "Sanity check: an all-zero delta must reproduce the normal forecast exactly",
    box: { lat0: 8, lat1: 20, lon0: 82, lon1: 94 },
    components: [{ v: "t2m", mode: "abs", value: 0 }],
  },
];

export function presetCells(p: Preset): Set<number> {
  return new Set(cellsInBox(p.box.lat0, p.box.lat1, p.box.lon0, p.box.lon1));
}

/** Build the flat [6*32*36] delta. `init` is the 3-day input stack; changes are relative to its last day. */
export function buildPerturbation(init: Stack, cells: Set<number>, comps: ShockComponent[]): Float32Array {
  const d = new Float32Array(NC * N);
  const last = init.days - 1;
  for (const raw of comps) {
    const c = clampComponent(raw);
    const spec = SHOCK_VARS[c.v];
    for (const ch of spec.channels) {
      const cur = init.field(last, ch);
      for (const i of cells) {
        let delta: number;
        if (c.mode === "abs") delta = c.value * (spec.abs?.toPhysical ?? 1);
        else delta = (cur[i] * c.value) / 100;
        if (ch === CH_INDEX.tp) delta = Math.max(delta, -cur[i] - d[ch * N + i]); // rain can't go below zero
        d[ch * N + i] += delta;
      }
    }
  }
  return d;
}

export function maxAbs(a: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const v = Math.abs(a[i]);
    if (v > m) m = v;
  }
  return m;
}

export function fmtSigned(v: number, digits = 0): string {
  const s = Math.abs(v).toFixed(digits);
  if (Number(s) === 0) return (0).toFixed(digits);
  return (v > 0 ? "+" : "−") + s;
}

export function describeComponent(c: ShockComponent): string {
  const s = SHOCK_VARS[c.v];
  const r = rangeFor(c);
  const val = c.mode === "abs" ? `${fmtSigned(c.value, r.step < 1 ? 1 : 0)} ${r.unit}` : `${fmtSigned(c.value)}%`;
  return `${val} ${s.noun}`;
}

/** e.g. "+3.0 K temperature and +15% moisture" */
export function describeComponents(comps: ShockComponent[]): string {
  const parts = mergeWind(comps).map(describeComponent);
  if (parts.length <= 1) return parts[0] ?? "no change";
  return parts.slice(0, -1).join(", ") + " and " + parts[parts.length - 1];
}

export function mergeWind(comps: ShockComponent[]): ShockComponent[] {
  const u = comps.find((c) => c.v === "u10" && c.mode === "pct");
  const v = comps.find((c) => c.v === "v10" && c.mode === "pct");
  if (u && v && u.value === v.value) return [{ v: "wind", mode: "pct", value: u.value }, ...comps.filter((c) => c !== u && c !== v)];
  return comps;
}

export function cellsCentroid(cells: Set<number>): { lat: number; lon: number } {
  let la = 0,
    lo = 0;
  for (const i of cells) {
    la += cellLat(Math.floor(i / W));
    lo += cellLon(i % W);
  }
  const n = Math.max(1, cells.size);
  return { lat: la / n, lon: lo / n };
}

/** Rough place name for a custom shock, from its centroid. */
export function placeName(cells: Set<number>, landFrac: Float32Array): string {
  if (!cells.size) return "nowhere";
  const { lat, lon } = cellsCentroid(cells);
  let land = 0;
  for (const i of cells) land += landFrac[i];
  const seaShare = 1 - land / cells.size;
  const at = `${lat.toFixed(1)}°N ${lon.toFixed(1)}°E`;
  if (seaShare > 0.6) {
    if (lon >= 80 && lat < 23) return "the Bay of Bengal";
    if (lon < 78 && lat < 25) return "the Arabian Sea";
    if (lat < 10) return "the Indian Ocean";
  }
  return `around ${at}`;
}
