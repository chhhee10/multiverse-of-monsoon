// Demo windows (real ERA5 starting days) and the grid geometry.
// Fields are stored flat: one channel of one day is a Float32Array of H*W, index = row * W + col, row 0 = SOUTH.

export const H = 32;
export const W = 36;
export const N = H * W;
export const NC = 6;
export const LAT0 = 6; // southern edge of row 0
export const LON0 = 65; // western edge of col 0

export const CHANNELS = ["t2m", "tp", "u10", "v10", "msl", "tcwv"] as const;
export type Channel = (typeof CHANNELS)[number];
export const CH_INDEX: Record<Channel, number> = { t2m: 0, tp: 1, u10: 2, v10: 3, msl: 4, tcwv: 5 };

/** A stack of days: data[(day * NC + ch) * N + cell]. */
export class Stack {
  readonly days: number;
  readonly data: Float32Array;
  constructor(days: number, data?: Float32Array) {
    this.days = days;
    this.data = data ?? new Float32Array(days * NC * N);
  }
  field(day: number, ch: number): Float32Array {
    const o = (day * NC + ch) * N;
    return this.data.subarray(o, o + N);
  }
  static fromNested(a: number[][][][]): Stack {
    const s = new Stack(a.length);
    let i = 0;
    for (const day of a) for (const ch of day) for (const row of ch) for (const v of row) s.data[i++] = v;
    return s;
  }
}

export interface Meta {
  channels: string[];
  units: string[];
  lat: number[];
  lon: number[];
  land: Float32Array; // land fraction, flat
  orography: Float32Array;
}

export interface DemoWindow {
  id: string;
  label: string;
  first_forecast_day: string;
  last_input_day: string;
  doy_sincos: [number, number];
  initNested: number[][][][]; // [3][6][32][36], sent as-is to /simulate
  init: Stack; // 3 days
  truth: Stack; // 10 days, truth day k = forecast day k+1
}

export interface DemoData {
  meta: Meta;
  windows: DemoWindow[];
}

interface RawWindow {
  id: string;
  label: string;
  first_forecast_day: string;
  last_input_day: string;
  doy_sincos: [number, number];
  init_states: number[][][][];
  truth: number[][][][];
}

const flat2 = (a: number[][]) => Float32Array.from(a.flat());

export async function loadDemo(): Promise<DemoData> {
  const r = await fetch(`${import.meta.env.BASE_URL}demo_windows.json`);
  if (!r.ok) throw new Error(`demo_windows.json: HTTP ${r.status}`);
  const raw = await r.json();
  const m = raw.meta;
  const meta: Meta = {
    channels: m.channels,
    units: m.units,
    lat: m.lat,
    lon: m.lon,
    land: flat2(m.land_sea_mask),
    orography: flat2(m.orography_m),
  };
  const windows: DemoWindow[] = (raw.windows as RawWindow[]).map((w) => ({
    id: w.id,
    label: w.label,
    first_forecast_day: w.first_forecast_day,
    last_input_day: w.last_input_day,
    doy_sincos: w.doy_sincos,
    initNested: w.init_states,
    init: Stack.fromNested(w.init_states),
    truth: Stack.fromNested(w.truth),
  }));
  return { meta, windows };
}

export function cellLat(row: number) {
  return LAT0 + row + 0.5;
}
export function cellLon(col: number) {
  return LON0 + col + 0.5;
}

/** Cells whose centre lies inside the lat/lon box (inclusive edges). */
export function cellsInBox(lat0: number, lat1: number, lon0: number, lon1: number): number[] {
  const out: number[] = [];
  for (let r = 0; r < H; r++)
    for (let c = 0; c < W; c++) {
      const la = cellLat(r),
        lo = cellLon(c);
      if (la >= lat0 && la <= lat1 && lo >= lon0 && lo <= lon1) out.push(r * W + c);
    }
  return out;
}

export function addDays(iso: string, d: number): Date {
  const t = new Date(iso + "T00:00:00Z");
  t.setUTCDate(t.getUTCDate() + d);
  return t;
}

export function fmtDate(t: Date): string {
  return t.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
}
