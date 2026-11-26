// Impact module: computed in the browser from the live normal and shocked rollouts.
import { CH_INDEX, H, N, Stack, W, cellsInBox } from "./data";

export interface Region {
  id: string;
  name: string;
  box: [number, number, number, number]; // lat0, lat1, lon0, lon1 (cell centres inside are used)
  cells: number[]; // land (and coastal) cells used for every metric
}

const REGION_BOXES: Array<[string, string, [number, number, number, number]]> = [
  ["kerala", "Kerala", [8, 12.6, 74.4, 76.6]],
  ["konkan", "Konkan / Mumbai", [15.4, 20.6, 72.4, 73.6]],
  ["gujarat", "Gujarat", [20.4, 24.6, 68.4, 74.6]],
  ["vidarbha", "Vidarbha", [19.4, 21.6, 76.4, 80.6]],
  ["delhi", "Delhi NCR", [27.4, 29.6, 76.4, 77.6]],
  ["gangetic", "Gangetic plain", [24.4, 27.6, 78.4, 87.6]],
  ["odisha", "Odisha coast", [19.4, 21.6, 84.4, 87.6]],
  ["tamilnadu", "Tamil Nadu", [8.4, 13.6, 77.4, 80.6]],
];

export function buildRegions(land: Float32Array): Region[] {
  return REGION_BOXES.map(([id, name, box]) => {
    const all = cellsInBox(...box);
    const landCells = all.filter((i) => land[i] >= 0.3);
    return { id, name, box, cells: landCells.length ? landCells : all };
  });
}

/** Land cells (≥ 50% land) plus sea cells touching land: where wind farms could plausibly go. */
export function landAndCoastal(land: Float32Array, inside?: Uint8Array): number[] {
  const out: number[] = [];
  for (let r = 0; r < H; r++)
    for (let c = 0; c < W; c++) {
      const i = r * W + c;
      const isLand = land[i] >= 0.5 && (!inside || inside[i]);
      let coastal = false;
      if (!isLand && land[i] < 0.5) {
        for (let dr = -1; dr <= 1 && !coastal; dr++)
          for (let dc = -1; dc <= 1; dc++) {
            const rr = r + dr,
              cc = c + dc;
            if (rr < 0 || rr >= H || cc < 0 || cc >= W) continue;
            const j = rr * W + cc;
            if (land[j] >= 0.5 && (!inside || inside[j])) {
              coastal = true;
              break;
            }
          }
      }
      if (isLand || coastal) out.push(i);
    }
  return out;
}

export interface RegionImpact {
  region: Region;
  rainN: number; // 10-day total of the regional-mean rain, mm
  rainS: number;
  rainPct: number;
  rainFlag: boolean; // drop of more than 25 %
  heatN: number; // days with any land cell above the threshold
  heatS: number;
  windN: number; // mean |wind|^3, m^3/s^3
  windS: number;
  windPct: number;
}

export interface ImpactResult {
  rows: RegionImpact[];
  india: { rainN: number; rainS: number; rainPct: number; hotCellDaysN: number; hotCellDaysS: number; windN: number; windS: number; windPct: number };
  heatThresholdC: number;
  days: number;
}

const pct = (a: number, b: number) => (Math.abs(a) < 1e-9 ? (Math.abs(b) < 1e-9 ? 0 : 100) : ((b - a) / a) * 100);

function rainTotal(s: Stack, cells: number[]): number {
  let t = 0;
  for (let k = 0; k < s.days; k++) {
    const f = s.field(k, CH_INDEX.tp);
    let m = 0;
    for (const i of cells) m += f[i];
    t += m / cells.length;
  }
  return t;
}

function heatDays(s: Stack, cells: number[], thrK: number): number {
  let d = 0;
  for (let k = 0; k < s.days; k++) {
    const f = s.field(k, CH_INDEX.t2m);
    if (cells.some((i) => f[i] > thrK)) d++;
  }
  return d;
}

function hotCellDays(s: Stack, cells: number[], thrK: number): number {
  let n = 0;
  for (let k = 0; k < s.days; k++) {
    const f = s.field(k, CH_INDEX.t2m);
    for (const i of cells) if (f[i] > thrK) n++;
  }
  return n;
}

function windCube(s: Stack, cells: number[]): number {
  let t = 0;
  for (let k = 0; k < s.days; k++) {
    const u = s.field(k, CH_INDEX.u10),
      v = s.field(k, CH_INDEX.v10);
    for (const i of cells) t += Math.pow(Math.hypot(u[i], v[i]), 3);
  }
  return t / (s.days * cells.length);
}

export function computeImpact(
  normal: Stack,
  shocked: Stack,
  regions: Region[],
  land: Float32Array,
  indiaMask: Uint8Array,
  heatThresholdC = 35,
): ImpactResult {
  const thrK = heatThresholdC + 273.15;
  const rows = regions.map((region): RegionImpact => {
    const landOnly = region.cells; // already land-filtered
    const rainN = rainTotal(normal, landOnly),
      rainS = rainTotal(shocked, landOnly);
    const rainPct = pct(rainN, rainS);
    const windN = windCube(normal, region.cells),
      windS = windCube(shocked, region.cells);
    return {
      region,
      rainN,
      rainS,
      rainPct,
      rainFlag: rainN > 1 && rainPct < -25,
      heatN: heatDays(normal, landOnly, thrK),
      heatS: heatDays(shocked, landOnly, thrK),
      windN,
      windS,
      windPct: pct(windN, windS),
    };
  });
  const indiaLand: number[] = [];
  for (let i = 0; i < N; i++) if (indiaMask[i] && land[i] >= 0.5) indiaLand.push(i);
  const coastal = landAndCoastal(land, indiaMask);
  const rainN = rainTotal(normal, indiaLand),
    rainS = rainTotal(shocked, indiaLand);
  const windN = windCube(normal, coastal),
    windS = windCube(shocked, coastal);
  return {
    rows,
    india: {
      rainN,
      rainS,
      rainPct: pct(rainN, rainS),
      hotCellDaysN: hotCellDays(normal, indiaLand, thrK),
      hotCellDaysS: hotCellDays(shocked, indiaLand, thrK),
      windN,
      windS,
      windPct: pct(windN, windS),
    },
    heatThresholdC,
    days: normal.days,
  };
}
