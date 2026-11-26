// Country outlines (Natural Earth, India point of view, clipped to the twin's box) drawn over the grid.
import { geoContains, geoPath, geoTransform } from "d3-geo";
import type { Feature, FeatureCollection, MultiPolygon } from "geojson";
import { H, LAT0, LON0, N, W, cellLat, cellLon } from "./data";

export const PX = 10; // SVG units per degree / per cell
export const VB_W = W * PX;
export const VB_H = H * PX;
export const LAT1 = LAT0 + H;

export const xOf = (lon: number) => (lon - LON0) * PX;
export const yOf = (lat: number) => (LAT1 - lat) * PX;

type Country = Feature<MultiPolygon, { iso: string; name: string }>;

export interface Outline {
  india: string; // SVG path
  others: string;
  indiaFeature: Country | null;
}

const projection = geoTransform({
  point(lon: number, lat: number) {
    this.stream.point(xOf(lon), yOf(lat));
  },
});
const path = geoPath(projection);

export async function loadOutline(): Promise<Outline> {
  const r = await fetch(`${import.meta.env.BASE_URL}india_outline.geojson`);
  if (!r.ok) throw new Error(`india_outline.geojson: HTTP ${r.status}`);
  const fc = (await r.json()) as FeatureCollection<MultiPolygon, { iso: string; name: string }>;
  const india = fc.features.find((f) => f.properties.iso === "IND") ?? null;
  const others = fc.features.filter((f) => f.properties.iso !== "IND");
  return {
    india: india ? path(india) ?? "" : "",
    others: others.map((f) => path(f) ?? "").join(""),
    indiaFeature: india,
  };
}

/** 1 where the cell centre is inside India's boundary. */
export function indiaMask(o: Outline): Uint8Array {
  const m = new Uint8Array(N);
  if (!o.indiaFeature) return m.fill(1);
  for (let r = 0; r < H; r++)
    for (let c = 0; c < W; c++) m[r * W + c] = geoContains(o.indiaFeature, [cellLon(c), cellLat(r)]) ? 1 : 0;
  // d3 expects clockwise rings; if the file's winding is the other way, everything *outside* India matches.
  let n = 0;
  for (let i = 0; i < N; i++) n += m[i];
  if (n > N / 2) for (let i = 0; i < N; i++) m[i] = 1 - m[i];
  return m;
}

/** SVG path tracing the outer edges of a set of cells (works for boxes and painted shapes). */
export function cellsOutlinePath(cells: Set<number> | number[]): string {
  const s = cells instanceof Set ? cells : new Set(cells);
  let d = "";
  for (const i of s) {
    const r = Math.floor(i / W),
      c = i % W;
    const x0 = c * PX,
      x1 = x0 + PX,
      yTop = (H - 1 - r) * PX,
      yBot = yTop + PX;
    if (r === H - 1 || !s.has(i + W)) d += `M${x0},${yTop}H${x1}`; // north edge
    if (r === 0 || !s.has(i - W)) d += `M${x0},${yBot}H${x1}`; // south edge
    if (c === 0 || !s.has(i - 1)) d += `M${x0},${yTop}V${yBot}`; // west edge
    if (c === W - 1 || !s.has(i + 1)) d += `M${x1},${yTop}V${yBot}`; // east edge
  }
  return d;
}
