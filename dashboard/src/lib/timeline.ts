// Day-indexed views: day 0 = last input day, day k (1..10) = forecast day k.
import { H, N, NC, Stack, W } from "./data";
import { type ViewKey, getField, VIEWS } from "./views";

/** [day0, ...rest] as one 11-day stack. day0 is the last day of `init`, optionally with a delta added. */
export function timeline(init: Stack, rest: Stack, delta?: Float32Array | null): Stack {
  const out = new Stack(rest.days + 1);
  const off = (init.days - 1) * NC * N;
  out.data.set(init.data.subarray(off, off + NC * N), 0);
  if (delta) for (let i = 0; i < NC * N; i++) out.data[i] += delta[i];
  out.data.set(rest.data, NC * N);
  return out;
}

export function diffField(a: Stack, b: Stack, day: number, view: ViewKey): Float32Array {
  const fa = getField(a, day, view),
    fb = getField(b, day, view);
  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) out[i] = fb[i] - fa[i];
  return out;
}

/** Largest |b − a| in display units over the given days. */
export function maxAbsDiff(a: Stack, b: Stack, view: ViewKey, days: number[]): number {
  const spec = VIEWS[view];
  let m = 0;
  for (const d of days) {
    const f = diffField(a, b, d, view);
    for (let i = 0; i < N; i++) {
      const v = Math.abs(spec.diffToDisplay(f[i]));
      if (v > m) m = v;
    }
  }
  return m;
}

const COSW = (() => {
  const w = new Float32Array(N);
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) w[r * W + c] = Math.cos(((6.5 + r) * Math.PI) / 180);
  return w;
})();

/** Latitude-weighted RMSE between two fields, in display (difference) units. */
export function rmse(a: Float32Array, b: Float32Array, view: ViewKey): number {
  const spec = VIEWS[view];
  let s = 0,
    ws = 0;
  for (let i = 0; i < N; i++) {
    const d = spec.diffToDisplay(a[i] - b[i]);
    s += COSW[i] * d * d;
    ws += COSW[i];
  }
  return Math.sqrt(s / ws);
}

export function stackMaxAbsDiff(a: Stack, b: Stack): number {
  let m = 0;
  for (let i = 0; i < a.data.length; i++) {
    const v = Math.abs(a.data[i] - b.data[i]);
    if (v > m) m = v;
  }
  return m;
}
