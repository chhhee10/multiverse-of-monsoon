// Live calls to our own inference server (serve.py). Nothing here is pre-rendered.
import { H, N, NC, Stack, W } from "./data";

export const API_URL: string = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") || "http://localhost:8000";

export interface Health {
  status: string;
  model_name: string;
}

export async function health(signal?: AbortSignal): Promise<Health> {
  const r = await fetch(`${API_URL}/health`, { method: "POST", signal });
  if (!r.ok) throw new Error(`health: HTTP ${r.status}`);
  return r.json();
}

export interface SimResult {
  rollout: Stack; // steps days; rollout day k = forecast day k+1
  inferenceMs: number; // as reported by the server
  roundTripMs: number; // measured in the browser
}

/** perturbation: flat Float32Array [6*32*36] (channel-major, row 0 = south) or null. */
export async function simulate(
  initStates: number[][][][],
  doySinCos: [number, number],
  steps: number,
  perturbation: Float32Array | null,
  signal?: AbortSignal,
): Promise<SimResult> {
  const t0 = performance.now();
  const body = {
    init_states: initStates,
    doy_sincos: doySinCos,
    steps,
    perturbation: perturbation ? nest(perturbation) : null,
  };
  const r = await fetch(`${API_URL}/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try {
      const j = await r.json();
      if (j?.error) msg += `: ${j.error}`;
    } catch {
      /* ignore */
    }
    throw new Error(`simulate failed (${msg})`);
  }
  const j = (await r.json()) as { rollout: number[][][][]; inference_ms: number };
  const rollout = Stack.fromNested(j.rollout);
  return { rollout, inferenceMs: j.inference_ms, roundTripMs: performance.now() - t0 };
}

function nest(p: Float32Array): number[][][] {
  const out: number[][][] = [];
  for (let c = 0; c < NC; c++) {
    const ch: number[][] = [];
    for (let r = 0; r < H; r++) {
      const row: number[] = new Array(W);
      for (let x = 0; x < W; x++) row[x] = p[c * N + r * W + x];
      ch.push(row);
    }
    out.push(ch);
  }
  return out;
}
