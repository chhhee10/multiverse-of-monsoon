// One plain-English sentence describing the shock and its biggest regional rain effect.
import type { ImpactResult } from "./impact";
import { type ShockComponent, describeComponents, mergeWind } from "./shock";

const n0 = (v: number) => (Math.round(v * 10) / 10).toString().replace(/\.0$/, "");

/** "A 3 K warmer, 15% moister Bay of Bengal" / "A 40% weaker Arabian Sea wind" / generic fallback. */
export function shockPhrase(comps: ShockComponent[], place: string): string {
  const named = !place.startsWith("around ");
  const noun = place.replace(/^the /, "");
  const over = named ? `over ${place}` : place;
  const cs = mergeWind(comps).filter((c) => c.value !== 0);
  if (!cs.length) return `A zero shock ${over}`;
  if (cs.length === 1 && ["wind", "u10", "v10"].includes(cs[0].v) && cs[0].mode === "pct") {
    const c = cs[0];
    const what = c.v === "wind" ? "wind" : c.v === "u10" ? "east-west wind" : "north-south wind";
    const size = `${n0(Math.abs(c.value))}% ${c.value < 0 ? "weaker" : "stronger"}`;
    return named ? `A ${size} ${noun} ${what}` : `A ${size} ${what} ${place}`;
  }
  const adj: string[] = [];
  for (const c of cs) {
    const a = Math.abs(c.value);
    if (c.v === "t2m" && c.mode === "abs") adj.push(`${n0(a)} K ${c.value > 0 ? "warmer" : "cooler"}`);
    else if (c.v === "tcwv" && c.mode === "pct") adj.push(`${n0(a)}% ${c.value > 0 ? "moister" : "drier"}`);
    else if (c.v === "tp" && c.mode === "pct") adj.push(`${n0(a)}% ${c.value > 0 ? "rainier" : "less rainy"}`);
    else if (c.v === "wind" && c.mode === "pct") adj.push(`${n0(a)}% ${c.value > 0 ? "windier" : "calmer"}`);
    else return `A shock of ${describeComponents(cs)} ${over}`;
  }
  return named ? `A ${adj.join(", ")} ${noun}` : `A ${adj.join(", ")} patch ${place}`;
}

export interface Headline {
  kind: "zero" | "effect" | "none";
  text: string;
}

export function headline(comps: ShockComponent[], place: string, impact: ImpactResult, pertMax: number, outMaxDiff: number): Headline {
  if (pertMax === 0) {
    return {
      kind: "zero",
      text:
        outMaxDiff === 0
          ? "Zero shock: the shocked run is identical to the normal run (max |shocked − normal| = 0 over all 6 variables and 10 days)."
          : `Zero shock, but the runs differ by up to ${outMaxDiff.toExponential(2)}: the model is not deterministic.`,
    };
  }
  const lead = shockPhrase(comps, place);
  const rows = impact.rows.filter((r) => r.rainN >= 2);
  const pool = rows.length ? rows : impact.rows;
  const best = [...pool].sort((a, b) => Math.abs(b.rainPct) - Math.abs(a.rainPct))[0];
  if (!best || Math.abs(best.rainPct) < 1) {
    return { kind: "none", text: `${lead} leaves rain in every region within 1% of normal over 10 days.` };
  }
  const verb = best.rainPct < 0 ? "cuts" : "raises";
  return {
    kind: "effect",
    text: `${lead} ${verb} ${best.region.name} rain by ${Math.abs(best.rainPct).toFixed(0)}% over 10 days (${best.rainN.toFixed(0)} → ${best.rainS.toFixed(0)} mm).`,
  };
}
