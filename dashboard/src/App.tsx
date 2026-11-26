import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ForecastPanel } from "./components/ForecastPanel";
import type { EditTool } from "./components/GridMap";
import { ImpactPanel } from "./components/ImpactPanel";
import { ShockEditor } from "./components/ShockEditor";
import { WhatIfPanel } from "./components/WhatIfPanel";
import { API_URL, type SimResult, health, simulate } from "./lib/api";
import { type DemoData, type DemoWindow, addDays, fmtDate, loadDemo } from "./lib/data";
import { type Outline, indiaMask, loadOutline } from "./lib/geo";
import { buildRegions, computeImpact } from "./lib/impact";
import { PRESETS, type Preset, type ShockComponent, buildPerturbation, maxAbs, placeName, presetCells } from "./lib/shock";
import { headline } from "./lib/summary";
import { stackMaxAbsDiff, timeline } from "./lib/timeline";
import type { ViewKey } from "./lib/views";

interface ShockRun {
  winId: string;
  cells: Set<number>;
  comps: ShockComponent[];
  pert: Float32Array;
  res: SimResult;
  place: string;
  presetId: string | null;
}

type Server = { state: "checking" } | { state: "ok"; name: string } | { state: "down"; err: string };

const sameCells = (a: Set<number>, b: Set<number>) => a.size === b.size && [...a].every((i) => b.has(i));
const PRESET_VIEW: Record<string, ViewKey> = { warm_bob: "t2m", monsoon_stall: "ws", zero: "t2m" };

export default function App() {
  const [demo, setDemo] = useState<DemoData | null>(null);
  const [outline, setOutline] = useState<Outline | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [server, setServer] = useState<Server>({ state: "checking" });
  const [winId, setWinId] = useState("monsoon_onset");

  const normalCache = useRef(new Map<string, SimResult>());
  const normalToken = useRef(0);
  const [normal, setNormal] = useState<{ winId: string; res: SimResult } | null>(null);
  const [normalBusy, setNormalBusy] = useState(false);
  const [normalErr, setNormalErr] = useState<string | null>(null);

  const [cells, setCellsRaw] = useState<Set<number>>(() => presetCells(PRESETS[0]));
  const [comps, setCompsRaw] = useState<ShockComponent[]>(() => PRESETS[0].components);
  const [activePreset, setActivePreset] = useState<string | null>(null);
  const [tool, setTool] = useState<EditTool | null>("box");
  const [brushRadius, setBrushRadius] = useState(1);
  const shockToken = useRef(0);
  const [shock, setShock] = useState<ShockRun | null>(null);
  const [shockBusy, setShockBusy] = useState(false);
  const [shockErr, setShockErr] = useState<string | null>(null);
  const [rippleKey, setRippleKey] = useState(0);
  const [whatIfView, setWhatIfView] = useState<ViewKey>("t2m");
  const [selectedCell, setSelectedCell] = useState<number | null>(null);
  const [heatThr, setHeatThr] = useState(35); // daily-mean t2m: 35 °C mean is already a severe heat day

  // ---------- load static inputs + check the server ----------
  useEffect(() => {
    loadDemo().then(setDemo, (e) => setLoadErr(String(e?.message ?? e)));
    loadOutline().then(setOutline, (e) => setLoadErr(String(e?.message ?? e)));
  }, []);

  const checkServer = useCallback(async () => {
    try {
      const h = await health();
      setServer({ state: "ok", name: h.model_name });
      return true;
    } catch (e) {
      setServer({ state: "down", err: String((e as Error)?.message ?? e) });
      return false;
    }
  }, []);
  useEffect(() => {
    checkServer();
    const id = window.setInterval(checkServer, 8000);
    return () => window.clearInterval(id);
  }, [checkServer]);

  const win: DemoWindow | null = demo?.windows.find((w) => w.id === winId) ?? null;

  // ---------- live runs ----------
  const runNormal = useCallback(
    async (w: DemoWindow, force = false): Promise<SimResult | null> => {
      const cached = normalCache.current.get(w.id);
      if (cached && !force) {
        setNormal({ winId: w.id, res: cached });
        return cached;
      }
      const token = ++normalToken.current;
      setNormalBusy(true);
      setNormalErr(null);
      try {
        const res = await simulate(w.initNested, w.doy_sincos, 10, null);
        normalCache.current.set(w.id, res);
        if (token === normalToken.current) setNormal({ winId: w.id, res });
        checkServer();
        return res;
      } catch (e) {
        if (token === normalToken.current) setNormalErr(String((e as Error)?.message ?? e));
        checkServer();
        return null;
      } finally {
        if (token === normalToken.current) setNormalBusy(false);
      }
    },
    [checkServer],
  );

  useEffect(() => {
    if (win) runNormal(win);
  }, [win, runNormal]);

  const runShock = useCallback(
    async (cellsArg: Set<number>, compsArg: ShockComponent[], presetId: string | null) => {
      if (!win || !demo) return;
      const token = ++shockToken.current;
      setShockBusy(true);
      setShockErr(null);
      try {
        const base = normalCache.current.get(win.id) ?? (await runNormal(win));
        if (!base) throw new Error("normal forecast unavailable");
        const pert = buildPerturbation(win.init, cellsArg, compsArg);
        const res = await simulate(win.initNested, win.doy_sincos, 10, pert);
        if (token !== shockToken.current) return;
        const preset = PRESETS.find((p) => p.id === presetId);
        const place = preset ? preset.place : placeName(cellsArg, demo.meta.land);
        setShock({ winId: win.id, cells: new Set(cellsArg), comps: compsArg.map((c) => ({ ...c })), pert, res, place, presetId });
        setRippleKey((k) => k + 1);
      } catch (e) {
        if (token === shockToken.current) setShockErr(String((e as Error)?.message ?? e));
        checkServer();
      } finally {
        if (token === shockToken.current) setShockBusy(false);
      }
    },
    [win, demo, runNormal, checkServer],
  );

  const setCells = (s: Set<number>) => {
    setCellsRaw(s);
    setActivePreset(null);
  };
  const setComps = (c: ShockComponent[]) => {
    setCompsRaw(c);
    setActivePreset(null);
  };
  const onPreset = (p: Preset) => {
    const pc = presetCells(p);
    setCellsRaw(pc);
    setCompsRaw(p.components.map((c) => ({ ...c })));
    setActivePreset(p.id);
    setWhatIfView(PRESET_VIEW[p.id] ?? "t2m");
    runShock(pc, p.components, p.id);
  };
  const onWindow = (id: string) => {
    if (id === winId) return;
    shockToken.current++;
    setShockBusy(false);
    setShock(null);
    setWinId(id);
  };

  // ---------- derived ----------
  const normalTL = useMemo(() => (win && normal && normal.winId === win.id ? timeline(win.init, normal.res.rollout) : null), [win, normal]);
  const truthTL = useMemo(() => (win ? timeline(win.init, win.truth) : null), [win]);
  const shockRun = shock && win && shock.winId === win.id ? shock : null;
  const shockedTL = useMemo(() => (win && shockRun ? timeline(win.init, shockRun.res.rollout, shockRun.pert) : null), [win, shockRun]);

  const regions = useMemo(() => (demo ? buildRegions(demo.meta.land) : []), [demo]);
  const mask = useMemo(() => (outline ? indiaMask(outline) : null), [outline]);
  const normalRes = normal && win && normal.winId === win.id ? normal.res : null;

  const impact = useMemo(() => {
    if (!demo || !mask || !normalRes || !shockRun) return null;
    return computeImpact(normalRes.rollout, shockRun.res.rollout, regions, demo.meta.land, mask, heatThr);
  }, [demo, mask, normalRes, shockRun, regions, heatThr]);

  const head = useMemo(() => {
    if (!impact || !shockRun || !normalRes) return null;
    return headline(shockRun.comps, shockRun.place, impact, maxAbs(shockRun.pert), stackMaxAbsDiff(normalRes.rollout, shockRun.res.rollout));
  }, [impact, shockRun, normalRes]);

  const dirty = !shockRun || !sameCells(shockRun.cells, cells) || JSON.stringify(shockRun.comps) !== JSON.stringify(comps);

  // ---------- render ----------
  if (loadErr) return <div className="fatal">Could not load dashboard data: {loadErr}</div>;
  if (!demo || !win || !truthTL) return <div className="boot">Loading demo windows…</div>;

  const normalInfo = normalErr
    ? `Error: ${normalErr}`
    : normalRes
      ? `Live run · 10 days in ${normalRes.inferenceMs.toFixed(0)} ms on the server (${normalRes.roundTripMs.toFixed(0)} ms round trip)`
      : "—";

  const summaryNode = shockBusy ? (
    <span className="summary__busy">Simulating the shocked world…</span>
  ) : shockErr ? (
    <span className="summary__err">What-if failed: {shockErr}</span>
  ) : head ? (
    <span className={`summary__text summary__text--${head.kind}`}>{head.text}</span>
  ) : (
    <span className="summary__idle">Pick a preset, or draw a box on a map and press Run what-if.</span>
  );

  const runInfo = shockRun
    ? `Last run: ${shockRun.res.inferenceMs.toFixed(0)} ms on the server · perturbation max |Δ| ${maxAbs(shockRun.pert).toFixed(2)} (physical units)${dirty ? " · edited since" : ""}`
    : null;

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <svg viewBox="0 0 32 32" width="28" height="28" aria-hidden>
            <rect width="32" height="32" rx="6" fill="#0e1a2b" />
            <path d="M5 21c4-8 7 3 11-4s6-4 11-6" stroke="#7dd3fc" strokeWidth="2.2" fill="none" strokeLinecap="round" />
            <path d="M5 25c4-6 7 2 11-3s6-3 11-4" stroke="#f59e0b" strokeWidth="1.6" fill="none" strokeLinecap="round" opacity="0.9" />
          </svg>
          <div>
            <h1>Multiverse of Monsoon</h1>
            <p>A digital twin of India's weather · 1° grid, 6 variables, 10-day rollouts</p>
          </div>
        </div>
        <div className={`server server--${server.state}`} title={API_URL}>
          <span className="server__dot" />
          {server.state === "ok" ? (
            <span>
              Model server live · <code>{server.name}</code>
            </span>
          ) : server.state === "checking" ? (
            <span>Checking model server…</span>
          ) : (
            <span>
              Model server unreachable at <code>{API_URL}</code>. Start it with <code>python serve.py --port 8000</code>
            </span>
          )}
        </div>
      </header>

      <nav className="windows" aria-label="Demo window">
        {demo.windows.map((w) => (
          <button key={w.id} className={`win ${w.id === winId ? "is-on" : ""}`} onClick={() => onWindow(w.id)}>
            <span className="win__label">{w.label}</span>
            <span className="win__date">
              {fmtDate(addDays(w.first_forecast_day, 0))} → {fmtDate(addDays(w.first_forecast_day, 9))}
            </span>
          </button>
        ))}
        <button className="btn btn--ghost btn--sm windows__rerun" onClick={() => runNormal(win, true)} disabled={normalBusy} title="Run the normal forecast again (live)">
          Re-run normal
        </button>
      </nav>

      <ForecastPanel win={win} normalTL={normalTL} truthTL={truthTL} outline={outline} loading={normalBusy} info={normalInfo} />

      <WhatIfPanel
        win={win}
        normalTL={normalTL}
        shockedTL={shockedTL}
        truthTL={truthTL}
        outline={outline}
        editCells={cells}
        ranCells={shockRun?.cells ?? null}
        setCells={setCells}
        tool={tool}
        brushRadius={brushRadius}
        view={whatIfView}
        setView={setWhatIfView}
        rippleKey={rippleKey}
        summary={summaryNode}
        selectedCell={selectedCell}
        setSelectedCell={setSelectedCell}
        regions={regions}
        editor={
          <ShockEditor
            cells={cells}
            setCells={setCells}
            comps={comps}
            setComps={setComps}
            activePreset={activePreset}
            onPreset={onPreset}
            tool={tool}
            setTool={setTool}
            brushRadius={brushRadius}
            setBrushRadius={setBrushRadius}
            onRun={() => runShock(cells, comps, activePreset)}
            running={shockBusy}
            canRun={cells.size > 0 && server.state !== "down"}
            dirty={dirty}
            runInfo={runInfo}
          />
        }
      />

      <ImpactPanel impact={impact} threshold={heatThr} setThreshold={setHeatThr} headline={head?.text} />

      <footer className="foot">
        <span>
          Every map is drawn from a live <code>POST {API_URL}/simulate</code> response; inputs and truth are real ERA5 days (CC BY 4.0) from <code>demo_windows.json</code>.
        </span>
        <span>Boundaries: Natural Earth, India point of view (public domain).</span>
      </footer>
    </div>
  );
}
