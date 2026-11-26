import { useEffect, useRef } from "react";
import { addDays, fmtDate } from "../lib/data";
import { VIEWS, VIEW_ORDER, type ViewKey } from "../lib/views";

export function ViewTabs({ value, onChange }: { value: ViewKey; onChange: (v: ViewKey) => void }) {
  return (
    <div className="seg" role="tablist" aria-label="Variable">
      {VIEW_ORDER.map((k) => (
        <button key={k} role="tab" aria-selected={k === value} className={`seg__btn ${k === value ? "is-on" : ""}`} onClick={() => onChange(k)} title={VIEWS[k].label}>
          {VIEWS[k].short}
        </button>
      ))}
    </div>
  );
}

/** Day slider with play. Day 0 = last input day; days 1..max are forecast days. */
export function DayControl({
  day,
  setDay,
  max,
  playing,
  setPlaying,
  baseDate,
  from = 0,
  playLabel = "Play",
  intervalMs = 650,
}: {
  day: number;
  setDay: (d: number) => void;
  max: number;
  playing: boolean;
  setPlaying: (b: boolean) => void;
  baseDate: string;
  from?: number;
  playLabel?: string;
  intervalMs?: number;
}) {
  const dayRef = useRef(day);
  dayRef.current = day;
  useEffect(() => {
    if (!playing) return;
    if (dayRef.current >= max) setDay(from);
    const id = window.setInterval(() => {
      const next = dayRef.current + 1;
      if (next > max) {
        setPlaying(false);
        return;
      }
      setDay(next);
    }, intervalMs);
    return () => window.clearInterval(id);
  }, [playing, max, from, intervalMs, setDay, setPlaying]);

  return (
    <div className="day">
      <button className="btn btn--ghost day__play" onClick={() => setPlaying(!playing)} aria-label={playing ? "Pause" : playLabel}>
        {playing ? (
          <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden>
            <rect x="3" y="2" width="3.5" height="12" fill="currentColor" />
            <rect x="9.5" y="2" width="3.5" height="12" fill="currentColor" />
          </svg>
        ) : (
          <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden>
            <path d="M4 2l10 6-10 6z" fill="currentColor" />
          </svg>
        )}
        <span>{playing ? "Pause" : playLabel}</span>
      </button>
      <input
        className="day__slider"
        type="range"
        min={0}
        max={max}
        step={1}
        value={day}
        onChange={(e) => {
          setPlaying(false);
          setDay(Number(e.target.value));
        }}
        aria-label="Forecast day"
      />
      <div className="day__label">
        <strong>{day === 0 ? "Day 0" : `Day +${day}`}</strong>
        <span>{fmtDate(addDays(baseDate, day))}</span>
        <span className="day__kind">{day === 0 ? "last input day" : "forecast"}</span>
      </div>
    </div>
  );
}
