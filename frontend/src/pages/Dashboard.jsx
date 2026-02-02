import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { fetchPredictionRecords } from "../api";
import "../styles/Dashboard.css";

/**
 * Notes:
 * - Dashboard uses backend database records (saved entries).
 * - Charts use simple SVG (no external chart libraries).
 */

function startOfWeekMonday(d) {
  const date = new Date(d);
  const day = date.getDay();
  const diffToMonday = (day + 6) % 7;
  date.setHours(0, 0, 0, 0);
  date.setDate(date.getDate() - diffToMonday);
  return date;
}

function addDays(date, n) {
  const d = new Date(date);
  d.setDate(d.getDate() + n);
  return d;
}

function sameDay(a, b) {
  return (
    a.getFullYear() === b.getFullYear() &&
    a.getMonth() === b.getMonth() &&
    a.getDate() === b.getDate()
  );
}

function safeNumber(x) {
  if (x === null || x === undefined) return null;
  const n = typeof x === "number" ? x : parseFloat(String(x).trim());
  return Number.isFinite(n) ? n : null;
}

function parseAcValueToNumber(ac) {
  if (ac === null || ac === undefined) return null;
  const s = String(ac).toLowerCase().trim();
  const num = parseFloat(s.replace(/[^\d.]/g, ""));
  if (!Number.isFinite(num)) return null;
  if (s.includes("mm")) return num / 10.0;
  return num;
}

function formatMaybe(n, digits = 2) {
  if (n === null || n === undefined) return "—";
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(digits);
}

/** ---------- SVG Line Chart ---------- */
function LineChart({ labels, values, height = 140 }) {
  const w = 520;
  const h = height;
  const pad = 26;

  const maxV = Math.max(1, ...values);
  const minV = 0;

  const xStep = values.length > 1 ? (w - pad * 2) / (values.length - 1) : 0;

  const points = values.map((v, i) => {
    const x = pad + i * xStep;
    const y = pad + (h - pad * 2) * (1 - (v - minV) / (maxV - minV));
    return { x, y, v };
  });

  const pathD = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`)
    .join(" ");

  return (
    <div className="chart-card">
      <div className="chart-head">
        <h3>Predictions (This Week)</h3>
        <p className="chart-sub">Counts per day (Mon → Sun)</p>
      </div>

      <svg
        className="chart-svg"
        viewBox={`0 0 ${w} ${h}`}
        role="img"
        aria-label="Predictions per day line chart"
      >
        {[0.25, 0.5, 0.75].map((t) => {
          const y = pad + (h - pad * 2) * t;
          return (
            <line
              key={t}
              x1={pad}
              x2={w - pad}
              y1={y}
              y2={y}
              className="grid-line"
            />
          );
        })}

        <path d={pathD} className="line-path" />

        {points.map((p, idx) => (
          <g key={idx}>
            <circle cx={p.x} cy={p.y} r="4" className="dot" />
            <text
              x={p.x}
              y={p.y - 10}
              textAnchor="middle"
              className="dot-label"
            >
              {p.v}
            </text>
          </g>
        ))}

        {labels.map((lab, i) => {
          const x = pad + i * xStep;
          return (
            <text
              key={lab}
              x={x}
              y={h - 8}
              textAnchor="middle"
              className="x-label"
            >
              {lab}
            </text>
          );
        })}
      </svg>
    </div>
  );
}

/** ---------- SVG Scatter Plot ---------- */
function ScatterPlot({ points, height = 160 }) {
  const w = 520;
  const h = height;

  const padL = 52; 
  const padR = 20;
  const padT = 24;
  const padB = 40;

  const xs = points.map((p) => p.bmi);
  const ys = points.map((p) => p.ac);

  const minX = Math.min(...xs, 0);
  const maxX = Math.max(...xs, 40);
  const minY = Math.min(...ys, 0);
  const maxY = Math.max(...ys, 1);

  const scaleX = (x) =>
    padL + ((x - minX) / (maxX - minX || 1)) * (w - padL - padR);

  const scaleY = (y) =>
    padT + (1 - (y - minY) / (maxY - minY || 1)) * (h - padT - padB);

  const xAxisY = h - padB;
  const yAxisX = padL;

  return (
    <div className="chart-card">
      <div className="chart-head">
        <h3>AC vs BMI</h3>
        <p className="chart-sub">Scatter plot from saved records</p>
      </div>

      <svg className="chart-svg" viewBox={`0 0 ${w} ${h}`} role="img" aria-label="AC vs BMI scatter plot">
        {/* Axes */}
        <line x1={yAxisX} y1={xAxisY} x2={w - padR} y2={xAxisY} className="axis-line" />
        <line x1={yAxisX} y1={padT} x2={yAxisX} y2={xAxisY} className="axis-line" />

        {/* X ticks */}
        {[20, 25, 30, 35, 40].map((t) => (
          <g key={t}>
            <line x1={scaleX(t)} y1={xAxisY} x2={scaleX(t)} y2={xAxisY + 6} className="tick" />
            <text x={scaleX(t)} y={xAxisY + 18} textAnchor="middle" className="x-label">
              {t}
            </text>
          </g>
        ))}

        {/* Y ticks */}
        {[minY, (minY + maxY) / 2, maxY].map((t, i) => (
          <g key={i}>
            <line x1={yAxisX - 6} y1={scaleY(t)} x2={yAxisX} y2={scaleY(t)} className="tick" />
            <text x={yAxisX - 10} y={scaleY(t) + 4} textAnchor="end" className="y-label">
              {formatMaybe(t, 1)}
            </text>
          </g>
        ))}

        {/* Points */}
        {points.map((p, idx) => (
          <circle key={idx} cx={scaleX(p.bmi)} cy={scaleY(p.ac)} r="4" className="dot" />
        ))}

        {/* Axis titles*/}
        <text x={(yAxisX + (w - padR)) / 2} y={h - 6} textAnchor="middle" className="axis-title">
          BMI
        </text>

        <text
          x={16}
          y={(padT + xAxisY) / 2}
          textAnchor="middle"
          className="axis-title"
          transform={`rotate(-90 16 ${(padT + xAxisY) / 2})`}
        >
          AC (cm)
        </text>
      </svg>
    </div>
  );
}


export default function Dashboard() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [raw, setRaw] = useState([]);
  const [err, setErr] = useState("");

  useEffect(() => {
    async function load() {
      setLoading(true);
      setErr("");
      try {
        const data = await fetchPredictionRecords();
        setRaw(Array.isArray(data) ? data : []);
      } catch (e) {
        console.error(e);
        setErr(e.message || "Failed to load dashboard data.");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const normalized = useMemo(() => {
    return raw
      .map((item) => {
        const createdAt = item.created_at ? new Date(item.created_at) : null;

        return {
          id: item.id,
          patientRN: item.patient_rn ?? "",
          imageFilename: item.image_filename ?? "",
          bmi: safeNumber(item.bmi),
          scale: safeNumber(item.scale),
          ac: parseAcValueToNumber(item.ac_result),
          createdAt,

          processingTime: safeNumber(item.processing_time_sec),
        };
      })
      .filter((x) => x.createdAt && !Number.isNaN(x.createdAt.getTime()));
  }, [raw]);

  const kpis = useMemo(() => {
    const total = normalized.length;

    const now = new Date();
    const weekStart = startOfWeekMonday(now);
    const weekEnd = addDays(weekStart, 7);

    const thisWeek = normalized.filter(
      (r) => r.createdAt >= weekStart && r.createdAt < weekEnd
    ).length;

    const obese = normalized.filter((r) => (r.bmi ?? 0) >= 30).length;
    const nonObese = total - obese;

    const procVals = normalized
      .map((r) => r.processingTime)
      .filter((v) => v !== null && Number.isFinite(v));

    const avgProc =
      procVals.length > 0
        ? procVals.reduce((a, b) => a + b, 0) / procVals.length
        : null;

    return { total, thisWeek, obese, nonObese, avgProc };
  }, [normalized]);

  const chartWeek = useMemo(() => {
    const now = new Date();
    const weekStart = startOfWeekMonday(now);

    const days = Array.from({ length: 7 }, (_, i) => addDays(weekStart, i));
    const labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    const values = days.map((d) =>
      normalized.filter((r) => sameDay(r.createdAt, d)).length
    );

    return { labels, values };
  }, [normalized]);

  const scatterPoints = useMemo(() => {
    return normalized
      .map((r) => ({ bmi: r.bmi, ac: r.ac }))
      .filter((p) => p.bmi !== null && p.ac !== null);
  }, [normalized]);

  const recentTop5 = useMemo(() => {
    return [...normalized].sort((a, b) => b.createdAt - a.createdAt).slice(0, 5);
  }, [normalized]);

  return (
    <div className="dash2">
      <div className="dash2-top">
        <div>
          <h1 className="dash2-title">Dashboard</h1>
          <p className="dash2-sub">Monitoring overview from saved records.</p>
        </div>

        <div className="dash2-actions">
          <button className="btn-primary" onClick={() => navigate("/prediction")}>
            Go to Prediction
          </button>
          <button className="btn-ghost" onClick={() => navigate("/records")}>
            Go to Records
          </button>
          <button
            className="btn-outline2"
            type="button"
            onClick={() =>
              window.open(
                "https://drive.google.com/file/d/1tlYFHIbFV41_AAIoq77dvyxuTd0bk-h9/view",
                "_blank",
                "noopener,noreferrer"
              )
            }
            title="Open FetoVision User Guide (PDF)"
          >
            View Guide
          </button>
        </div>
      </div>

      {loading ? (
        <div className="dash2-loading">Loading dashboard data...</div>
      ) : err ? (
        <div className="dash2-error">{err}</div>
      ) : (
        <>
          {/* KPIs */}
          <div className="kpi-grid">
            <div className="kpi-card">
              <p className="kpi-label">Total predictions</p>
              <h2 className="kpi-value">{kpis.total}</h2>
              <p className="kpi-foot">saved in records</p>
            </div>

            <div className="kpi-card">
              <p className="kpi-label">Predictions this week</p>
              <h2 className="kpi-value">{kpis.thisWeek}</h2>
              <p className="kpi-foot">Mon → Sun</p>
            </div>

            <div className="kpi-card">
              <p className="kpi-label">Avg processing time</p>
              <h2 className="kpi-value">{formatMaybe(kpis.avgProc, 2)}s</h2>
              <p className="kpi-foot">from saved records</p>
            </div>

            <div className="kpi-card">
              <p className="kpi-label">Obese vs Non-obese</p>
              <h2 className="kpi-value">
                {kpis.obese} <span className="kpi-split">/</span> {kpis.nonObese}
              </h2>
              <p className="kpi-foot">BMI ≥ 30 / BMI &lt; 30</p>
            </div>
          </div>

          {/* Charts */}
          <div className="chart-grid">
            <LineChart labels={chartWeek.labels} values={chartWeek.values} />
            <ScatterPlot points={scatterPoints} />
          </div>

          {/* Recent saved records table */}
          <div className="table-card">
            <div className="table-head">
              <h2>Recent saved records</h2>
              <button className="link-btn2" onClick={() => navigate("/records")}>
                View all
              </button>
            </div>

            <div className="table-wrap">
              <table className="dash-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Patient RN</th>
                    <th>Image filename</th>
                    <th>BMI</th>
                    <th>AC (cm)</th>
                    <th>Created at</th>
                    <th>Processing time</th>
                  </tr>
                </thead>
                <tbody>
                  {recentTop5.length === 0 ? (
                    <tr>
                      <td colSpan="7" className="empty-row">
                        No saved records yet.
                      </td>
                    </tr>
                  ) : (
                    recentTop5.map((r, idx) => (
                      <tr key={r.id ?? idx}>
                        <td>{idx + 1}</td>
                        <td>{r.patientRN || "—"}</td>
                        <td className="mono">{r.imageFilename || "—"}</td>
                        <td>{r.bmi ?? "—"}</td>
                        <td>{r.ac !== null ? formatMaybe(r.ac, 2) : "—"}</td>
                        <td>{r.createdAt ? r.createdAt.toLocaleString() : "—"}</td>
                        <td>
                          {r.processingTime !== null
                            ? `${formatMaybe(r.processingTime, 2)}s`
                            : "—"}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

            <div className="table-note">
              Note: Dashboard shows database records only. For unsaved runs, check
              Home → Recent activity.
            </div>
          </div>
        </>
      )}
    </div>
  );
}
