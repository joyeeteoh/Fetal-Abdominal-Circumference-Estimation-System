import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/Home.css";

import { getCurrentUser } from "../api";
import { getPredictionHistory } from "../utils/PredictionHistory";

function formatPrettyDate(d = new Date()) {
  return d.toLocaleDateString("en-MY", {
    weekday: "long",
    day: "numeric",
    month: "long",
    year: "numeric",
  });
}

function formatRunDate(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString("en-MY");
}

function toDisplayName(me) {
  const name =
    me?.username || (me?.email ? me.email.split("@")[0] : "User") || "User";
  const s = String(name);
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export default function Home() {
  const navigate = useNavigate();

  const [username, setUsername] = useState(
    localStorage.getItem("username") || "User"
  );

  // Local recent runs history (saved + not saved)
  const [history, setHistory] = useState([]);

  // View all toggle (5 vs 10)
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    async function loadMe() {
      try {
        const me = await getCurrentUser();
        const display = toDisplayName(me);
        setUsername(display);
        localStorage.setItem("username", display);
      } catch {
      }
    }

    loadMe();

    setHistory(getPredictionHistory());

    const onStorage = (e) => {
      if (e.key === "prediction_history_v1") {
        setHistory(getPredictionHistory());
      }
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const rows = useMemo(() => {
    const arr = Array.isArray(history) ? history : [];
    const sorted = [...arr].sort(
      (a, b) => new Date(b.createdAt || 0) - new Date(a.createdAt || 0)
    );
    return showAll ? sorted.slice(0, 10) : sorted.slice(0, 5);
  }, [history, showAll]);

  return (
    <div className="home2">
      {/* 1) Welcome + date */}
      <div className="home2-top">
        <div>
          <h1 className="home2-title">Hi, {username}</h1>
          <p className="home2-sub">
            Welcome back to FetoVision! Ready to analyze some ultrasound images?
          </p>
        </div>

        <div className="home2-dateChip">{formatPrettyDate(new Date())}</div>
      </div>

      {/* 2) Quick Actions */}
      <div className="qa-grid">
        <button className="qa-card" onClick={() => navigate("/prediction")}>
          <div className="qa-left">
            <div className="qa-icon qa-purple">
              <i className="fas fa-flask" />
            </div>
            <div className="qa-text">
              <div className="qa-title">New Prediction</div>
              <div className="qa-desc">
                Upload ultrasound image and run AC estimation.
              </div>
            </div>
          </div>
          <div className="qa-arrow">
            <i className="fas fa-angle-right" />
          </div>
        </button>

        <button className="qa-card" onClick={() => navigate("/records")}>
          <div className="qa-left">
            <div className="qa-icon qa-pink">
              <i className="fas fa-clock" />
            </div>
            <div className="qa-text">
              <div className="qa-title">View Records</div>
              <div className="qa-desc">
                View your saved records.
              </div>
            </div>
          </div>
          <div className="qa-arrow">
            <i className="fas fa-angle-right" />
          </div>
        </button>

        <button className="qa-card" onClick={() => navigate("/dashboard")}>
          <div className="qa-left">
            <div className="qa-icon qa-blue">
              <i className="fas fa-chart-line" />
            </div>
            <div className="qa-text">
              <div className="qa-title">Dashboard</div>
              <div className="qa-desc">
                View KPIs and trends from saved records.
              </div>
            </div>
          </div>
          <div className="qa-arrow">
            <i className="fas fa-angle-right" />
          </div>
        </button>

        <button
          className="qa-card"
          onClick={() =>
            window.open(
              "https://drive.google.com/file/d/1tlYFHIbFV41_AAIoq77dvyxuTd0bk-h9/view",
              "_blank"
            )
          }
        >
          <div className="qa-left">
            <div className="qa-icon qa-green">
              <i className="fas fa-book-open" />
            </div>
            <div className="qa-text">
              <div className="qa-title">View Guide</div>
              <div className="qa-desc">
                Learn workflow and best practices.
              </div>
            </div>
          </div>
          <div className="qa-arrow">
            <i className="fas fa-angle-right" />
          </div>
        </button>
      </div>

      {/* 3) Recent activity + 4) Tips */}
      <div className="home2-grid2">
        {/* Recent activity (localStorage, saved + not saved) */}
        <div className="home-card">
          <div className="home-card-head">
            <h2 className="home-card-title">Recent activity</h2>

            {/* View all toggles to show 10 runs */}
            <button
              className="linkBtn"
              type="button"
              onClick={() => setShowAll((p) => !p)}
            >
              {showAll ? "Show less" : "View all"}
            </button>
          </div>

          <div className="recent-tableWrap">
            <table className="recent-table">
              <thead>
                <tr>
                  <th>Patient&apos;s RN</th>
                  <th>AC</th>
                  <th>BMI</th>
                  <th>Date</th>
                  <th style={{ textAlign: "right" }}>Status</th>
                </tr>
              </thead>

              <tbody>
                {rows.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="emptyCell">
                      No activity yet. Click <b>New Prediction</b> to start.
                    </td>
                  </tr>
                ) : (
                  rows.map((r) => (
                    <tr key={r.id}>
                      <td>{r.patientRN ? r.patientRN : "—"}</td>
                      <td>{r.acResult ?? "—"}</td>
                      <td>{r.bmi ?? "—"}</td>
                      <td>{formatRunDate(r.createdAt)}</td>
                      <td style={{ textAlign: "right" }}>
                        <span
                          className={`statusPill ${
                            r.saved ? "saved" : "notSaved"
                          }`}
                        >
                          {r.saved ? "Saved" : "Not saved"}
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          <div className="recent-note">
            Note: To view your saved prediction results, please visit the Records page.
          </div>
        </div>

        {/* Tips */}
        <div className="home-card">
          <div className="home-card-head">
            <h2 className="home-card-title">Tips / reminders</h2>
          </div>

          <div className="tips-list">
            <div className="tipCard">
              <div className="tipIcon tip-yellow">
                <i className="fas fa-lightbulb" />
              </div>
              <div>
                <div className="tipTitle">Use a clear lateral abdominal image</div>
                <div className="tipDesc">
                  Avoid heavy annotations and ensure fetal abdomen is visible.
                </div>
              </div>
            </div>

            <div className="tipCard">
              <div className="tipIcon tip-blue">
                <i className="fas fa-ruler-combined" />
              </div>
              <div>
                <div className="tipTitle">Ensure scale is entered correctly</div>
                <div className="tipDesc">
                  Wrong pixel-to-centimeter ratio will distort AC results.
                </div>
              </div>
            </div>

            <div className="tipCard">
              <div className="tipIcon tip-pink">
                <i className="fas fa-heart" />
              </div>
              <div>
                <div className="tipTitle">
                  If segmented result looks off, re-upload image
                </div>
                <div className="tipDesc">
                  Try a clearer frame and re-run estimation for better mask output.
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
