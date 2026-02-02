import React, { useState, useEffect } from "react";
import {
  Routes,
  Route,
  Navigate,
  Outlet,
  useNavigate,
  NavLink,
} from "react-router-dom";
import { getCurrentUser } from "./api";

import Navbar from "./components/Navbar";

// public pages
import LandingPage from "./pages/LandingPage";
import Auth from "./pages/Auth";

// private pages
import AppHome from "./pages/Home";
import Dashboard from "./pages/Dashboard";
import Prediction from "./pages/Prediction";
import Records from "./pages/Records";
import Profile from "./pages/Profile";

import "./styles/Shell.css";

// ---- guard ----
function PrivateRoute({ children }) {
  const token = localStorage.getItem("token");
  return token ? children : <Navigate to="/login" replace />;
}

// ---- public layout ----
function PublicLayout() {
  return (
    <>
      <Navbar />
      <main style={{ marginTop: "64px" }}>
        <Outlet />
      </main>
    </>
  );
}

// ---- private layout ----
function PrivateLayout() {
  const navigate = useNavigate();
  const [collapsed, setCollapsed] = useState(false);

  const [user, setUser] = useState({
    username: localStorage.getItem("username") || "",
    email: localStorage.getItem("email") || "",
  });

  useEffect(() => {
    async function fetchMe() {
      try {
        const me = await getCurrentUser();
        const name =
          me.username ||
          me.full_name ||
          (me.email ? me.email.split("@")[0] : "User");

        const formattedName =
          name.charAt(0).toUpperCase() + name.slice(1).toLowerCase();

        setUser({ username: formattedName, email: me.email || "" });

        localStorage.setItem("username", formattedName);
        if (me.email) localStorage.setItem("email", me.email);
      } catch (err) {
        console.log("cannot fetch current user", err);
      }
    }
    fetchMe();
  }, [navigate]);

  const firstLetter = (user.username || "U").charAt(0).toUpperCase();

  function handleLogout() {
    localStorage.removeItem("token");
    localStorage.removeItem("username");
    localStorage.removeItem("email");
    navigate("/login");
  }

  const linkClass = ({ isActive }) => `sb-link ${isActive ? "active" : ""}`;

  return (
    <div className={`app-shell2 ${collapsed ? "collapsed" : ""}`}>
      {/* Sidebar */}
      <aside className="sb">
        <div className="sb-brandRow">
          {/* brand */}
          <div className="sb-brand">
            <div className="sb-brandText">FetoVision</div>
          </div>

          <button
            className="sb-collapseBtn"
            onClick={() => setCollapsed((p) => !p)}
            aria-label="Toggle sidebar"
            title="Collapse"
            type="button"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <path d="M15.41 16.59 10.83 12l4.58-4.59L14 6l-6 6 6 6z" />
            </svg>
          </button>
        </div>

        <div className="sb-sectionTitle">General</div>
        <nav className="sb-nav">
          <NavLink to="/app" className={linkClass}>
            <div className="sb-ico">
              <i className="fas fa-house" />
            </div>
            <span>Home</span>
          </NavLink>

          <NavLink to="/dashboard" className={linkClass}>
            <div className="sb-ico">
              <i className="fas fa-chart-line" />
            </div>
            <span>Dashboard</span>
          </NavLink>

          <NavLink to="/prediction" className={linkClass}>
            <div className="sb-ico">
              <i className="fas fa-flask" />
            </div>
            <span>Prediction</span>
          </NavLink>

          <NavLink to="/records" className={linkClass}>
            <div className="sb-ico">
              <i className="fas fa-folder-open" />
            </div>
            <span>Records</span>
          </NavLink>
        </nav>

        <div className="sb-sectionTitle">Support</div>
        <nav className="sb-nav">
          <NavLink to="/profile" className={linkClass}>
            <div className="sb-ico">
              <i className="fas fa-user" />
            </div>
            <span>Profile</span>
          </NavLink>
        </nav>

        <div className="sb-spacer" />

        <div className="sb-userCard">
          <div className="sb-avatar">{firstLetter}</div>
          <div className="sb-userMeta">
            <p className="sb-userName">{user.username || "Clinician"}</p>
            <p className="sb-userEmail">{user.email || "—"}</p>
          </div>
        </div>

        <button className="sb-logout" onClick={handleLogout} type="button">
          <span className="sb-logoutIco">
            <i className="fas fa-right-from-bracket" />
          </span>
          <span className="sb-logoutText">Log out</span>
        </button>
      </aside>

      {/* Content */}
      <div className="shellContent">
        <main className="pageWrap">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      {/* public */}
      <Route element={<PublicLayout />}>
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" element={<Auth />} />
        <Route path="/register" element={<Auth />} />
      </Route>

      {/* private */}
      <Route
        element={
          <PrivateRoute>
            <PrivateLayout />
          </PrivateRoute>
        }
      >
        <Route path="/app" element={<AppHome />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/prediction" element={<Prediction />} />
        <Route path="/records" element={<Records />} />
        <Route path="/profile" element={<Profile />} />
      </Route>

      {/* fallback */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
