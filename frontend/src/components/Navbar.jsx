import { Link, useLocation } from "react-router-dom";
import "../styles/Navbar.css";

export default function Navbar() {
  const location = useLocation();

  return (
    <header className="navbar">
      <div className="navbar-inner">
        {/* brand */}
        <Link to="/" className="logo">
          FetoVision
        </Link>

        {/* center links */}
        <nav className="nav-links">
          <a href="/#about">About us</a>
          <a href="/#features">Features</a>
          <a href="/#help">Help Center</a>
          <a href="/#contact">Contact us</a>
          <a href="/#faqs">FAQs</a>
        </nav>

        {/* always login on landing */}
        <div className="nav-right">
          {location.pathname !== "/login" && location.pathname !== "/register" && (
            <Link to="/login" className="btn-outline">
              Login
            </Link>
          )}
        </div>
      </div>
    </header>
  );
}
