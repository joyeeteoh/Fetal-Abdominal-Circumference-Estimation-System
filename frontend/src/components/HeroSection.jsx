import { Link } from "react-router-dom";
import heroImg from "../assets/pregnancy ultrasound image.jpg"; // <-- import image

export default function HeroSection() {
  return (
    <section
      className="hero"
      style={{ backgroundImage: `url(${heroImg})` }}
    >
      <div className="hero-overlay" />
      <div className="hero-content">
        <h1 className="hero-title">
          Enhancing Fetal Care with
          <br /> AI-Powered Imaging
        </h1>
        <p className="hero-sub">
          Empowering clinicians with advanced image processing tools
        </p>
        <Link to="/register" className="hero-btn">
          Getting started
        </Link>
      </div>
    </section>
  );
}
