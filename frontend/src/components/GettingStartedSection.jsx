import { Link } from "react-router-dom";

export default function GettingStartedSection() {
  return (
    <section className="getting-started">
      <h2>Getting started</h2>
      <p>Start leveraging AI for fetal abdominal circumference prediction.</p>
      <Link to="/register" className="btn-light">
        Sign Up
      </Link>
    </section>
  );
}
