import { Link } from "react-router-dom";
import acIcon from "../assets/ac-estimation-icon.svg"; 

export default function FeaturesSection() {
  return (
    <section id="features" className="features">
      <p className="section-kicker">Discover Our Benefits &amp; Features</p>
      <h2>Features</h2>

      <div className="feature-card">
        <div className="feature-card-body">
          <h3>Abdominal Circumference Prediction</h3>
          <p>
            Once an image is uploaded, our model will automatically process the
            ultrasound image to predict the abdominal circumference.
          </p>
          <Link to="/prediction" className="btn-secondary">
            Start Prediction
          </Link>
        </div>

        <div className="feature-card-illustration">
          <div className="feature-icon-circle">
            <img src={acIcon} alt="AC estimation icon" />
          </div>
        </div>
      </div>
    </section>
  );
}
