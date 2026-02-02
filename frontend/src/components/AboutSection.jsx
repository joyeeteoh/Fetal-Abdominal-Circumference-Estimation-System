import ultrasoundImg from "../assets/ultrasound scan.jpg";

export default function AboutSection() {
  return (
    <section id="about" className="about">
      <div className="about-text">
        <p className="section-kicker">
          Empowering Clinicians with AI-Driven Fetal Care Solutions
        </p>
        <h2 className="about-title">About us</h2>
        <div className="underline" />
        <p className="about-desc">
          At FetoVision, we are dedicated to revolutionizing fetal care through the power of AI and
          advanced ultrasound imaging. Our platform leverages state-of-the-art deep learning
          algorithms to provide clinicians with accurate, real-time fetal biometry predictions,
          enhancing diagnostic precision and supporting better prenatal care.
        </p>
        <button className="btn-secondary">Learn more</button>
      </div>

      <div className="about-image">
        <img src={ultrasoundImg} alt="Ultrasound machine" />
      </div>
    </section>
  );
}
