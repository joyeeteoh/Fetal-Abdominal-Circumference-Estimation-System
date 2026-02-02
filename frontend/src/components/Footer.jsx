export default function Footer() {
  return (
    <footer id="contact" className="footer">
      <div className="footer-top">
        <div className="footer-brand">FetoVision</div>
        <div className="footer-links">
          <a href="#about">About us</a>
          <a href="#features">Features</a>
          <a href="#help">Help Center</a>
          <a href="#contact">Contact us</a>
          <a href="#faqs">FAQs</a>
        </div>
      </div>

      <div className="newsletter">
        <p>Subscribe to our newsletter</p>
        <div className="newsletter-input">
          <input type="email" placeholder="Input your email" />
          <button>Subscribe</button>
        </div>
      </div>

      <p className="footer-bottom">© 2025 FetoVision. All rights reserved.</p>
    </footer>
  );
}
