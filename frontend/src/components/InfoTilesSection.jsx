export default function InfoTilesSection() {
  return (
    <section className="info-tiles">
      <div className="info-tile">
        <div className="info-icon">
          <i className="fas fa-lock"></i>
        </div>
        <h3>Data Security</h3>
        <p>Your uploaded data is securely stored and processed with encryption.</p>
      </div>

      <div className="info-tile">
        <div className="info-icon">
          <i className="fas fa-gear"></i>
        </div>
        <h3>System Settings</h3>
        <p>Manage your account, preferences, and system configurations.</p>
      </div>

      <div className="info-tile">
        <div className="info-icon">
          <i className="fas fa-circle-info"></i>
        </div>
        <h3>Help & Info</h3>
        <p>Learn more about FetoVision and how to use the platform effectively.</p>
      </div>
    </section>
  );
}
