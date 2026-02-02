import React from "react";
import "../styles/RecordsModal.css";

export default function RecordsDetailModal({ open, onClose, record }) {
  if (!open || !record) return null;

  return (
    <div className="records-modal-backdrop" onClick={onClose}>
      <div className="records-modal" onClick={(e) => e.stopPropagation()}>
        <div className="records-modal-header">
          <h2>Saved Result</h2>
          <button className="records-modal-close-btn" onClick={onClose}>×</button>
        </div>

        <div className="records-modal-body">
          <div className="records-info-grid">
            <div>
              <p className="label">Patient&apos;s RN</p>
              <p className="val">{record.patientRN}</p>
            </div>
            <div>
              <p className="label">Image Filename</p>
              <p className="val">{record.imageFilename || "—"}</p>
            </div>
            <div>
              <p className="label">BMI</p>
              <p className="val">{record.bmi || "—"}</p>
            </div>
            <div>
              <p className="label">Scale</p>
              <p className="val">{record.scale || "—"}</p>
            </div>
            <div>
              <p className="label">AC Result</p>
              <p className="val">{record.acResult || "—"}</p>
            </div>
            <div>
              <p className="label">Date</p>
              <p className="val">
                {record.date ? new Date(record.date).toLocaleString() : "—"}
              </p>
            </div>
          </div>

          <div className="records-images">
            <div className="img-block">
              <p className="label">Input Image</p>
              {record.inputImage ? (
                <img src={record.inputImage} alt="input" />
              ) : (
                <div className="img-placeholder">No preview</div>
              )}
            </div>

            <div className="img-block">
              <p className="label">Segmented Result</p>
              {record.segmentedImage ? (
                <img src={record.segmentedImage} alt="segmented" />
              ) : (
                <div className="img-placeholder">No segmented image</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
