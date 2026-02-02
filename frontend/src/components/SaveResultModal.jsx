import React, { useState, useEffect } from "react";
import "../styles/SaveModal.css";

export default function SaveResultModal({ open, onClose, initialData, onSave }) {
  const [patientRN, setPatientRN] = useState("");

  useEffect(() => {
    if (open) {
      setPatientRN("");
    }
  }, [open]);

  if (!open) return null;

  function handleSubmit(e) {
    e.preventDefault();
    if (!patientRN) {
      alert("Please enter patient's registration number.");
      return;
    }

    onSave({
      imageFilename: initialData?.imageFilename || "",
      scale: initialData?.scale || "",
      bmi: initialData?.bmi || "",
      output: initialData?.output || "",
      patientRN,
    });
  }

  return (
    <div className="save-modal-backdrop" onClick={onClose}>
      <div className="save-modal" onClick={(e) => e.stopPropagation()}>
        <div className="save-modal-header">
          <h2>Save Result</h2>
          <button className="close-btn" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        <form className="save-modal-body" onSubmit={handleSubmit}>
          <label className="save-label">
            Image:
            <input
              type="text"
              value={initialData?.imageFilename || ""}
              readOnly
              className="save-input read-only"
            />
          </label>

          <label className="save-label">
            Scale:
            <input
              type="text"
              value={initialData?.scale || ""}
              readOnly
              className="save-input read-only"
            />
          </label>

          <label className="save-label">
            BMI Value:
            <input
              type="text"
              value={initialData?.bmi || ""}
              readOnly
              className="save-input read-only"
            />
          </label>

          <label className="save-label">
            Patient&apos;s Registration Number:
            <input
              type="text"
              value={patientRN}
              onChange={(e) => setPatientRN(e.target.value)}
              className="save-input"
              required
            />
          </label>

          <label className="save-label">
            Output:
            <input
              type="text"
              value={initialData?.output || ""}
              readOnly
              className="save-input read-only"
            />
          </label>

          <button type="submit" className="save-btn-primary">
            Save
          </button>
        </form>
      </div>
    </div>
  );
}
