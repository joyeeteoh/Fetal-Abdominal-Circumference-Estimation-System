import React, { useEffect, useMemo, useState } from "react";
import { fetchPredictionRecords } from "../api";
import "../styles/Records.css";
import RecordsDetailModal from "../components/RecordsDetailModal";

export default function Records() {
  const [records, setRecords] = useState([]);
  const [selectedRecord, setSelectedRecord] = useState(null);
  const [detailOpen, setDetailOpen] = useState(false);

  const [searchRN, setSearchRN] = useState("");

  useEffect(() => {
    async function load() {
      try {
        const data = await fetchPredictionRecords();
        const normalized = data.map((item) => ({
          id: item.id,
          patientRN: item.patient_rn,
          imageFilename: item.image_filename,
          bmi: item.bmi,
          scale: item.scale,
          segmentedResult: item.segmented_image_url ? "Available" : "Not available",
          acResult: item.ac_result,
          date: item.created_at,
          inputImage: item.input_image_url || "",         // Input image URL
          segmentedImage: item.segmented_image_url || "", // Segmented mask URL
        }));
        setRecords(normalized);
      } catch (err) {
        console.error(err);
      }
    }
    load();
  }, []);

  function handleRowClick(rec) {
    setSelectedRecord(rec);
    setDetailOpen(true);
  }

  const filteredRecords = useMemo(() => {
    if (!searchRN.trim()) return records;
    return records.filter((r) =>
      String(r.patientRN).toLowerCase().includes(searchRN.toLowerCase())
    );
  }, [records, searchRN]);

  return (
    <div className="records-page">
      <div className="records-topbar">
        <div>
          <h1>Prediction Records</h1>
          <p className="records-sub">Click any row to view your saved prediction results.</p>
        </div>

        <input
          type="text"
          className="records-search"
          placeholder="Search by Patient RN…"
          value={searchRN}
          onChange={(e) => setSearchRN(e.target.value)}
        />
      </div>

      <div className="records-table-wrap">
        <table className="records-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Patient&apos;s RN</th>
              <th>Image Filename</th>
              <th>BMI</th>
              <th>Scale</th>
              <th>Segmented Result</th>
              <th>AC Result</th>
              <th>Date</th>
            </tr>
          </thead>
          <tbody>
            {filteredRecords.length === 0 ? (
              <tr>
                <td colSpan="8" className="empty-row">
                  No matching records.
                </td>
              </tr>
            ) : (
              filteredRecords.map((item, idx) => (
                <tr
                  key={item.id}
                  className="records-row"
                  onClick={() => handleRowClick(item)}
                >
                  <td>{idx + 1}</td>
                  <td>{item.patientRN}</td>
                  <td>{item.imageFilename}</td>
                  <td>{item.bmi}</td>
                  <td>{item.scale}</td>
                  <td>{item.segmentedResult}</td>
                  <td>{item.acResult}</td>
                  <td>{item.date ? new Date(item.date).toLocaleString() : "—"}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <RecordsDetailModal
        open={detailOpen}
        onClose={() => setDetailOpen(false)}
        record={selectedRecord}
      />
    </div>
  );
}
