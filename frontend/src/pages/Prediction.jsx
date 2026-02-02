import React, { useEffect, useRef, useState } from "react";
import "../styles/Prediction.css";

import { runPrediction, savePredictionRecords, uploadImage } from "../api";
import SaveResultModal from "../components/SaveResultModal";
import { addPredictionRunToHistory, markHistoryRunSaved } from "../utils/PredictionHistory";

function makeRunId() {
  return `${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function isImageFile(file) {
  if (!file) return false;
  const t = (file.type || "").toLowerCase();
  return t.includes("png") || t.includes("jpeg") || t.includes("jpg");
}

export default function Prediction() {
  const fileInputRef = useRef(null);

  // upload + preview
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(""); 
  const [uploadedUrl, setUploadedUrl] = useState(""); 
  const [serverFilename, setServerFilename] = useState(""); 

  // output
  const [segmentedUrl, setSegmentedUrl] = useState(""); 
  const [result, setResult] = useState(null); 
  const [processingTime, setProcessingTime] = useState(null);

  // form
  const [scale, setScale] = useState("");
  const [bmi, setBmi] = useState("");

  // state
  const [errors, setErrors] = useState({});
  const [isRunning, setIsRunning] = useState(false);

  // save
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [currentRunId, setCurrentRunId] = useState(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  async function processFile(file) {
    if (!isImageFile(file)) {
      setErrors((p) => ({ ...p, image: "Only PNG or JPG is allowed." }));
      return;
    }

    if (previewUrl) URL.revokeObjectURL(previewUrl);

    const localUrl = URL.createObjectURL(file);
    setSelectedFile(file);
    setPreviewUrl(localUrl);
    setErrors((p) => ({ ...p, image: "" }));

    try {
      const resp = await uploadImage(file); 
      setUploadedUrl(resp.url);
      setServerFilename(resp.filename || "");
    } catch (err) {
      console.error(err);
      setErrors((p) => ({ ...p, image: err.message || "Upload failed." }));
    }
  }

  function handleFileChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    processFile(file);
  }

  function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    processFile(file);
  }

  function validate() {
    const newErr = {};

    if (!selectedFile) newErr.image = "Image is required.";
    if (!scale) newErr.scale = "Scale is required.";
    if (!bmi) newErr.bmi = "BMI is required.";

    if (scale && (!Number.isFinite(Number(scale)) || Number(scale) <= 0)) {
      newErr.scale = "Scale must be a positive number.";
    }
    if (bmi && (!Number.isFinite(Number(bmi)) || Number(bmi) <= 0)) {
      newErr.bmi = "BMI must be a positive number.";
    }

    if (!uploadedUrl) newErr.image = newErr.image || "Upload is not completed yet.";

    setErrors(newErr);
    return Object.keys(newErr).length === 0;
  }

  async function handleRun() {
    if (!validate()) return;

    setIsRunning(true);
    setErrors((p) => ({ ...p, run: "" }));

    try {
      const resp = await runPrediction({
        input_image_url: uploadedUrl, // Cloudinary URL of uploaded input image
        image_filename: serverFilename || selectedFile?.name || "uploaded.png",
        scale: Number(scale),
        bmi: Number(bmi),
      });

      const pt = resp.processing_time_sec != null ? Number(resp.processing_time_sec) : null;
      setProcessingTime(Number.isFinite(pt) ? pt : null);

      const raw = resp.ac_result;
      const cleaned = String(raw ?? "").replace(/\s*cm/i, "").trim();
      const maybeNum = parseFloat(cleaned);
      const finalValue = Number.isFinite(maybeNum) ? maybeNum.toFixed(2) : cleaned || "—";

      setResult({ message: resp.message, value: finalValue });

      const segUrl = resp.segmented_image_url || "";
      setSegmentedUrl(segUrl); // Cloudinary URL of segmented output image

      const runId = makeRunId();
      setCurrentRunId(runId);

      addPredictionRunToHistory({
        id: runId,
        createdAt: new Date().toISOString(),
        imageFilename: serverFilename || selectedFile?.name || "",
        bmi: Number(bmi),
        scale: Number(scale),
        acResult: finalValue,
        inputImageUrl: uploadedUrl,
        segmentedImageUrl: segUrl,
        processing_time_sec: Number.isFinite(pt) ? pt : null,
        saved: false,
        patientRN: "",
        status: "Completed",
      });
    } catch (err) {
      console.error(err);
      setErrors((p) => ({ ...p, run: err.message || "Failed to run prediction." }));
    } finally {
      setIsRunning(false);
    }
  }

  function handleReset() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);

    setSelectedFile(null);
    setPreviewUrl("");
    setUploadedUrl("");
    setServerFilename("");

    setResult(null);
    setSegmentedUrl("");
    setProcessingTime(null);

    setScale("");
    setBmi("");
    setErrors({});
    setIsRunning(false);

    setShowSaveModal(false);
    setCurrentRunId(null);

    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function handleOpenSave() {
    if (!result) {
      setErrors((p) => ({ ...p, run: "Run estimation first, then you can save the record." }));
      return;
    }
    setShowSaveModal(true);
  }

  async function handleSaveRecord(record) {
    try {
      await savePredictionRecords({
        image_filename: serverFilename || record.imageFilename,
        scale: Number(record.scale),
        bmi: Number(record.bmi),
        patient_rn: record.patientRN,
        ac_result: record.output,
        input_image_url: uploadedUrl,       // Persist input image URL
        segmented_image_url: segmentedUrl,  // Persist output mask URL
        processing_time_sec: processingTime != null ? Number(processingTime) : null,
      });

      if (currentRunId) {
        markHistoryRunSaved(currentRunId, record.patientRN);
      }
      setShowSaveModal(false);
    } catch (err) {
      console.error(err);
      setErrors((p) => ({ ...p, run: err.message || "Failed to save record." }));
    }
  }

  const canRun = !!selectedFile && !!scale && !!bmi && !isRunning;
  const canSave = !!result;

  return (
    <div className="pred2">
      <div className="pred2-top">
        <div>
          <h1 className="pred2-title">Prediction</h1>
          <p className="pred2-sub">Upload an ultrasound image and estimate fetal abdominal circumference (AC).</p>
        </div>

        <div className="pred2-actions">
          <button className="btn-outline2" type="button" onClick={handleReset} disabled={isRunning}>
            Reset
          </button>
          <button className="btn-primary" type="button" onClick={handleOpenSave} disabled={!canSave}>
            Save record
          </button>
        </div>
      </div>

      <div className="pred2-grid">
        <div className="predCard">
          <div className="predCard-head">
            <h2 className="predCard-title">Input</h2>
            <p className="predCard-sub">Unannotated image + parameters.</p>
          </div>

          <div
            className={`dropBox ${errors.image ? "has-error" : ""} ${previewUrl ? "has-file" : ""}`}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => !previewUrl && fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
          >
            {!previewUrl ? (
              <div className="dropBox-inner">
                <div className="dropIcon">
                  <i className="fas fa-cloud-upload-alt" />
                </div>
                <div className="dropTitle">Drop image here</div>
                <div className="dropDesc">or click to upload (PNG / JPG)</div>

                <button
                  type="button"
                  className="btn-ghost"
                  onClick={(e) => {
                    e.stopPropagation();
                    fileInputRef.current?.click();
                  }}
                >
                  Upload file
                </button>
              </div>
            ) : (
              <div className="filePreview">
                <img className="filePreview-img" src={previewUrl} alt="Uploaded preview" />
                <div className="filePreview-meta">
                  <div className="filePreview-name">{selectedFile?.name || "uploaded"}</div>
                  <div className="filePreview-actions">
                    <button
                      type="button"
                      className="btn-outline2"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleReset();
                      }}
                      disabled={isRunning}
                    >
                      Remove
                    </button>
                  </div>
                </div>
              </div>
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept="image/png, image/jpeg"
              hidden
              onChange={handleFileChange}
            />
          </div>

          {errors.image ? <div className="fieldError">{errors.image}</div> : null}

          <div className="formGrid">
            <div className="field">
              <label className="label" htmlFor="scale">
                Scale (pixel-to-cm)
              </label>
              <input
                id="scale"
                type="number"
                className={`input ${errors.scale ? "has-error" : ""}`}
                placeholder="e.g. 15.88"
                value={scale}
                onChange={(e) => {
                  setScale(e.target.value);
                  if (e.target.value) setErrors((p) => ({ ...p, scale: "" }));
                }}
              />
              {errors.scale ? <div className="fieldError">{errors.scale}</div> : null}
            </div>

            <div className="field">
              <label className="label" htmlFor="bmi">
                BMI
              </label>
              <input
                id="bmi"
                type="number"
                className={`input ${errors.bmi ? "has-error" : ""}`}
                placeholder="e.g. 33.9"
                value={bmi}
                onChange={(e) => {
                  setBmi(e.target.value);
                  if (e.target.value) setErrors((p) => ({ ...p, bmi: "" }));
                }}
              />
              {errors.bmi ? <div className="fieldError">{errors.bmi}</div> : null}
            </div>
          </div>

          {errors.run ? <div className="inlineAlert">{errors.run}</div> : null}

          <button className="runBtn" type="button" onClick={handleRun} disabled={!canRun}>
            {isRunning ? "Running estimation..." : "Run estimation"}
          </button>

          <div className="hintRow">
            <span className="hintDot" />
            <span className="hintText">
              Tip: Use a clear fetal abdominal frame. Wrong scale will distort AC.
            </span>
          </div>
        </div>

        <div className="predCard">
          <div className="predCard-head predCard-headRow">
            <div>
              <h2 className="predCard-title">Output</h2>
              <p className="predCard-sub">Estimated AC and segmentation mask.</p>
            </div>

            <div className="procChip">
              {processingTime != null ? `Processing: ${Number(processingTime).toFixed(2)}s` : "—"}
            </div>
          </div>

          <div className="segCard">
            <div className="segHead">
              <div>
                <h3 className="segTitle">Abdominal Circumference (cm)</h3>
              </div>
            </div>

            <div className="acBody">
              {result ? (
                <div className="acValue">
                  {result.value}
                  <span className="acUnit"> cm</span>
                </div>
              ) : (
                <div className="acEmpty">Run estimation to view AC result.</div>
              )}
            </div>
          </div>

          <div className="segCard">
            <div className="segHead">
              <div>
                <h3 className="segTitle">Segmentation Mask</h3>
              </div>

              {segmentedUrl ? (
                <a className="segLink" href={segmentedUrl} target="_blank" rel="noreferrer">
                  Open
                </a>
              ) : (
                <span className="segLink disabled">Open</span>
              )}
            </div>

            <div className={`segBody ${segmentedUrl ? "has-img" : ""}`}>
              {segmentedUrl ? (
                <img className="segImg" src={segmentedUrl} alt="Segmented result" />
              ) : (
                <div className="segEmpty">No segmented image yet. Run estimation to generate the mask.</div>
              )}
            </div>
          </div>

          <div className="outActions">
            <button className="btn-ghost" type="button" onClick={handleOpenSave} disabled={!canSave}>
              Save this result
            </button>
            <button className="btn-outline2" type="button" onClick={handleReset} disabled={isRunning}>
              Predict another
            </button>
          </div>
        </div>
      </div>

      <SaveResultModal
        open={showSaveModal}
        onClose={() => setShowSaveModal(false)}
        initialData={{
          imageFilename: serverFilename || (selectedFile ? selectedFile.name : ""),
          scale,
          bmi,
          output: result ? result.value : "",
          processing_time_sec: processingTime,
        }}
        onSave={handleSaveRecord}
      />
    </div>
  );
}
