const KEY = "prediction_history_v1";
const LIMIT = 10;

export function getPredictionHistory() {
  try {
    const raw = localStorage.getItem(KEY);
    const arr = raw ? JSON.parse(raw) : [];
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
}

export function addPredictionRunToHistory(run) {
  const current = getPredictionHistory();

  const next = [run, ...current]
    .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
    .slice(0, LIMIT);

  localStorage.setItem(KEY, JSON.stringify(next));
  return next;
}

export function markHistoryRunSaved(runId, patientRN) {
  const current = getPredictionHistory();
  const next = current.map((r) =>
    r.id === runId
      ? { ...r, saved: true, patientRN: patientRN || r.patientRN || "" }
      : r
  );
  localStorage.setItem(KEY, JSON.stringify(next));
  return next;
}

export function clearPredictionHistory() {
  localStorage.removeItem(KEY);
}
