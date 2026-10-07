import { useMemo, useState } from "react";
import "./index.css";
import "./App.css";
import { analyzeContract } from "./services/api";

function formatConfidence(value) {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "—";
}

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const clauses = result?.contract_report?.clauses ?? [];
  const reviewItems = result?.review_report?.review_items ?? [];
  const reviewSummary = result?.review_report?.review_summary ?? {};
  const contractSummary = result?.contract_report?.contract_summary ?? {};
  const riskSummary = result?.risk_scoring ?? result?.risk_summary ?? result?.contract_report?.risk_scoring ?? result?.contract_report?.risk_summary ?? {};
  const riskAssessments = useMemo(() => result?.contract_report?.risk_assessments ?? [], [result]);

  const riskByClause = useMemo(() => {
    const entries = new Map();
    riskAssessments.forEach((item) => {
      const key = item.clause_number || `${item.article || ""}:${item.title || ""}`;
      entries.set(key, item);
    });
    return entries;
  }, [riskAssessments]);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile && (selectedFile.type === "application/pdf" || selectedFile.name.toLowerCase().endsWith(".pdf"))) {
      setFile(selectedFile);
      setError("");
    } else {
      setFile(null);
      setError("Please select a PDF file.");
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      setResult(await analyzeContract(file));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="navbar">
        <div>
          <h1>AI Contract Intelligence</h1>
          <p>Contract analysis and clause-level risk indicators</p>
        </div>
        <div className="status"><span className="status-dot" />System Ready</div>
      </header>

      <main className="container">
        <section className="hero">
          <div>
            <h2>Understand your contract</h2>
            <p>Upload a PDF to review clause classifications, backend risk indicators, and clauses that need human attention.</p>
          </div>
          <div className="upload-card">
            <div className="upload-icon">↑</div>
            <h3>{file ? file.name : "Upload Contract PDF"}</h3>
            <p>{file ? "PDF selected and ready for analysis." : "Select a PDF contract to begin analysis."}</p>
            <label className="file-button">Choose PDF<input type="file" accept=".pdf,application/pdf" onChange={handleFileChange} hidden /></label>
            {error && <div className="error-message">{error}</div>}
            {file && <button className="analyze-button" onClick={handleAnalyze} disabled={loading}>{loading ? "Analyzing contract…" : "Analyze contract"}</button>}
          </div>
        </section>

        <section className={`risk-summary ${result ? "has-result" : ""}`} aria-live="polite">
          <div className="risk-summary-heading">
            <div>
              <span className="eyebrow">CONTRACT OVERVIEW</span>
              <h2>Risk Summary</h2>
              <p>{result ? "Overall risk is aggregated by the backend from its clause-level screening results." : "Run an analysis to see the risk information returned by the contract analysis service."}</p>
            </div>
            <div className={`overall-risk ${String(riskSummary.overall_risk_level || "unavailable").toLowerCase()}`}>
              <span>Overall Risk Level</span>
              <strong>{riskSummary.overall_risk_level ? `${riskSummary.overall_risk_level} RISK` : result ? "REVIEW REQUIRED" : "—"}</strong>
              {typeof riskSummary.overall_risk_score === "number" && <b className="overall-score">{riskSummary.overall_risk_score} / 100</b>}
              <small>{riskSummary.assessment_status === "REVIEW_REQUIRED" ? "Risk level is inconclusive until flagged clauses are reviewed" : riskSummary.assessment_status === "NO_RISK_ASSESSMENTS" ? "No clause risk assessments returned" : "Based on backend clause risk levels"}</small>
            </div>
          </div>
          <div className="summary-grid">
            <div className="summary-card high-count"><span>High Risk Clauses</span><strong>{riskSummary.high_risk_clause_count ?? "—"}</strong></div>
            <div className="summary-card"><span>Medium-Risk Clauses</span><strong>{riskSummary.medium_risk_clause_count ?? riskSummary.review_clause_count ?? "—"}</strong></div>
            <div className="summary-card"><span>Low Risk Clauses</span><strong>{riskSummary.low_risk_clause_count ?? "—"}</strong></div>
            <div className="summary-card"><span>Clauses analyzed</span><strong>{riskSummary.clauses_analyzed ?? contractSummary.successfully_processed ?? "—"}</strong></div>
            <div className="summary-card review"><span>Recommended for Human Review</span><strong>{riskSummary.human_review_clause_count ?? reviewSummary.clauses_requiring_review ?? "—"}</strong></div>
          </div>
          {riskSummary.major_risk_factors?.length > 0 && <div className="major-factors"><h3>Major Risk Factors</h3><ul>{riskSummary.major_risk_factors.map((factor, index) => <li key={index}>{factor}</li>)}</ul></div>}
          <p className="score-note">AI-assisted screening only; this score is not a legal determination.</p>
          {result && <div className="backend-risk-list">
            <h3>Backend clause risk assessments</h3>
            {riskAssessments.length ? riskAssessments.map((item, index) => {
              const assessment = item.risk_assessment || {};
              const level = String(item.risk_level || assessment.level || "Unknown").toUpperCase();
              const tone = ["HIGH", "ATTENTION"].includes(level) ? "high" : ["MEDIUM", "INDICATOR", "REVIEW"].includes(level) ? "medium" : "low";
              return <div className={`risk-row ${tone}`} key={`${item.clause_number || "clause"}-${index}`}>
                <span className="risk-level-pill">{level}</span>
                <span><b>{item.title || `Clause ${item.clause_number || index + 1}`}</b><small>{item.clause_type || "Type unavailable"} · {typeof item.risk_score === "number" ? `${item.risk_score}/100` : "Score unavailable"}</small></span>
                <span className="risk-row-review">{assessment.requires_human_review ? "Human review required" : "No risk review flag"}</span>
              </div>;
            }) : <p className="muted">No clause-level risk assessments were returned.</p>}
          </div>}
        </section>

        <section className="section">
          <div className="section-header"><div><h2>Clause Analysis</h2><p>Predicted clause types, model confidence, and risk or review signals from the backend.</p></div></div>
          <div className="clause-list">
            {clauses.length ? clauses.map((clause, index) => {
              const key = clause.clause_number || `${clause.article || ""}:${clause.title || ""}`;
              const riskItem = riskByClause.get(key);
              const risk = riskItem?.risk_assessment || clause.risk_assessment || {};
              const scoredLevel = riskItem?.risk_level || clause.risk_level || risk.level;
              const confidence = clause.classification_confidence;
              const title = clause.title || `Clause ${clause.clause_number || index + 1}`;
              return <article className="clause-card" key={`${key}-${index}`}>
                <div className="clause-card-top"><div><span className="clause-number">{clause.clause_number ? `Clause ${clause.clause_number}` : `Clause ${index + 1}`}</span><h3>{title}</h3></div><span className={`risk-level-pill ${String(scoredLevel || "").toLowerCase()}`}>{scoredLevel || "Risk not assessed"}{typeof (riskItem?.risk_score ?? clause.risk_score) === "number" ? ` · ${riskItem?.risk_score ?? clause.risk_score}/100` : ""}</span></div>
                <div className="clause-meta">
                  <div><span>Predicted type</span><strong>{clause.clause_type || "Unknown"}</strong></div>
                  <div><span>Model confidence</span><strong>{formatConfidence(confidence)}</strong></div>
                  <div><span>Human review</span><strong>{risk.requires_human_review ? "Required" : "Not flagged"}</strong></div>
                </div>
                {clause.classification_validation?.status === "unsupported_label" && <p className="muted">Model predicted {clause.predicted_clause_type}; text evidence did not support that category. Human review is recommended.</p>}
                {(riskItem?.risk_factors?.length || risk.reasons?.length) > 0 && <div className="clause-reasons"><span>Risk notes</span><ul>{(riskItem?.risk_factors?.length ? riskItem.risk_factors.map((factor) => factor.factor) : risk.reasons).map((reason, reasonIndex) => <li key={reasonIndex}>{reason}</li>)}</ul></div>}
                {clause.clause_text && <details><summary>View clause text</summary><p>{clause.clause_text}</p></details>}
              </article>;
            }) : <div className="table-empty">Analyze a contract to view clause-level results.</div>}
          </div>
        </section>

        <section className="section">
          <div className="section-header"><div><h2>Human Review</h2><p>Clauses listed by the backend review report for manual attention.</p></div><span className="review-count">{reviewItems.length} {reviewItems.length === 1 ? "clause" : "clauses"}</span></div>
          {reviewItems.length ? <div className="review-list">{reviewItems.map((item, index) => <article className="review-card" key={`${item.clause_number || "review"}-${index}`}>
            <div className="review-card-heading"><div><span className="clause-number">{item.clause_number ? `Clause ${item.clause_number}` : `Review item ${index + 1}`}</span><h3>{item.title || "Untitled clause"}</h3></div><span className="review-badge">Manual review</span></div>
            <div className="clause-meta"><div><span>Predicted type</span><strong>{item.predicted_clause_type || "Unknown"}</strong></div><div><span>Model confidence</span><strong>{formatConfidence(item.classification_confidence)}</strong></div></div>
            {item.clause_text && <details><summary>View clause text</summary><p>{item.clause_text}</p></details>}
          </article>)}</div> : <div className="review-empty">{result ? "No clauses were flagged for human review by the backend." : "Human-review clauses will appear here after analysis."}</div>}
        </section>
      </main>
    </div>
  );
}

export default App;
