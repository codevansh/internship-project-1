import { useState } from "react";
import "./index.css";
import { analyzeContract } from "./services/api";

function App() {
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("");

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];

    if (selectedFile && selectedFile.type === "application/pdf") {
      setFile(selectedFile);
    } else {
      setFile(null);
      alert("Please select a PDF file.");
    }
  }

  const handleAnalyze = async () => {
    if (!file) {
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const data = await analyzeContract(file);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="navbar">
        <div>
          <h1>AI Contract Intelligence</h1>
          <p>Contract Intelligence & Risk Scoring</p>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          System Ready
        </div>
      </header>

      <main className="container">
        <section className="hero">
          <div>
            <h2>Analyze Your Contract</h2>
            <p>
              Upload a legal contract to extract clauses, entities,
              contractual indicators, and intelligence for review.
            </p>
          </div>

          <div className="upload-card">
            <div className="upload-icon">↑</div>

            <h3>
              {file ? file.name : "Upload Contract PDF"}
            </h3>

            <p>
              {file
                ? "PDF selected and ready for analysis."
                : "Select a PDF contract to begin analysis."}
            </p>

            <label className="file-button">
              Choose PDF
              <input
                type="file"
                accept=".pdf,application/pdf"
                onChange={handleFileChange}
                hidden
              />
            </label>

            {error && (
              <div className="error-message">
                {error}
              </div>
            )}

            {file && (
              <button className="analyze-button" onClick={handleAnalyze} disabled={loading}>
                {loading ? "Analyzing Contract..." : "Analyze Contract"}
              </button>
            )}
          </div>
        </section>

        <section className="section">
          <div className="section-header">
            <div>
              <h2>Contract Intelligence</h2>
              <p>
                Clause classification, extracted entities, confidence,
                and human-review information.
              </p>
            </div>
          </div>

          <div className="summary-grid">
            <div className="summary-card">
              <span>Total Clauses</span>
              <strong>
                {result?.contract_report?.contract_summary?.total_clauses ?? "—"}
              </strong>
            </div>

            <div className="summary-card">
              <span>Processed</span>
              <strong>
                {result?.contract_report?.contract_summary?.successfully_processed ?? "—"}
              </strong>
            </div>

            <div className="summary-card">
              <span>Processing Errors</span>
              <strong>
                {result?.contract_report?.contract_summary?.processing_errors ?? "—"}
              </strong>
            </div>

            <div className="summary-card review">
              <span>Requires Review</span>
              <strong>
                {result?.review_report?.review_summary?.clauses_requiring_review ?? "—"}
              </strong>
            </div>
          </div>
        </section>

        <section className="section">
          <div className="section-header">
            <div>
              <h2>Risk Intelligence</h2>
              <p>
                Contractual indicators and potentially significant language
                identified during analysis.
              </p>
            </div>
          </div>

          <div className="risk-grid">
            {result?.contract_report?.risk_assessments?.length > 0 ? (
              result.contract_report.risk_assessments.map((item, index) => {
                const assessment = item?.risk_assessment || {};
                const evidence = assessment.evidence || {};
                const phrases = evidence.contractual_phrases || {};
                const indicators = Object.entries(phrases).flatMap(
                  ([category, values]) =>
                    (Array.isArray(values) ? values : []).map((value) => ({
                      category,
                      value,
                    }))
                );
                const reasons = Array.isArray(assessment.reasons)
                  ? assessment.reasons
                  : [];
                const dates = Array.isArray(evidence.date_entities)
                  ? evidence.date_entities
                  : [];
                const confidence = item?.classification_confidence;

                return (
                  <div className="risk-card" key={index}>
                    <div className="risk-card-header">
                      <div>
                        <span className="risk-clause">
                          Clause {item?.clause_number || "Unknown"}
                        </span>
                        <h3>{item?.title || "Untitled clause"}</h3>
                      </div>
                    </div>

                    <div className="risk-indicators">
                      <div className="risk-indicator">
                        <span className="risk-category">Type</span>
                        <span className="risk-value">{item?.clause_type || "Unknown"}</span>
                      </div>
                      <div className="risk-indicator">
                        <span className="risk-category">Level</span>
                        <span className="risk-value">{assessment.level || "Unknown"}</span>
                      </div>
                      <div className="risk-indicator">
                        <span className="risk-category">Classification Confidence</span>
                        <span className="risk-value">
                          {typeof confidence === "number" ? `${(confidence * 100).toFixed(1)}%` : "Unknown"}
                        </span>
                      </div>
                      <div className="risk-indicator">
                        <span className="risk-category">Human Review</span>
                        <span className="risk-value">
                          {assessment.requires_human_review ? "Required" : "Not required"}
                        </span>
                      </div>
                    </div>

                    {reasons.length > 0 && (
                      <div className="risk-indicators">
                        <span className="risk-category">Reasons</span>
                        {reasons.map((reason, reasonIndex) => (
                          <div className="risk-indicator" key={reasonIndex}>
                            <span className="risk-value">- {reason}</span>
                          </div>
                        ))}
                      </div>
                    )}

                    {indicators.length > 0 && (
                      <div className="risk-indicators">
                        <span className="risk-category">Contractual Indicators</span>
                        {indicators.map((indicator, indicatorIndex) => (
                          <div className="risk-indicator" key={indicatorIndex}>
                            <span className="risk-category">
                              {indicator.category
                                .replace(/_/g, " ")
                                .replace(/\b\w/g, (char) => char.toUpperCase())}
                            </span>
                            <span className="risk-value">"{indicator.value}"</span>
                          </div>
                        ))}
                      </div>
                    )}

                    {dates.length > 0 && (
                      <div className="risk-indicators">
                        <span className="risk-category">Detected Dates</span>
                        {dates.map((date, dateIndex) => (
                          <div className="risk-indicator" key={dateIndex}>
                            <span className="risk-category">Date</span>
                            <span className="risk-value">"{date}"</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })
            ) : (
              <div className="table-empty">
                Analyze a contract to view risk indicators.
              </div>
            )}
          </div>
        </section>

        <section className="section">
          <div className="section-header">
            <div>
              <h2>Clause Analysis</h2>
              <p>
                AI-generated clause classifications and confidence scores.
              </p>
            </div>
          </div>

          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Clause</th>
                  <th>Title</th>
                  <th>Classification</th>
                  <th>Confidence</th>
                  <th>Review</th>
                </tr>
              </thead>

              <tbody>
                {result?.contract_report?.clauses?.length > 0 ? (
                  result.contract_report.clauses.map((clause, index) => {
                    const confidence = clause.classification_confidence ?? 0;
                    const needsReview = confidence < 0.70;

                    return (
                      <tr key={index}>
                        <td>{index + 1}</td>

                        <td>
                          {clause.clause_text?.slice(0, 100)}
                          {clause.clause_text?.length > 100 ? "..." : ""}
                        </td>

                        <td>{clause.clause_type || "Unknown"}</td>

                        <td>
                          {(confidence * 100).toFixed(1)}%
                        </td>

                        <td>
                          {needsReview ? (
                            <span className="review-badge">
                              Review
                            </span>
                          ) : (
                            <span>—</span>
                          )}
                        </td>
                      </tr>
                    );
                  })
                ) : (
                  <tr>
                    <td colSpan="5" className="table-empty">
                      Analyze a contract to view clause-level results.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
