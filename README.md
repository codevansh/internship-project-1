# AI-Powered Contract Intelligence and Risk Scoring System

## Problem and objectives

Long contracts are difficult to review consistently. This project provides an AI-assisted first-pass screening workflow that extracts contract text, segments provisions, predicts clause categories, surfaces textual indicators, and organizes results for human review. It is intended to support review, not make legal decisions.

The project objectives are to:

- Extract text and meaningful clauses from PDF contracts while preserving article and section references.
- Use the configured Legal-RoBERTa clause classifier and spaCy entity extraction to summarize contract content.
- Show the model confidence, evidence indicators, clause scores, and clauses that need human attention.
- Aggregate clause-level screening scores into a bounded contract-level summary.

## Architecture and data flow

```text
Contract PDF
-> OCR / text extraction (PyMuPDF + Tesseract)
-> line cleanup and structural clause extraction
-> Legal-RoBERTa CUAD-label classification
-> spaCy entity extraction
-> high-impact label/text consistency validation
-> contractual phrase and quantified-term indicators
-> confidence and review signals
-> explainable clause scores
-> weighted overall contract score
-> React dashboard
```

The API is implemented in `src/api/main.py`. `src/data_processing/ocr_processor.py` rasterizes PDF pages and runs Tesseract. `src/data_processing/clause_processor.py` recognizes ARTICLE, SECTION, decimal numbered headings, recitals, and schedule headings. It filters common page headers and footers, address blocks, contents entries, redaction notices, and OCR scraps. Article titles, section titles, schedule, section, article, clause number, title, and clause text are retained where identified. `src/model/inference_transformer.py` calls the configured Hugging Face tokenizer and sequence-classification model, preserves the top predictions, and applies a narrow text-evidence check to selected high-impact CUAD categories. An unsupported prediction becomes `Uncertain classification`, has reduced confidence, and is sent for human review; the check does not assign a replacement category.

`src/risk/entity_extractor.py` uses spaCy `en_core_web_sm`. `src/risk/risk_engine.py` extracts observable phrase, date, money, percentage, and duration indicators and produces a confidence-based review signal. `src/risk/risk_scoring.py` combines those signals and the validated model category. The pipeline writes clause intelligence and contract/review reports, and builds a FAISS vector store. The frontend is a React application built with Vite.

## Technology stack

- Python, FastAPI, Uvicorn, and `python-multipart`
- PyMuPDF and Tesseract OCR
- PyTorch and Hugging Face Transformers with the configured Legal-RoBERTa classifier
- spaCy `en_core_web_sm`
- Sentence Transformers embeddings with a FAISS-backed clause index
- React and Vite
- pytest for backend tests; ESLint for frontend linting

## Clause extraction and model validation

The parser recognizes `ARTICLE 1`, `ARTICLE I`, `SECTION 1.1`, numbered provisions such as `1.`, `1.1`, and `1.1.1`, and `RECITALS` / `WHEREAS` material. It retains the article, section, clause number, title, and clause text where identified. Filtering uses document structure and common artifact patterns rather than a minimum clause length, so short punctuated provisions are retained.

The transformer remains the clause classifier. A bounded validation map checks only selected high-impact CUAD labels (for example, Audit Rights, Cap On Liability, Insurance, and Covenant Not To Sue) against corresponding evidence in the clause text. If no evidence is present, the model label is retained as `predicted_clause_type` for review, while the active category becomes uncertain and confidence is reduced. Labels outside this narrow check retain the original model output. This validation is a safeguard, not a substitute classifier, and it cannot guarantee that every prediction is correct.

## Risk scoring methodology

Scores are deterministic screening scores, not probabilities of legal harm. `score_clause` begins at 10 points. The implementation may add a modest prior for selected material clause categories; points for observed limitation, unlimited-liability, damages, competition, commitment, termination, assignment, indemnification, confidentiality, or dispute phrases; points for monetary/percentage or duration evidence; and 15 points when confidence is below 0.70 or the risk engine already recommends review. Boilerplate such as “including, without limitation” and the word “assigns” in “successors and assigns” are excluded from the corresponding unlimited-liability and assignment indicators. “Minimum amount” is treated as a commitment only when nearby language indicates a purchase, quantity, volume, or commitment. Clause scores are rounded and clamped to 0–100. The returned factors and evidence support inspection of how the score was formed.

Risk thresholds are LOW (0–39), MEDIUM (40–69), and HIGH (70–100). The contract score is a risk-level-weighted mean (HIGH weight 1.75, MEDIUM 1.2, LOW 1.0), plus 5 points for each HIGH clause up to a 20-point premium, then capped at 100. This aggregation depends on scored clauses, not document length. The summary reports high-, medium-, and low-risk counts separately from the number of clauses recommended for human review. A high score is a screening signal; it does not establish that a term is legally unacceptable.

## API

Canonical backend entry point: `src.api.main:app`.

- `GET /` — service status
- `GET /health` — health status
- `POST /analyze` — accepts multipart form data with a PDF in the `file` field
- FastAPI also exposes `/docs` and `/openapi.json`

The analyze response includes `filename`, `contract_report`, `review_report`, `risk_scoring`, and `risk_summary`. The frontend posts to `/analyze` at `http://127.0.0.1:8000`.

Example response shape (values abbreviated):

```json
{
  "filename": "agreement.pdf",
  "contract_report": {
    "clauses": [{
      "article": "7",
      "clause_number": "7.5",
      "title": "Termination",
      "clause_type": "Termination For Convenience",
      "classification_confidence": 0.82,
      "clause_text": "Either party may terminate ...",
      "risk_score": 42,
      "risk_level": "MEDIUM",
      "risk_factors": [],
      "explanation": "Screening score combines clause classification, detected contractual indicators, quantified terms, and review signals."
    }],
    "risk_scoring": {"overall_risk_score": 42, "overall_risk_level": "MEDIUM"},
    "risk_summary": {}
  },
  "review_report": {"review_items": []},
  "risk_scoring": {"overall_risk_score": 42, "overall_risk_level": "MEDIUM"},
  "risk_summary": {}
}
```

## Frontend

The dashboard accepts a PDF, displays the overall risk level and score, high/medium/low clause counts, clauses analyzed, clauses recommended for human review, major risk factors, clause-level classifications and confidence, and the backend review queue. Medium risk and human-review recommendations are separate measures.

## Setup and run

Create and activate a Python environment, install backend dependencies, install Tesseract OCR, and ensure the spaCy model and Hugging Face model assets are available. The Tesseract executable is configured in `src/data_processing/ocr_processor.py` and may need adjustment for the local machine.

```powershell
python -m pip install -r requirements.txt
python -m uvicorn src.api.main:app --reload
```

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

The frontend development origin `http://localhost:5173` is allowed by the API. The compatibility import in root `main.py` also supports `python -m uvicorn main:app --reload`, but `src.api.main:app` is the canonical command.

## Testing

Run the backend suite from the project root and frontend checks from `frontend`:

```powershell
python -m pytest
cd frontend
npm run lint
npm run build
```

Parser tests cover article, section, numbered, recital, contents, address, header/footer, OCR artifact, and short-clause cases. Scoring and validation tests cover supported and unsupported labels, review behavior, score levels, and bounded aggregation. Model-backed PDF analysis additionally depends on working local OCR, spaCy, and Hugging Face assets.

## Known limitations

- OCR quality and PDF layout affect the text available to downstream stages; the parser is structural and cannot recover every table or document layout.
- The Legal-RoBERTa classifier can still misclassify clauses. The consistency safeguard covers selected labels only, and human review remains necessary.
- The phrase indicators and score weights are heuristic screening features, not calibrated legal-risk probabilities.
- Pretrained model downloads, the Tesseract executable, and the spaCy model must be installed or available locally. Full PDF analysis may fail when one of these dependencies is missing.
- Generated retrieval indexes and model outputs are not a substitute for a separately evaluated production retrieval or legal review workflow.

**This system is an AI-assisted screening and decision-support tool. It is not a substitute for legal advice or review by a qualified legal professional.**
