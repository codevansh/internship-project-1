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

The API is implemented in `src/api/main.py`. `src/data_processing/ocr_processor.py` extracts selectable text from each PDF page and runs Tesseract OCR only on pages with little or no text. Set `TESSERACT_CMD` when Tesseract is not available on `PATH`. `src/data_processing/clause_processor.py` recognizes ARTICLE, SECTION, decimal numbered headings, recitals, and schedule headings. It filters common page headers and footers, address blocks, contents entries, redaction notices, and OCR scraps. Article titles, section titles, schedule, section, article, clause number, title, and clause text are retained where identified. `src/model/inference_transformer.py` calls the configured Hugging Face tokenizer and sequence-classification model, preserves the top predictions, and applies a narrow text-evidence check to selected high-impact CUAD categories. An unsupported prediction becomes `Uncertain classification`, has reduced confidence, and is sent for human review; the check does not assign a replacement category.

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

Scores are deterministic screening scores, not probabilities of legal harm. `score_clause` begins at 10 points. The implementation may add a modest prior for selected material clause categories; points for observed limitation, unlimited-liability, damages, competition, commitment, termination, assignment, indemnification, confidentiality, dispute, penalty, and one-sided-term phrases; 5 points for monetary or percentage evidence; and 3 points for duration evidence. Low confidence and existing review flags create a human-review recommendation but do not add severity points. Boilerplate such as “including, without limitation” and the word “assigns” in “successors and assigns” are excluded from the corresponding unlimited-liability and assignment indicators. “Minimum amount” is treated as a commitment only when nearby language indicates a purchase, quantity, volume, or commitment. Clause scores are rounded and clamped to 0–100. The returned factors and evidence support inspection of how the score was formed.

The category prior is applied once, using the first matching group: liability or indemnity 18 points; termination, renewal, or default 12; competition restriction 12; confidentiality or data 10; dispute or governing law 10; payment or commitment 8; and unilateral contract terms 8. Phrase evidence has its own configured contribution (for example, unlimited language 30, damages 15, penalty 20, termination 5, assignment 5, indemnification 18, and confidentiality 8). Repeated matches for the same indicator do not multiply its contribution. Review recommendations and risk severity are shown as separate outputs.

Risk thresholds are LOW (0–39), MEDIUM (40–69), and HIGH (70–100). The contract score is a risk-level-weighted mean (HIGH weight 1.75, MEDIUM 1.2, LOW 1.0), plus 5 points for each HIGH clause up to a 20-point premium, then capped at 100. This aggregation depends on scored clauses, not document length. The summary reports high-, medium-, and low-risk counts separately from the number of clauses recommended for human review. A high score is a screening signal; it does not establish that a term is legally unacceptable.

## API

Canonical backend entry point: `src.api.main:app`.

- `GET /` — service status
- `GET /health` — health status
- `POST /analyze` — accepts multipart form data with a PDF in the `file` field
- FastAPI also exposes `/docs` and `/openapi.json`

The analyze response includes `filename`, `contract_report`, `review_report`, `risk_scoring`, and `risk_summary`. The frontend posts to `/analyze` at `http://127.0.0.1:8000` by default; `VITE_API_BASE_URL` can override the backend base URL. Uploaded files and generated reports/indexes are stored in a temporary per-request directory and removed after the response. Empty, malformed, encrypted, and textless PDFs are rejected with a clear client error. If every clause fails model processing, the API returns an error instead of an empty successful report. Partial clause errors are included in `contract_report.contract_summary.processing_errors`.

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

Use Python 3.13 on Windows, create a virtual environment, and install the project dependencies. Install pytest for the test suite and the spaCy English model for named-entity recognition. Hugging Face model assets are loaded by the configured model IDs and may need to be downloaded on first use. Install Tesseract OCR and make its executable available on `PATH`, or set `TESSERACT_CMD` to its full path. Text-based PDFs do not require OCR.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload
```

The `py -3.13` command requires the Windows Python Launcher. In this workspace that launcher was unavailable, so the existing venv was repaired to use the installed uv-managed Python runtime; the `.venv\Scripts\python.exe` commands below are the verified commands here.

In another terminal:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

The frontend development origin `http://localhost:5173` is allowed by the API. The compatibility import in root `main.py` also supports `.\.venv\Scripts\python.exe -m uvicorn main:app --reload`, but `src.api.main:app` is the canonical command.

## Render deployment with Docker

The root `Dockerfile` builds the FastAPI backend on Python 3.13, installs Tesseract with its English language data, installs the packages in `requirements.txt`, and installs spaCy's `en_core_web_sm` model. The API entry point is `src.api.main:app`. On Render, create a **Web Service**, select **Docker** as the runtime, and use the repository root as the Docker build context. No custom build or start command is needed: the image starts Uvicorn on `0.0.0.0` and uses Render's `PORT` environment variable (defaulting to `10000`).

Set `CORS_ORIGINS` on the backend to a comma-separated list of the exact frontend origins after the frontend has been deployed, for example the origin shown in its Render service URL. No production origin is preconfigured. The default remains the two local Vite origins. If the frontend is deployed separately, configure its `VITE_API_BASE_URL` at frontend build time to the backend service's base URL; the current localhost fallback is for local development.

The API downloads the configured Hugging Face tokenizer/classifier (`Saibo-creator/legal-roberta-base` and `KAKAROT0304/legal-roberta-clause-classifier-day10`) and Sentence Transformers model (`all-MiniLM-L6-v2`) when first used. These are not baked into the image. First analysis therefore needs outbound network access and writable model cache storage. The local `models/legal_roberta_clause_classifier_day9` checkpoint is not loaded by the API. The Docker build intentionally excludes `models/`, datasets, generated reports, vector files, tests, frontend files, virtual environments, caches, and secrets; it copies the clause label map required at inference time. OCR uses Tesseract's default `eng` language; add a language pack only if the code is changed to request another language.

The Dockerfile and ignore rules prepare the backend image only; they do not build or host the React frontend. This project has not been deployed to Render.

### Docker validation (2026-10-26)

- `docker build -t contract-intelligence-api:local .` completed successfully using the existing requirements file, a CPU-only PyTorch wheel, and the English spaCy model.
- A local container started on the configured `PORT`; `GET /health` returned HTTP 200 with status `healthy`, and `GET /docs` returned HTTP 200.
- The image contains Tesseract language data `eng` and `osd`, the clause label map, and no local training checkpoint or raw dataset. Its image size was approximately 762 MB.
- Backend tests: `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=.test-tmp/pytest` — **49 passed**. The custom temp path was needed because this Windows environment denies pytest access to its default user-level temporary directory.
- No Render service has been created, and a full model-backed PDF analysis was not tested inside the container. The Hugging Face and Sentence Transformers assets download on first analysis, so that path still depends on their availability and outbound network access at runtime.

## Testing

Run the backend suite from the project root and frontend checks from `frontend`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
cd frontend
npm.cmd run lint
npm.cmd run build
```

Parser tests cover article, section, numbered, recital, contents, address, header/footer, OCR artifact, and short-clause cases. Scoring and validation tests cover supported and unsupported labels, review behavior, score levels, and bounded aggregation. Model-backed PDF analysis additionally depends on working local OCR, spaCy, and Hugging Face assets.

### Validation in this workspace (2026-10-26)

- Python environment: `.venv` uses Python 3.13.14. Its original `pyvenv.cfg` pointed to a Windows Apps alias and could not launch. The original file was preserved as `.venv/pyvenv.cfg.windows-apps-backup`, and the venv was retargeted to the installed uv-managed Python 3.13.14 runtime. Every package listed in `requirements.txt` was already installed; `pip check` reported no broken requirements, so no packages were reinstalled or upgraded.
- `requirements.txt` does not pin Python or package versions; verification was performed with Python 3.13.14 and the package versions already present in `.venv`.
- Backend tests: `.\.venv\Scripts\python.exe -m pytest -q` — **47 passed**.
- API: started with `.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000`. `GET /health` and `GET /docs` both returned **200**.
- Live PDF analysis: `data/test_contracts/low_risk_contract_day23.pdf` returned **200**; 12 clauses detected, all 12 processed, zero processing errors, contract score **19 LOW**, 11 LOW clauses, 1 MEDIUM clause, 0 HIGH clauses, and 4 human-review items.
- OCR: Tesseract **5.5.3** was available. A temporary image-only PDF made from a test-contract page had zero native text characters and one embedded image; the real OCR path extracted **1,546 characters**.
- Additional fixture: `data/test_contracts/high_risk_contract_day23.pdf` returned **200**; 12/12 clauses processed with zero errors, contract score **43 MEDIUM**, one HIGH clause, and 4 review items. The contract-level score follows the documented aggregation; an individual HIGH clause can coexist with an overall MEDIUM level.
- Invalid PDF: a non-PDF file submitted as `broken.pdf` returned **422** with `The uploaded file is not a readable PDF.` An initial run against the stale server worker returned 500 because the uploaded file was locked during temporary-directory cleanup. Opening the PDF from in-memory bytes fixed that Windows-specific failure; the new regression test passes, and the restarted server returned 422.
- Frontend: `npm.cmd run lint` passed without warnings, and `npm.cmd run build` passed. A headless Edge check uploaded the low-risk fixture through the dashboard and confirmed it rendered **LOW RISK**, **19 / 100**, 12 clause cards, 4 review cards, all five summary counts (0 HIGH, 1 MEDIUM, 11 LOW, 12 analyzed, 4 for review), entity details on 8 clauses, and indicator details on 7 clauses. No upload error appeared.

## Known limitations

- OCR quality and PDF layout affect the text available to downstream stages; the parser is structural and cannot recover every table or document layout.
- The Legal-RoBERTa classifier can still misclassify clauses. The consistency safeguard covers selected labels only, and human review remains necessary.
- The phrase indicators and score weights are heuristic screening features, not calibrated legal-risk probabilities.
- Pretrained model downloads, the Tesseract executable, and the spaCy model must be installed or available locally. Full PDF analysis may fail when one of these dependencies is missing.
- Generated retrieval indexes and model outputs are not a substitute for a separately evaluated production retrieval or legal review workflow.

**This system is an AI-assisted screening and decision-support tool. It is not a substitute for legal advice or review by a qualified legal professional.**
