# compliance assistant

Frontend: React + Vite  
Backend: FastAPI

## Run

Backend:

```bash
cd backend
# activate your virtual environment (venv/conda/etc.)
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## URLs

- Frontend: `http://localhost:5173`
- Backend: `http://127.0.0.1:8000`

## Basic flow

1. Upload documents
2. Create a case
3. Run compliance
4. Review saved compliance results and parsed preview

## Compliance methods

The backend currently supports two compliance methods:

- `non_rag`: direct analysis over parsed document sections and normalized tables only
- `simple_rag`: retrieval-augmented analysis using indexed record content

Defined but not implemented:

- `nested_rag`
