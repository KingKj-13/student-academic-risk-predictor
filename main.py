"""FastAPI application: web pages and JSON API.

Run:  uvicorn main:app --reload
"""

from __future__ import annotations

import csv
import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

import alerts
import config
import security
from db import STUDENT_COLUMNS, Database
from predictor import RiskPredictor
from schemas import PredictionOut, RowError, StudentIn, UploadOut

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

SAMPLE_ROWS = [
    [2001, "Asha Verma", 92, 81, 18, 0, 86, "Female", "Yes"],
    [2002, "Vikram Rao", 68, 52, 9, 1, 58, "Male", "No"],
    [2003, "Neel Das", 47, 31, 4, 3, 36, "Male", "No"],
]


def score_and_save(app: FastAPI, students: list[StudentIn], background: BackgroundTasks) -> list[PredictionOut]:
    """Predict, store, and queue alerts for a batch of validated students."""
    records = [student.model_dump() for student in students]
    predictions = app.state.predictor.predict(records)
    for record, prediction in zip(records, predictions):
        record.update(prediction)
    previous_levels = app.state.db.save_predictions(records)

    outputs, new_alerts = [], []
    for record, previous in zip(records, previous_levels):
        alert = alerts.is_new_alert(record["risk_level"], previous)
        if alert:
            new_alerts.append(record)
        outputs.append(
            PredictionOut(
                student_id=record["student_id"],
                name=record["name"],
                risk_level=record["risk_level"],
                confidence=record["confidence"],
                probabilities=record["probabilities"],
                status=alerts.status_for(record["risk_level"]),
                contributing_factors=record["contributing_factors"],
                recommendations=alerts.recommendations_for(record["risk_level"], record["contributing_factors"]),
                alert=alert,
            )
        )
    if new_alerts:
        background.add_task(alerts.raise_alerts, new_alerts)
    return outputs


def read_csv_upload(content: bytes) -> tuple[list[StudentIn], list[RowError], int]:
    """Validate a CSV file row by row. One bad row never blocks the good ones."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(400, "The file is not UTF-8 text. Save it as 'CSV UTF-8' and try again.")

    reader = csv.DictReader(io.StringIO(text, newline=""))
    try:
        header = [(name or "").strip().lower() for name in (reader.fieldnames or [])]
        missing = [column for column in STUDENT_COLUMNS if column not in header]
        if missing:
            raise HTTPException(400, f"Missing column(s): {', '.join(missing)}")
        reader.fieldnames = header
        rows = list(reader)
    except csv.Error as error:
        raise HTTPException(400, f"The file is not a valid CSV: {error}")

    if not rows:
        raise HTTPException(400, "The file has no data rows.")
    if len(rows) > config.MAX_UPLOAD_ROWS:
        raise HTTPException(413, f"Too many rows (limit {config.MAX_UPLOAD_ROWS}).")

    valid: list[StudentIn] = []
    errors: list[RowError] = []
    seen: set[int] = set()
    for number, row in enumerate(rows, start=2):  # row 1 is the header
        values = {column: (row.get(column) or "").strip() for column in STUDENT_COLUMNS}
        shown_id = values["student_id"][:20]
        blank = [column for column, value in values.items() if not value]
        if blank:
            errors.append(RowError(row=number, student_id=shown_id, errors=[f"{c}: missing" for c in blank]))
            continue
        try:
            student = StudentIn.model_validate(values)
        except ValidationError as error:
            messages = [f"{item['loc'][0]}: {item['msg']}" for item in error.errors()]
            errors.append(RowError(row=number, student_id=shown_id, errors=messages))
            continue
        if student.student_id in seen:
            errors.append(RowError(row=number, student_id=shown_id, errors=["student_id: repeated in this file"]))
            continue
        seen.add(student.student_id)
        valid.append(student)
    return valid, errors, len(rows)


def create_app(db_path: Path | None = None, model_path: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = RiskPredictor(model_path or config.MODEL_PATH)
        app.state.db = Database(db_path or config.DB_PATH)
        yield

    app = FastAPI(
        title="Student Academic Risk Prediction & Early-Warning System",
        description="Predicts Low, Medium or High academic risk to support (not replace) a teacher's judgement.",
        version="1.0.0",
        lifespan=lifespan,
    )
    security.install(app)
    app.mount("/static", StaticFiles(directory=config.BASE_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=config.BASE_DIR / "templates")

    def page(request: Request, name: str, **context) -> HTMLResponse:
        return templates.TemplateResponse(request, name, context)

    # ---------------------------------------------------------------- pages
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(request: Request):
        return page(request, "index.html", active="predict")

    @app.get("/upload", response_class=HTMLResponse, include_in_schema=False)
    def upload_page(request: Request):
        return page(request, "upload.html", active="upload")

    @app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    def dashboard_page(request: Request):
        return page(request, "dashboard.html", active="dashboard")

    @app.get("/students/{student_id}", response_class=HTMLResponse, include_in_schema=False)
    def student_page(request: Request, student_id: int):
        return page(request, "student_details.html", active="dashboard", student_id=student_id)

    # ---------------------------------------------------------------- API
    @app.get("/health", tags=["system"])
    def health(request: Request) -> dict:
        return {"status": "ok", "model": request.app.state.predictor.model_name}

    @app.post("/api/predict", response_model=PredictionOut, tags=["prediction"])
    def predict(student: StudentIn, request: Request, background: BackgroundTasks):
        """Predict the risk of one student and store the result."""
        return score_and_save(request.app, [student], background)[0]

    @app.post("/api/upload", response_model=UploadOut, tags=["prediction"])
    def upload(request: Request, background: BackgroundTasks, file: UploadFile = File(...)):
        """Predict every valid row of a CSV file."""
        if not (file.filename or "").lower().endswith(".csv"):
            raise HTTPException(400, "Please upload a .csv file.")
        content = file.file.read(config.MAX_UPLOAD_BYTES + 1)
        if len(content) > config.MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File is larger than 2 MB.")
        valid, errors, total = read_csv_upload(content)
        results = score_and_save(request.app, valid, background) if valid else []
        order = {"High": 0, "Medium": 1, "Low": 2}
        results.sort(key=lambda r: (order[r.risk_level], -r.confidence))
        counts = {level: sum(r.risk_level == level for r in results) for level in config.RISK_LEVELS}
        return UploadOut(
            total_rows=total, processed=len(results), rejected=len(errors),
            counts=counts, results=results, errors=errors,
        )

    @app.get("/api/sample-csv", tags=["prediction"])
    def sample_csv():
        """A small, correctly formatted example file."""
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(STUDENT_COLUMNS)
        writer.writerows(SAMPLE_ROWS)
        return Response(
            buffer.getvalue(), media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="sample_students.csv"'},
        )

    @app.get("/api/students", tags=["records"])
    def students(
        request: Request,
        risk: Literal["Low", "Medium", "High"] | None = None,
        search: str | None = Query(None, max_length=100),
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> dict:
        """Latest result for each student, High risk first."""
        total, items = request.app.state.db.list_students(risk, search, limit, offset)
        return {"total": total, "items": items}

    @app.get("/api/students/{student_id}", tags=["records"])
    def student(student_id: int, request: Request) -> dict:
        """One student with the full prediction history."""
        record = request.app.state.db.get_student(student_id) if 0 < student_id < 10**12 else None
        if record is None:
            raise HTTPException(404, "Student not found.")
        latest = record["history"][-1]
        record["status"] = alerts.status_for(latest["risk_level"])
        record["recommendations"] = alerts.recommendations_for(latest["risk_level"], latest["contributing_factors"])
        return record

    @app.delete("/api/students/{student_id}", status_code=204, tags=["records"])
    def delete_student(student_id: int, request: Request):
        """Delete a student and their history."""
        if not (0 < student_id < 10**12 and request.app.state.db.delete_student(student_id)):
            raise HTTPException(404, "Student not found.")

    @app.get("/api/dashboard-stats", tags=["records"])
    def dashboard_stats(request: Request) -> dict:
        return request.app.state.db.stats()

    return app


app = create_app()
