import base64
import io

import pytest
from fastapi.testclient import TestClient

import alerts
from conftest import STRONG, WEAK
from db import STUDENT_COLUMNS
from main import create_app


def csv_file(rows, header=STUDENT_COLUMNS, name="class.csv"):
    lines = [",".join(header)] + [",".join(str(v) for v in row) for row in rows]
    return {"file": (name, io.BytesIO("\n".join(lines).encode()), "text/csv")}


def row(student):
    return [student[c] for c in STUDENT_COLUMNS]


@pytest.mark.parametrize("path", ["/", "/upload", "/dashboard", "/students/1", "/docs", "/health"])
def test_pages_open(client, path):
    assert client.get(path).status_code == 200


def test_weak_student_is_high_risk_with_factors(client):
    body = client.post("/api/predict", json=WEAK).json()
    assert body["risk_level"] == "High"
    assert body["status"] == "At Risk"
    assert body["alert"] is True
    assert body["contributing_factors"]
    assert 0 <= body["confidence"] <= 1
    assert sum(body["probabilities"].values()) == pytest.approx(1, abs=0.01)


def test_strong_student_is_low_risk(client):
    body = client.post("/api/predict", json=STRONG).json()
    assert body["risk_level"] == "Low"
    assert body["contributing_factors"] == []
    assert body["alert"] is False


def test_alert_is_raised_once_while_student_stays_high(client):
    assert client.post("/api/predict", json=WEAK).json()["alert"] is True
    assert client.post("/api/predict", json=WEAK).json()["alert"] is False


@pytest.mark.parametrize(
    "change",
    [
        {"attendance_pct": 101}, {"internal_marks": -1}, {"past_failures": 1.5}, {"name": "   "},
        {"gender": "x"}, {"extracurricular": "maybe"}, {"student_id": 0}, {"assignment_score": "abc"},
    ],
)
def test_invalid_input_is_rejected(client, change):
    assert client.post("/api/predict", json={**STRONG, **change}).status_code == 422


def test_missing_field_is_rejected(client):
    payload = {k: v for k, v in STRONG.items() if k != "attendance_pct"}
    assert client.post("/api/predict", json=payload).status_code == 422


def test_history_is_stored(client):
    client.post("/api/predict", json=STRONG)
    client.post("/api/predict", json={**WEAK, "student_id": 1, "name": "Strong Student"})
    record = client.get("/api/students/1").json()
    assert [h["risk_level"] for h in record["history"]] == ["Low", "High"]
    assert record["attendance_pct"] == WEAK["attendance_pct"]
    assert record["status"] == "At Risk"


def test_unknown_student_gives_404(client):
    assert client.get("/api/students/999").status_code == 404
    assert client.get("/api/students/99999999999999999999").status_code == 404


def test_high_risk_students_are_listed_first(client):
    client.post("/api/predict", json=STRONG)
    client.post("/api/predict", json=WEAK)
    items = client.get("/api/students").json()["items"]
    assert [i["student_id"] for i in items] == [2, 1]
    assert client.get("/api/students", params={"risk": "Low"}).json()["total"] == 1
    assert client.get("/api/students", params={"search": "weak"}).json()["total"] == 1
    assert client.get("/api/students", params={"search": "%"}).json()["total"] == 0
    assert client.get("/api/students", params={"search": "'; DROP TABLE students;--"}).json()["total"] == 0


def test_dashboard_statistics(client):
    assert client.get("/api/dashboard-stats").json()["total_students"] == 0
    client.post("/api/predict", json=STRONG)
    client.post("/api/predict", json=WEAK)
    stats = client.get("/api/dashboard-stats").json()
    assert stats["total_students"] == 2
    assert stats["risk_counts"] == {"Low": 1, "Medium": 0, "High": 1}
    assert stats["average_attendance"] == pytest.approx((94 + 45) / 2)


def test_delete_student_removes_history(client):
    client.post("/api/predict", json=STRONG)
    assert client.delete("/api/students/1").status_code == 204
    assert client.get("/api/students/1").status_code == 404
    assert client.get("/api/dashboard-stats").json()["total_students"] == 0


def test_csv_upload_processes_valid_rows_and_reports_bad_ones(client):
    bad_range = {**STRONG, "student_id": 3, "attendance_pct": 140}
    blank = {**STRONG, "student_id": 4, "internal_marks": ""}
    repeated = {**WEAK, "name": "Again"}
    response = client.post("/api/upload", files=csv_file([row(s) for s in (STRONG, WEAK, bad_range, blank, repeated)]))
    body = response.json()
    assert response.status_code == 200
    assert (body["total_rows"], body["processed"], body["rejected"]) == (5, 2, 3)
    assert [e["row"] for e in body["errors"]] == [4, 5, 6]
    assert body["results"][0]["risk_level"] == "High"
    assert client.get("/api/dashboard-stats").json()["total_students"] == 2


def test_csv_accepts_excel_style_file(client):
    header = [c.upper() for c in STUDENT_COLUMNS] + ["RISK_LEVEL"]
    values = row({**STRONG, "past_failures": "0.0", "gender": "female", "extracurricular": "yes"}) + ["ignored"]
    text = "\r\n".join([",".join(header), ",".join(str(v) for v in values)])
    files = {"file": ("x.CSV", io.BytesIO(text.encode("utf-8-sig")), "text/csv")}
    assert client.post("/api/upload", files=files).json()["processed"] == 1


@pytest.mark.parametrize(
    "files, status",
    [
        (csv_file([row(STRONG)], name="class.txt"), 400),
        (csv_file([["1", "A"]], header=["student_id", "name"]), 400),
        (csv_file([]), 400),
        ({"file": ("a.csv", io.BytesIO(b"\xff\xfe\x00bad"), "text/csv")}, 400),
        ({"file": ("big.csv", io.BytesIO(b"a" * (2 * 1024 * 1024 + 10)), "text/csv")}, 413),
    ],
)
def test_bad_files_are_refused(client, files, status):
    assert client.post("/api/upload", files=files).status_code == status


def test_sample_csv_uploads_cleanly(client):
    sample = client.get("/api/sample-csv").content
    body = client.post("/api/upload", files={"file": ("s.csv", io.BytesIO(sample), "text/csv")}).json()
    assert body["rejected"] == 0 and body["processed"] == 3


def test_cross_origin_post_is_refused(client):
    response = client.post("/api/predict", json=STRONG, headers={"Origin": "https://evil.example"})
    assert response.status_code == 403


def test_password_protection(trained, tmp_path, monkeypatch):
    monkeypatch.setenv("APP_USERNAME", "teacher")
    monkeypatch.setenv("APP_PASSWORD", "test-only-password")
    app = create_app(db_path=tmp_path / "auth.db", model_path=trained["model_path"])
    token = base64.b64encode(b"teacher:test-only-password").decode()
    wrong = base64.b64encode(b"teacher:wrong").decode()
    with TestClient(app) as secured:
        for path in ("/", "/docs", "/api/students", "/static/js/common.js"):
            assert secured.get(path).status_code == 401
        assert secured.get("/api/students", headers={"Authorization": f"Basic {wrong}"}).status_code == 401
        assert secured.get("/api/students", headers={"Authorization": f"Basic {token}"}).status_code == 200
        assert secured.get("/health").status_code == 200


def test_required_auth_without_password_refuses_to_start(monkeypatch):
    monkeypatch.setenv("REQUIRE_AUTH", "1")
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    with pytest.raises(RuntimeError):
        create_app()


def test_email_is_skipped_when_not_configured(monkeypatch):
    monkeypatch.delenv("SMTP_HOST", raising=False)
    assert alerts.send_alert_email([{**WEAK, "contributing_factors": []}]) is False


def test_email_digest_is_sent(monkeypatch):
    sent = []

    class FakeSMTP:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def starttls(self, **kwargs): pass
        def login(self, *args): pass
        def send_message(self, message): sent.append(message)

    monkeypatch.setattr(alerts.smtplib, "SMTP", FakeSMTP)
    settings = alerts.EmailSettings("smtp.example.com", 587, "u", "p", "from@example.com", "to@example.com")
    students = [{**WEAK, "contributing_factors": ["Low attendance"]}]
    assert alerts.send_alert_email(students, settings) is True
    assert "Weak Student" in sent[0].get_content()


def test_email_failure_does_not_crash(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(alerts.smtplib, "SMTP", boom)
    settings = alerts.EmailSettings("smtp.example.com", 587, "", "", "a@example.com", "b@example.com")
    assert alerts.send_alert_email([{**WEAK, "contributing_factors": []}], settings) is False
