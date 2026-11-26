"use strict";

const studentId = document.getElementById("student").dataset.studentId;

function when(iso) {
  return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function render(student) {
  const latest = student.history[student.history.length - 1];
  document.getElementById("student-name").textContent = `${student.name} · ID ${student.student_id}`;
  renderPrediction(document.getElementById("current"), {
    ...latest, status: student.status, recommendations: student.recommendations,
  });

  const fields = [
    ["Attendance", `${student.attendance_pct}%`],
    ["Internal marks", student.internal_marks],
    ["Study hours per week", student.study_hours_weekly],
    ["Previous failed subjects", student.past_failures],
    ["Assignment score", student.assignment_score],
    ["Gender", student.gender],
    ["Extracurricular", student.extracurricular],
  ];
  document.getElementById("record").replaceChildren(
    ...fields.flatMap(([label, value]) => [el("dt", { text: label }), el("dd", { text: String(value) })]),
  );

  document.getElementById("history-body").replaceChildren(
    ...[...student.history].reverse().map((h) => el("tr", {},
      el("td", { text: when(h.created_at) }),
      el("td", {}, badge(h.risk_level)),
      el("td", { text: percent(h.confidence) }),
      el("td", { text: `${h.attendance_pct}%` }),
      el("td", { text: String(h.internal_marks) }),
      el("td", { text: String(h.assignment_score) }),
    )),
  );

  document.getElementById("trend-note").textContent = student.history.length < 2
    ? "Only one record so far. A trend appears after the student is assessed again."
    : `${student.history.length} records.`;
  document.getElementById("details").hidden = false;

  const series = [
    ["attendance_pct", "Attendance %", "#243b6b"],
    ["internal_marks", "Internal marks", "#b7791f"],
    ["assignment_score", "Assignment score", "#2e7d5b"],
  ];
  new Chart(document.getElementById("trend-chart"), {
    type: "line",
    data: {
      labels: student.history.map((h) => when(h.created_at)),
      datasets: series.map(([key, label, colour]) => ({
        label, borderColor: colour, backgroundColor: colour, data: student.history.map((h) => h[key]),
      })),
    },
    options: { maintainAspectRatio: false, scales: { y: { beginAtZero: true, max: 100 } } },
  });
}

document.getElementById("delete-student").addEventListener("click", async () => {
  if (!window.confirm("Delete this student and all prediction history? This cannot be undone.")) return;
  try {
    await api(`/api/students/${studentId}`, { method: "DELETE" });
    window.location.href = "/dashboard";
  } catch (error) {
    showError("details-error", error.message);
  }
});

api(`/api/students/${studentId}`).then(render).catch((error) => {
  document.getElementById("student-name").textContent = `Student ${studentId}`;
  showError("details-error", error.message);
});
