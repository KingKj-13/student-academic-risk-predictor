"use strict";

const form = document.getElementById("predict-form");
const result = document.getElementById("result");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  showError("form-error", "");
  const button = form.querySelector("button[type=submit]");
  button.disabled = true;
  const data = Object.fromEntries(new FormData(form).entries());
  for (const key of ["student_id", "attendance_pct", "internal_marks", "study_hours_weekly", "past_failures", "assignment_score"]) {
    data[key] = Number(data[key]);
  }
  try {
    const prediction = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    renderPrediction(result, prediction);
    if (prediction.alert) {
      result.prepend(el("p", { className: "banner", text: "Early-warning alert raised for this student." }));
    }
    result.append(el("p", {}, el("a", { href: `/students/${prediction.student_id}`, text: "Open student record" })));
  } catch (error) {
    showError("form-error", error.message);
  } finally {
    button.disabled = false;
  }
});

form.addEventListener("reset", () => {
  showError("form-error", "");
  result.replaceChildren(el("p", { className: "muted", text: "The prediction will appear here." }));
});
