"use strict";

const uploadForm = document.getElementById("upload-form");

function tile(label, value, tone) {
  return el("div", { className: `tile ${tone || ""}` }, el("span", { text: label }), el("strong", { text: String(value) }));
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  showError("upload-error", "");
  document.getElementById("upload-output").hidden = true;
  const button = uploadForm.querySelector("button");
  button.disabled = true;
  button.textContent = "Processing…";
  try {
    const report = await api("/api/upload", { method: "POST", body: new FormData(uploadForm) });
    document.getElementById("upload-summary").replaceChildren(
      tile("Rows in file", report.total_rows),
      tile("Processed", report.processed),
      tile("Rejected", report.rejected),
      tile("High risk", report.counts.High, "tone-high"),
      tile("Medium risk", report.counts.Medium, "tone-medium"),
      tile("Low risk", report.counts.Low, "tone-low"),
    );
    document.getElementById("errors-card").hidden = report.errors.length === 0;
    document.getElementById("errors-body").replaceChildren(
      ...report.errors.map((e) => el("tr", {},
        el("td", { text: String(e.row) }),
        el("td", { text: e.student_id || "—" }),
        el("td", { text: e.errors.join("; ") }),
      )),
    );
    document.getElementById("results-body").replaceChildren(
      ...report.results.map((r) => el("tr", {},
        el("td", { text: String(r.student_id) }),
        el("td", {}, el("a", { href: `/students/${r.student_id}`, text: r.name })),
        el("td", {}, badge(r.risk_level)),
        el("td", { text: percent(r.confidence) }),
        el("td", { text: r.contributing_factors.join(", ") || "—" }),
      )),
    );
    document.getElementById("upload-output").hidden = false;
  } catch (error) {
    showError("upload-error", error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Upload and predict";
  }
});
