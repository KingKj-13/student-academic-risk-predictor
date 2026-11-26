"use strict";

const PAGE_SIZE = 25;
const state = { offset: 0, total: 0, request: 0 };

function statTile(label, value, tone) {
  return el("div", { className: `tile ${tone || ""}` }, el("span", { text: label }), el("strong", { text: String(value) }));
}

function drawCharts(stats) {
  const levels = ["Low", "Medium", "High"];
  new Chart(document.getElementById("risk-chart"), {
    type: "doughnut",
    data: {
      labels: levels,
      datasets: [{ data: levels.map((l) => stats.risk_counts[l]), backgroundColor: levels.map((l) => RISK_COLOURS[l]) }],
    },
    options: { maintainAspectRatio: false, plugins: { legend: { position: "right" } } },
  });
  const groups = levels.filter((l) => stats.averages_by_risk[l]);
  const series = [
    ["attendance", "Attendance %", "#243b6b"],
    ["internal_marks", "Internal marks", "#5b7bb2"],
    ["assignment_score", "Assignment score", "#a9bcdc"],
  ];
  new Chart(document.getElementById("average-chart"), {
    type: "bar",
    data: {
      labels: groups.map((l) => `${l} risk`),
      datasets: series.map(([key, label, colour]) => ({
        label, backgroundColor: colour, data: groups.map((l) => stats.averages_by_risk[l][key]),
      })),
    },
    options: { maintainAspectRatio: false, scales: { y: { beginAtZero: true, max: 100 } } },
  });
}

async function loadStats() {
  const stats = await api("/api/dashboard-stats");
  document.getElementById("tiles").replaceChildren(
    statTile("Students", stats.total_students),
    statTile("High risk", stats.risk_counts.High, "tone-high"),
    statTile("Medium risk", stats.risk_counts.Medium, "tone-medium"),
    statTile("Low risk", stats.risk_counts.Low, "tone-low"),
    statTile("Average attendance", `${stats.average_attendance}%`),
    statTile("Average internal marks", stats.average_internal_marks),
  );
  const banner = document.getElementById("alert-banner");
  const high = stats.risk_counts.High;
  banner.textContent = `Early warning: ${high} student${high === 1 ? " is" : "s are"} currently at high risk.`;
  banner.hidden = high === 0;
  if (stats.total_students > 0) drawCharts(stats);
}

async function loadStudents() {
  // Only the newest request may update the table (typing fires several).
  const ticket = ++state.request;
  const params = new URLSearchParams({ limit: PAGE_SIZE, offset: state.offset });
  const search = document.getElementById("search").value.trim();
  const risk = document.getElementById("risk-filter").value;
  if (search) params.set("search", search);
  if (risk) params.set("risk", risk);
  const data = await api(`/api/students?${params}`);
  if (ticket !== state.request) return;
  state.total = data.total;

  const rows = data.items.map((s) => el("tr", { className: s.risk_level === "High" ? "row-high" : "" },
    el("td", { text: String(s.student_id) }),
    el("td", {}, el("a", { href: `/students/${s.student_id}`, text: s.name })),
    el("td", {}, badge(s.risk_level)),
    el("td", { text: percent(s.confidence) }),
    el("td", { text: `${s.attendance_pct}%` }),
    el("td", { text: String(s.internal_marks) }),
    el("td", { text: String(s.assignment_score) }),
    el("td", { text: String(s.past_failures) }),
  ));
  if (!rows.length) {
    const cell = el("td", { className: "muted", text: "No students found. Add one or upload a CSV file." });
    cell.colSpan = 8;
    rows.push(el("tr", {}, cell));
  }
  document.getElementById("students-body").replaceChildren(...rows);
  const end = Math.min(state.offset + PAGE_SIZE, state.total);
  document.getElementById("page-info").textContent = state.total ? `${state.offset + 1}–${end} of ${state.total}` : "";
  document.getElementById("prev").disabled = state.offset === 0;
  document.getElementById("next").disabled = end >= state.total;
}

function refresh() {
  loadStudents().catch((error) => showError("dashboard-error", error.message));
}

let timer;
document.getElementById("search").addEventListener("input", () => {
  clearTimeout(timer);
  timer = setTimeout(() => { state.offset = 0; refresh(); }, 250);
});
document.getElementById("risk-filter").addEventListener("change", () => { state.offset = 0; refresh(); });
document.getElementById("prev").addEventListener("click", () => { state.offset -= PAGE_SIZE; refresh(); });
document.getElementById("next").addEventListener("click", () => { state.offset += PAGE_SIZE; refresh(); });

loadStats().catch((error) => showError("dashboard-error", error.message));
refresh();
