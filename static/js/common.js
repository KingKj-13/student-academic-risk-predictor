"use strict";

// Shared helpers. All user data is written with textContent, never innerHTML.
const RISK_COLOURS = { Low: "#2e7d5b", Medium: "#b7791f", High: "#b3362f" };

function el(tag, options = {}, ...children) {
  const node = document.createElement(tag);
  if (options.className) node.className = options.className;
  if (options.text !== undefined) node.textContent = options.text;
  if (options.href) node.href = options.href;
  node.append(...children);
  return node;
}

function badge(level) {
  return el("span", { className: `badge badge-${level.toLowerCase()}`, text: `${level} risk` });
}

// A probability is never shown as certainty: 0.998 reads ">99%", not "100%".
function percent(value) {
  if (value >= 0.995) return ">99%";
  return `${Math.round(value * 100)}%`;
}

function showError(id, message) {
  const node = document.getElementById(id);
  node.textContent = message || "";
  node.hidden = !message;
}

async function api(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new Error("Could not reach the server. Check that it is running.");
  }
  if (response.status === 204) return null;
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body.detail;
    if (Array.isArray(detail)) {
      throw new Error(detail.map((d) => `${d.loc[d.loc.length - 1]}: ${d.msg}`).join("; "));
    }
    throw new Error(detail || `Request failed (${response.status}).`);
  }
  return body;
}

// Fill a container with one prediction.
function renderPrediction(container, p) {
  const meter = el("div", { className: "meter" });
  const bar = el("span", { className: `fill fill-${p.risk_level.toLowerCase()}` });
  bar.style.width = `${p.confidence * 100}%`;
  meter.append(bar);

  container.replaceChildren(
    el("div", { className: "result-head" }, badge(p.risk_level), el("strong", { text: p.status })),
    el("p", { className: "muted", text: `Model confidence ${percent(p.confidence)}. A probability, not a guarantee.` }),
    meter,
  );
  if (p.contributing_factors.length) {
    container.append(
      el("h3", { text: "Contributing factors" }),
      el("ul", {}, ...p.contributing_factors.map((f) => el("li", { text: f }))),
    );
  }
  container.append(
    el("h3", { text: "Suggested next steps" }),
    el("ul", {}, ...p.recommendations.map((r) => el("li", { text: r }))),
  );
}
