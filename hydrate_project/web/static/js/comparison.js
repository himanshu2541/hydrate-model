// comparison.js -- Comparison page: composition-sweep controls (port of the
// old sweep.js / ui/launcher/sweep_panel.py) plus its OWN Plot Builder +
// Properties Table instances, so comparison results get a dedicated results
// surface instead of routing into the Setup page's Plot Builder. The plot
// multiselects already let you pick any numeric column (SF_*, Ideal_SF_*,
// P_eq, Delta_Mu_*, dH_diss, ...) -- not hardcoded to separation factor.
import { api } from "./api.js";
import { initPlotBuilder } from "./plotbuilder.js";
import { initPropertiesView } from "./properties.js";

const $ = (root, sel) => root.querySelector(sel);
const $$ = (root, sel) => Array.from(root.querySelectorAll(sel));

export function initComparisonPage(page, promoters, eosNames, { getTRange, getGasComp, getLiqComp }, logEvent) {
  const promoterSelect = $(page, '[data-testid="sweep-promoter"]');
  const dioxRow = $(page, '[data-testid="sweep-diox-row"]');
  const ratioRow = $(page, '[data-testid="sweep-ratio-row"]');
  const dioxValuesInput = $(page, '[data-testid="sweep-diox-values"]');
  const ratioText = $(page, '[data-testid="sweep-ratio-text"]');
  const eosSelect = $(page, '[data-testid="sweep-eos"]');
  const runBtn = $(page, '[data-testid="sweep-run"]');
  const statusEl = $(page, '[data-testid="sweep-status"]');

  for (const key of Object.keys(promoters)) {
    const o = document.createElement("option"); o.value = key; o.textContent = key;
    promoterSelect.appendChild(o);
  }
  for (const name of eosNames) {
    const o = document.createElement("option"); o.value = name; o.textContent = name;
    eosSelect.appendChild(o);
  }

  $$(page, 'input[name="sweep-axis"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      const isDiox = $(page, 'input[name="sweep-axis"]:checked').value === "diox";
      dioxRow.hidden = !isDiox;
      ratioRow.hidden = isDiox;
    });
  });

  const plotBuilder = initPlotBuilder(
    $(page, '[data-testid="cmp-plot-sidebar"]'),
    $(page, "#cmp-plot-canvas"),
    "cmp-plot-comparison"
  );
  const propertiesView = initPropertiesView($(page, '[data-testid="cmp-view-properties"]'));

  function switchTab(name) {
    for (const btn of $$(page, ".tabs .tab")) {
      btn.classList.toggle("active", btn.dataset.tab === name);
    }
    $(page, '[data-testid="cmp-view-plot"]').hidden = name !== "plot";
    $(page, '[data-testid="cmp-view-properties"]').hidden = name !== "properties";
  }
  $(page, '[data-testid="cmp-tab-plot"]').addEventListener("click", () => switchTab("plot"));
  $(page, '[data-testid="cmp-tab-properties"]').addEventListener("click", () => switchTab("properties"));

  runBtn.addEventListener("click", async () => {
    let T;
    try { T = getTRange(); } catch (exc) { alert(exc.message); return; }
    const axis = $(page, 'input[name="sweep-axis"]:checked').value;
    const eosName = eosSelect.value;
    if (!eosName) { alert("Select an EOS model."); return; }

    runBtn.disabled = true;
    statusEl.textContent = "Running comparison sweep…";
    logEvent("info", `Comparison sweep started (${axis === "diox" ? "vary DIOX%" : "vary gas ratio"}, ${eosName}).`);
    try {
      let resp;
      if (axis === "diox") {
        resp = await api.sweepDiox({
          gas_comp: getGasComp(),
          promoter_key: promoterSelect.value,
          diox_values: dioxValuesInput.value,
          ...T,
          eos_name: eosName,
        });
      } else {
        resp = await api.sweepRatio({
          gas_ratio_lines: ratioText.value,
          liq_comp: getLiqComp(),
          ...T,
          eos_name: eosName,
        });
      }
      if (resp.error) {
        statusEl.textContent = `⚠  Error: ${resp.error}`;
        logEvent("error", `Comparison sweep failed: ${resp.error}`);
      } else {
        statusEl.textContent = `✓  ${Object.keys(resp.results).length} composition(s) computed.`;
        logEvent("success", `Comparison sweep done: ${Object.keys(resp.results).length} composition(s).`);
        plotBuilder.setResults(resp.results, null, resp.series_label, ["P_eq (MPa)"], resp.default_cols);
        propertiesView.setResults(resp.results);
        switchTab("plot");
      }
    } catch (exc) {
      statusEl.textContent = `⚠  Error: ${exc.message}`;
      logEvent("error", `Comparison sweep failed: ${exc.message}`);
    } finally {
      runBtn.disabled = false;
    }
  });
}
