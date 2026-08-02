// app.js -- entrypoint: loads config, wires panels together, owns the
// currently-displayed result set (Plot Builder + Properties Table on the
// Setup/Plot Builder pages). Comparison and Console are self-contained
// pages (comparison.js / console.js); projects.js owns the project bar.
import { api } from "./api.js";
import { initGasPanel, initLiquidPanel, initTemperaturePanel, initEosPanel, initExpDataPanel } from "./panels.js";
import { initPlotBuilder } from "./plotbuilder.js";
import { initPropertiesView } from "./properties.js";
import { initComparisonPage } from "./comparison.js";
import { initConsolePage } from "./console.js";
import { initProjectBar } from "./projects.js";

const $ = (sel) => document.querySelector(sel);

async function main() {
  const config = await api.getConfig();

  const gasPanel = initGasPanel($('[data-testid="panel-gas"]'), config.gases);
  gasPanel.addRow("CO2", 0.4);
  gasPanel.addRow("H2", 0.6);

  const liquidPanel = initLiquidPanel($('[data-testid="panel-liquid"]'), config.promoters);
  const temperaturePanel = initTemperaturePanel($('[data-testid="panel-temperature"]'));
  const eosPanel = initEosPanel($('[data-testid="panel-eos"]'), config.eos_names);
  const expDataPanel = initExpDataPanel($('[data-testid="panel-expdata"]'), config.literature);

  const plotBuilder = initPlotBuilder($('[data-testid="plot-sidebar"]'), $("#plot-canvas"));
  const propertiesView = initPropertiesView($('[data-testid="view-properties"]'));

  const consolePage = initConsolePage($('[data-testid="page-console"]'));
  const { logEvent } = consolePage;

  initComparisonPage(
    $('[data-testid="page-comparison"]'),
    config.promoters,
    config.eos_names,
    {
      getTRange: () => temperaturePanel.getTRange(),
      getGasComp: () => gasPanel.getComposition(),
      getLiqComp: () => liquidPanel.getComposition(),
    },
    logEvent
  );

  // ── Project bar ──────────────────────────────────────────────────────
  const projectBar = initProjectBar(
    $('[data-testid="project-bar"]'),
    $('[data-testid="project-modal"]'),
    {
      getSetupState: () => ({
        gas_comp: gasPanel.getComposition(),
        liq_comp: liquidPanel.getComposition(),
        ...temperaturePanel.getTRange(),
        eos_names: eosPanel.getSelected(),
        ...expDataPanel.getState(),
      }),
      applySetupState: (cfg) => {
        if (cfg.gas_comp && Object.keys(cfg.gas_comp).length) gasPanel.setComposition(cfg.gas_comp);
        if (cfg.liq_comp && Object.keys(cfg.liq_comp).length) liquidPanel.setComposition(cfg.liq_comp);
        temperaturePanel.setTRange(cfg);
        if (cfg.eos_names && cfg.eos_names.length) eosPanel.setSelected(cfg.eos_names);
        expDataPanel.setState(cfg);
      },
    },
    logEvent
  );
  await projectBar.init();

  // ── Primary page nav (Setup / Plot Builder / Comparison / Console) ──────
  function switchPage(name) {
    for (const btn of document.querySelectorAll(".main-nav .tab")) {
      btn.classList.toggle("active", btn.dataset.page === name);
    }
    $('[data-testid="page-setup"]').hidden = name !== "setup";
    $('[data-testid="page-plot"]').hidden = name !== "plot";
    $('[data-testid="page-comparison"]').hidden = name !== "comparison";
    $('[data-testid="page-console"]').hidden = name !== "console";
  }
  $('[data-testid="nav-setup"]').addEventListener("click", () => switchPage("setup"));
  $('[data-testid="nav-plot"]').addEventListener("click", () => switchPage("plot"));
  $('[data-testid="nav-comparison"]').addEventListener("click", () => switchPage("comparison"));
  $('[data-testid="nav-console"]').addEventListener("click", () => switchPage("console"));

  // ── Plot Builder / Properties Table sub-tabs (Plot Builder page) ────────
  function switchTab(name) {
    for (const btn of document.querySelectorAll('[data-testid="page-plot"] .tabs .tab')) {
      btn.classList.toggle("active", btn.dataset.tab === name);
    }
    $('[data-testid="view-plot"]').hidden = name !== "plot";
    $('[data-testid="view-properties"]').hidden = name !== "properties";
  }
  $('[data-testid="tab-plot"]').addEventListener("click", () => switchTab("plot"));
  $('[data-testid="tab-properties"]').addEventListener("click", () => switchTab("properties"));

  // ── Run ──────────────────────────────────────────────────────────────
  const runBtn = $('[data-testid="run-button"]');
  const runStatus = $('[data-testid="run-status"]');

  runBtn.addEventListener("click", async () => {
    let gasComp, liqComp, T, eosNames, expData;
    try {
      gasComp = gasPanel.getComposition();
      liqComp = liquidPanel.getComposition();
      T = temperaturePanel.getTRange();
      eosNames = eosPanel.getSelected();
      expData = await expDataPanel.getExpData();
    } catch (exc) {
      runStatus.textContent = `⚠  ${exc.message}`;
      return;
    }

    runBtn.disabled = true;
    runStatus.textContent = "Running…";
    logEvent("info", "Run started.");
    try {
      const resp = await api.run({
        gas_comp: gasComp, liq_comp: liqComp,
        T_min: T.T_min, T_max: T.T_max, T_step: T.T_step,
        eos_names: eosNames,
      });
      if (resp.error) {
        runStatus.textContent = `⚠  Error: ${resp.error}`;
        logEvent("error", `Run failed: ${resp.error}`);
      } else {
        const nTotal = Object.values(resp.results).reduce((a, rows) => a + rows.length, 0);
        runStatus.textContent = resp.cached
          ? `⚡  Cache HIT — ${Object.keys(resp.results).length} model(s), ${nTotal} rows loaded instantly.`
          : `✓  Computed & cached — ${Object.keys(resp.results).length} model(s), ${nTotal} total rows.`;
        logEvent("success", resp.cached
          ? `Run done (cache hit): ${Object.keys(resp.results).length} model(s), ${nTotal} rows.`
          : `Run done (computed): ${Object.keys(resp.results).length} model(s), ${nTotal} rows.`);
        plotBuilder.setResults(resp.results, expData, "EOS models", ["P_eq (MPa)"], null);
        propertiesView.setResults(resp.results);
        switchPage("plot");
        switchTab("plot");
      }
    } catch (exc) {
      runStatus.textContent = `⚠  Error: ${exc.message}`;
      logEvent("error", `Run failed: ${exc.message}`);
    } finally {
      runBtn.disabled = false;
      consolePage.cachePanel.refresh();
    }
  });
}

main().catch((exc) => {
  console.error(exc);
  document.body.insertAdjacentHTML(
    "afterbegin",
    `<div style="background:#dc2626;color:#fff;padding:10px;" data-testid="app-error">Failed to load app: ${exc.message}</div>`
  );
});
