// plotbuilder.js -- Plotly.js port of ui/general_plotter/core.py + sidebar.py.
// Grid semantics match build_grid/draw_cell 1:1: row_vars pick how many
// (repeated) rows to draw, col_vars each get their own column of subplots,
// cell (r,c) always shows col_vars[c] as the Y variable.

const PALETTE = ["#2563eb", "#16a34a", "#dc2626", "#ea580c", "#7c3aed", "#0891b2"];
const SYMBOLS = ["circle", "square", "triangle-up", "diamond", "triangle-down", "cross"];
const EXP_COLOR = "#ca8a04";

const $ = (root, sel) => root.querySelector(sel);

function isNumericColumn(rows, col) {
  return rows.some((r) => typeof r[col] === "number" && r[col] !== null && !Number.isNaN(r[col]));
}

function getNumericColumns(resultsDict) {
  const cols = new Set();
  for (const rows of Object.values(resultsDict)) {
    if (!rows.length) continue;
    for (const key of Object.keys(rows[0])) {
      if (isNumericColumn(rows, key)) cols.add(key);
    }
  }
  const priority = ["T (K)", "P_eq (MPa)"];
  const rest = Array.from(cols).filter((c) => !priority.includes(c)).sort();
  return priority.filter((c) => cols.has(c)).concat(rest);
}

function fillMultiSelect(select, options, defaults) {
  select.innerHTML = "";
  for (const opt of options) {
    const o = document.createElement("option");
    o.value = opt; o.textContent = opt;
    if (defaults.includes(opt)) o.selected = true;
    select.appendChild(o);
  }
}

function getSelected(select) {
  return Array.from(select.selectedOptions).map((o) => o.value);
}

export function initPlotBuilder(sidebar, canvasEl, comparisonRadioName = "plot-comparison") {
  const xVarEl = $(sidebar, '[data-testid="plot-xvar"]');
  const rowVarsEl = $(sidebar, '[data-testid="plot-rowvars"]');
  const colVarsEl = $(sidebar, '[data-testid="plot-colvars"]');
  const comparisonEosLabel = $(sidebar, '[data-testid="plot-comparison-eos-label"]');
  const comparisonSingleLabel = $(sidebar, '[data-testid="plot-comparison-single-label"]');
  const expOverlayEl = $(sidebar, '[data-testid="plot-exp-overlay"]');
  const darkModeEl = $(sidebar, '[data-testid="plot-dark-mode"]');
  const updateBtn = $(sidebar, '[data-testid="plot-update"]');
  const saveBtn = $(sidebar, '[data-testid="plot-save-png"]');

  let state = { resultsDict: {}, expData: null, seriesLabel: "EOS models" };

  function setResults(resultsDict, expData, seriesLabel, defaultRowVars, defaultColVars) {
    state = { resultsDict, expData: expData || null, seriesLabel: seriesLabel || "EOS models" };
    const columns = getNumericColumns(resultsDict);
    const xOpts = ["T (K)", "P_eq (MPa)"].filter((c) => columns.includes(c))
      .concat(columns.filter((c) => !["T (K)", "P_eq (MPa)"].includes(c)));
    xVarEl.innerHTML = "";
    for (const c of xOpts) {
      const o = document.createElement("option"); o.value = c; o.textContent = c;
      xVarEl.appendChild(o);
    }
    if (xOpts.length) xVarEl.value = xOpts[0];

    const eosNames = Object.keys(resultsDict);
    const defRow = defaultRowVars && defaultRowVars.length ? defaultRowVars : ["P_eq (MPa)"];
    const defCol = defaultColVars && defaultColVars.length ? defaultColVars : ["P_eq (MPa)"];
    fillMultiSelect(rowVarsEl, columns, defRow.filter((c) => columns.includes(c)));
    fillMultiSelect(colVarsEl, columns, defCol.filter((c) => columns.includes(c)));

    comparisonEosLabel.textContent = `Overlay all (${eosNames.length} ${state.seriesLabel})`;
    comparisonSingleLabel.textContent = `Single (${eosNames[0] || ""})`;

    render();
  }

  function getConfig() {
    const rows = getSelected(rowVarsEl);
    const cols = getSelected(colVarsEl);
    return {
      xVar: xVarEl.value || "T (K)",
      rowVars: rows.length ? rows : ["P_eq (MPa)"],
      colVars: cols.length ? cols : ["P_eq (MPa)"],
      comparison: $(sidebar, `input[name="${comparisonRadioName}"]:checked`).value,
      expOverlay: expOverlayEl.checked,
      darkMode: darkModeEl.checked,
    };
  }

  function render() {
    const cfg = getConfig();
    const { resultsDict, expData } = state;
    const eosNames = Object.keys(resultsDict);
    const models = cfg.comparison === "eos" ? eosNames : [eosNames[0]];

    const nrows = Math.max(cfg.rowVars.length, 1);
    const ncols = Math.max(cfg.colVars.length, 1);
    const dark = cfg.darkMode;

    const gapX = 0.06, gapY = 0.10;
    const cellW = (1 - gapX * (ncols - 1)) / ncols;
    const cellH = (1 - gapY * (nrows - 1)) / nrows;

    const traces = [];
    const annotations = [];
    const layout = {
      paper_bgcolor: dark ? "#1e1e2e" : "#ffffff",
      plot_bgcolor: dark ? "#181825" : "#fafafa",
      font: { color: dark ? "#cdd6f4" : "#18181b", size: 11 },
      margin: { l: 50, r: 20, t: 30, b: 40 },
      showlegend: false,
    };

    let axisIdx = 0;
    for (let r = 0; r < nrows; r++) {
      for (let c = 0; c < ncols; c++) {
        const yVar = cfg.colVars[c];
        axisIdx += 1;
        const xKey = axisIdx === 1 ? "xaxis" : `xaxis${axisIdx}`;
        const yKey = axisIdx === 1 ? "yaxis" : `yaxis${axisIdx}`;
        const xref = axisIdx === 1 ? "x" : `x${axisIdx}`;
        const yref = axisIdx === 1 ? "y" : `y${axisIdx}`;

        const x0 = c * (cellW + gapX);
        const x1 = x0 + cellW;
        const yTop = 1 - r * (cellH + gapY);
        const y1 = yTop;
        const y0 = yTop - cellH;

        const gridColor = dark ? "#313244" : "#e4e4e7";
        const lineColor = dark ? "#45475a" : "#d4d4d8";
        layout[xKey] = {
          domain: [x0, x1], anchor: yref, gridcolor: gridColor, linecolor: lineColor,
          title: r === nrows - 1 ? { text: cfg.xVar, font: { size: 10 } } : undefined,
          zeroline: false,
        };
        layout[yKey] = {
          domain: [y0, y1], anchor: xref, gridcolor: gridColor, linecolor: lineColor,
          title: { text: yVar, font: { size: 10 } }, zeroline: false,
        };

        let plotted = false;
        models.forEach((model, mIdx) => {
          const rows = resultsDict[model];
          if (!rows || !isNumericColumn(rows, cfg.xVar) || !isNumericColumn(rows, yVar)) return;
          const xs = [], ys = [];
          for (const row of rows) {
            const xv = row[cfg.xVar], yv = row[yVar];
            if (typeof xv === "number" && typeof yv === "number") { xs.push(xv); ys.push(yv); }
          }
          if (!xs.length) return;
          plotted = true;
          traces.push({
            x: xs, y: ys, xaxis: xref, yaxis: yref,
            type: "scatter", mode: "lines+markers",
            name: models.length > 1 ? model : yVar,
            legendgroup: model,
            showlegend: axisIdx === 1,
            line: { color: PALETTE[mIdx % PALETTE.length], width: 1.8 },
            marker: { symbol: SYMBOLS[mIdx % SYMBOLS.length], size: 6, color: PALETTE[mIdx % PALETTE.length] },
          });
        });

        if (cfg.expOverlay && expData && yVar === "P_eq (MPa)") {
          plotted = true;
          traces.push({
            x: expData["T (K)"], y: expData["P_eq (MPa)"], xaxis: xref, yaxis: yref,
            type: "scatter", mode: "markers", name: "Experimental",
            showlegend: axisIdx === 1,
            marker: { symbol: "x-thin", size: 9, color: EXP_COLOR, line: { width: 2, color: EXP_COLOR } },
          });
        }

        if (!plotted) {
          annotations.push({
            xref: `${xref} domain`, yref: `${yref} domain`, x: 0.5, y: 0.5,
            text: "No data available", showarrow: false,
            font: { color: dark ? "#585b70" : "#d4d4d8", size: 11 },
          });
        }
      }
    }
    layout.annotations = annotations;
    layout.legend = { font: { size: 10 } };

    Plotly.react(canvasEl, traces, layout, { responsive: true, displaylogo: false });
  }

  updateBtn.addEventListener("click", render);
  saveBtn.addEventListener("click", () => {
    const ts = new Date().toISOString().replace(/[-:T.]/g, "").slice(0, 14);
    Plotly.downloadImage(canvasEl, { format: "png", filename: `hydrate_plot_${ts}`, scale: 2 });
  });

  return { setResults, render };
}
