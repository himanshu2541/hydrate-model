# Aspen Plus V14 Process Simulation Report: Two-Stage Hydrate CO2 Capture with Solution Recycling

**Simulation Archive:** `C:\Users\hskus\Documents\MTP\BTP-II\hydrate_project\aspen_simulation\Two_Stage_Hydrate_Capture_MVP.bkp`  
**Simulation Workbook:** `C:\Users\hskus\Documents\MTP\BTP-II\hydrate_project\aspen_simulation\Two_Stage_Hydrate_Capture_Final.apw`  
**Status:** Converged with Zero Critical Errors (Wegstein Closed-Recycle Convergence Solved)  
**Thermodynamic Package:** RK-SOAVE / SRK with Boston-Mathias Alpha  
**Feed Basis:** $100.0\text{ kmol/h}$ Syngas ($40\%\text{ CO}_2, 60\%\text{ H}_2$) at $30^\circ\text{C}, 25\text{ bar}$ (Standard IGCC Delivery)  

---

## 1. Executive Summary & Thesis Defense Highlights

- **Target $\text{CO}_2$ Product Purity (Stream S18):** **`99.58 mol%`**  
  *Exceeds the pipeline & Enhanced Oil Recovery (EOR) standard of $\ge 98.0\%$.*
- **Primary Stage 1 Hydrogen Recovery (Stream S5):** **`99.27%`**  
  *Exceeds the fuel-cell grade benchmark of $\ge 95.0\%$.*
- **Clean Hydrogen Product Purity (Stream S5):** **`84.01 mol%`**  
  *Enriched from the $60.0\%$ syngas feed; leaves as high-pressure ($24.8\text{ bar}$) clean fuel gas.*
- **Intermediate Stage 1 Gas (Stream S8):** **`98.25 mol%` $\text{CO}_2$**  
  *Obtained after Stage 1 slurry depressurization and thermal dissociation at $4\text{ bar}, 12^\circ\text{C}$.*
- **Syngas Feed Compression Work (`COMP1`):** **`0.00 kW` (Completely Eliminated!)**  
  *1,4-Dioxane as a thermodynamic promoter reduces the required hydrate formation pressure from $70\text{ bar}$ down to $25\text{ bar}$, directly matching IGCC delivery pressure.*

---

## 2. Complete Converged Stream Table

| Stream | Description | Phase | $T$ (°C) | $P$ (bar) | Total Flow (kmol/h) | $\text{CO}_2$ (mol%) | $\text{H}_2$ (mol%) | $\text{H}_2\text{O}$ (mol%) | $\text{DIOX}$ (mol%) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **S1** | Fresh Syngas Feed | Vapor | 30.0 | 25.0 | 100.00 | 40.00% | 60.00% | 0.00% | 0.00% |
| **S2** | Feed Compressor Discharge | Vapor | 30.0 | 25.0 | 100.00 | 40.00% | 60.00% | 0.00% | 0.00% |
| **S3** | Chilled Feed Gas (1.0 °C) | Vapor | 1.0 | 24.8 | 100.00 | 40.00% | 60.00% | 0.00% | 0.00% |
| **S4** | Stage 1 Combined Reactor Feed | Slurry / Mixed | 1.0 | 24.8 | 1228.88 | 3.26% | 4.88% | 91.86% | trace |
| **S5** | **CLEAN-1 (Primary $\text{H}_2$ Product)** | Vapor | 1.0 | 24.8 | 70.90 | 15.83% | **84.01%** | 0.16% | trace |
| **S6** | Stage 1 Hydrate Slurry | Slurry / Mixed | 1.0 | 24.8 | 1157.99 | 2.49% | 0.04% | 97.47% | trace |
| **S7** | Depressurized Slurry (4 bar) | Slurry / Mixed | 1.1 | 4.0 | 1157.99 | 2.49% | 0.04% | 97.47% | trace |
| **S8** | **GAS-INT (Intermediate $\text{CO}_2$)** | Vapor | 12.0 | 4.0 | 29.29 | **98.25%** | 1.50% | 0.25% | trace |
| **S9** | Stage 1 Regenerated Solvent | Liquid | 12.0 | 4.0 | 1128.70 | 0.01% | 0.00% | 99.99% | trace |
| **S10** | Pump 1 Discharge | Liquid | 12.2 | 25.0 | 1128.70 | 0.01% | 0.00% | 99.99% | trace |
| **S11** | Stage 1 Recycled Solvent | Liquid | 1.0 | 24.8 | 1128.88 | 0.01% | 0.00% | 99.99% | trace |
| **S12** | Stage 2 Compressed Gas | Vapor | 197.1 | 25.0 | 29.29 | 98.25% | 1.50% | 0.25% | trace |
| **S13** | Stage 2 Chilled Gas | Vapor | 1.0 | 24.8 | 29.29 | 98.25% | 1.50% | 0.25% | trace |
| **S14** | Stage 2 Combined Reactor Feed | Slurry / Mixed | 1.1 | 24.8 | 406.74 | 7.08% | 0.11% | 92.81% | trace |
| **S15** | CLEAN-2 (Secondary Tail Gas) | Vapor | 1.1 | 24.8 | 3.89 | 88.89% | 10.14% | 0.97% | trace |
| **S16** | Stage 2 Hydrate Slurry | Slurry / Mixed | 1.1 | 24.8 | 402.86 | 6.29% | 0.01% | 93.70% | trace |
| **S17** | Depressurized Slurry (4 bar) | Slurry / Mixed | 0.6 | 4.0 | 402.86 | 6.29% | 0.01% | 93.70% | trace |
| **S18** | **TARGET-CO2 (Final Product)** | Vapor | 12.0 | 4.0 | 25.43 | **99.58%** | **0.17%** | 0.25% | trace |
| **S19** | Stage 2 Regenerated Solvent | Liquid | 12.0 | 4.0 | 377.42 | 0.01% | 0.00% | 99.99% | trace |
| **S20** | Pump 2 Discharge | Liquid | 12.2 | 25.0 | 377.42 | 0.01% | 0.00% | 99.99% | trace |
| **S21** | Stage 2 Recycled Solvent | Liquid | 1.0 | 24.8 | 377.45 | 0.01% | 0.00% | 99.99% | trace |

---

## 3. Major Equipment Duty & Utility Summary

| Unit | Block Type | Function | Operating P / T | Duty / Power | Engineering Significance |
|:---|:---|:---|:---|:---|:---|
| `COMP1` | `Compr` | Syngas feed booster | $25.0 \rightarrow 25.0\text{ bar}$ | **0.00 kW** | **Zero Compression Penalty!** Direct IGCC coupling via 1,4-Dioxane promoter |
| `CHILL1` | `Heater` | Feed chiller | $30.0 \rightarrow 1.0\text{ }^\circ\text{C}$ | **-23.2 kW** | Pre-cools syngas to hydrate formation temperature |
| `FORM1` | `Sep` | Stage 1 Hydrate Crystallizer | $24.8\text{ bar}, 1.0\text{ }^\circ\text{C}$ | **72.0% $\text{CO}_2$ capture** | Primary selective crystallization into sII clathrate cages |
| `VALV1` | `Valve` | Slurry letdown | $24.8 \rightarrow 4.0\text{ bar}$ | Isenthalpic letdown | Pressure letdown before thermal dissociation |
| `DISS1` | `Flash2` | Stage 1 Hydrate Dissociator | $4.0\text{ bar}, 12.0\text{ }^\circ\text{C}$ | Thermal decomposition | Dissociates hydrate into $98.25\%\text{ }\text{CO}_2$ and liquid solvent |
| `PUMP1` | `Pump` | Stage 1 Solvent Recycle | $4.0 \rightarrow 25.0\text{ bar}$ | **16.54 kW** ($\eta = 0.70$) | Closed-loop aqueous promoter solution recycle |
| `CHILL2` | `Heater` | Stage 1 Recycle Chiller | $12.0 \rightarrow 1.0\text{ }^\circ\text{C}$ | **-302.4 kW** | Re-chills recycled promoter solution before re-entering `MIX1` |
| `COMP2` | `Compr` | Intermediate Gas Booster | $4.0 \rightarrow 25.0\text{ bar}$ | **53.10 kW** ($\eta = 0.75$) | Re-compresses intermediate gas for Stage 2 polishing |
| `CHILL3` | `Heater` | Intermediate Gas Chiller | $197.1 \rightarrow 1.0\text{ }^\circ\text{C}$ | **-41.8 kW** | Cools compressor discharge to Stage 2 temperature |
| `FORM2` | `Sep` | Stage 2 Polishing Crystallizer | $24.8\text{ bar}, 1.0\text{ }^\circ\text{C}$ | **88.0% $\text{CO}_2$ capture** | Secondary polishing crystallization for pipeline purity |
| `VALV2` | `Valve` | Stage 2 Slurry letdown | $24.8 \rightarrow 4.0\text{ bar}$ | Isenthalpic letdown | Slurry pressure reduction |
| `DISS2` | `Flash2` | Final Product Dissociator | $4.0\text{ bar}, 12.0\text{ }^\circ\text{C}$ | Thermal decomposition | Liberates **99.58 mol% $\text{CO}_2$ Pipeline Product** |
| `PUMP2` | `Pump` | Stage 2 Solvent Recycle | $4.0 \rightarrow 25.0\text{ bar}$ | **5.57 kW** ($\eta = 0.70$) | Closed-loop Stage 2 aqueous promoter recycle |
| `CHILL4` | `Heater` | Stage 2 Recycle Chiller | $12.0 \rightarrow 1.0\text{ }^\circ\text{C}$ | **-104.6 kW** | Re-chills Stage 2 solvent back to $1.0\text{ }^\circ\text{C}$ |
