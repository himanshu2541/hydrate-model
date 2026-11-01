# Two-Stage Gas Hydrate CO2 Capture Simulation Data Report

**Flowsheet Model:** Two-Stage Hydrate Capture with Dual CAPE-OPEN Python Unit Operations (`B1` & `B2`)
**Thermodynamic Engine:** Gibbs Free Energy Minimization + van der Waals–Platteeuw (vdW-P) Model (kdarnell)
**Host Simulator:** Aspen Plus V14.0 (RK-SOAVE Property Package)

---

## 1. Executive Summary & Key Performance Indicators (KPIs)

| Key Performance Metric | Value | Unit | Benchmark / Design Target |
| :--- | :---: | :---: | :---: |
| **Target $\text{CO}_2$ Product Purity (`S18`)** | **`99.52`** | **mol%** | $>95.0\text{ mol}\%$ (High Purity) |
| **Target $\text{CO}_2$ Product Purity (`S18`)** | **`99.77`** | **wt%** | $>95.0\text{ wt}\%$ |
| **Overall Plant $\text{CO}_2$ Capture Recovery** | **`63.31`** | **%** | $>60.0\text{ %}$ (2-Stage Cascade) |
| **Primary Clean $\text{H}_2$ Purity (`S5`)** | **`84.01`** | **mol%** | Fuel-cell / Pre-combustion spec |
| **Primary Clean $\text{H}_2$ Recovery (`S5`)** | **`99.27`** | **%** | $>95.0\text{ %}$ |
| **Intermediate Gas $\text{CO}_2$ Purity (`S8`)** | **`98.22`** | **mol%** | Stage 1 Enrichment |
| **Total Net $\text{CO}_2$ Captured** | **`1114.48`** | **kg/h** | $25.32\text{ kmol/h}$ |
| **Total Compression Power (`COMP1` + `COMP2`)** | **`71.54`** | **kW** | Mechanical shaft work |
| **Total Pumping Power (`PUMP1` + `PUMP2`)** | **`22.54`** | **kW** | Solvent circulation work |
| **Total Plant Electrical Power** | **`94.08`** | **kW** | Compressors + Pumps |
| **Specific Electrical Energy Consumption** | **`0.0844`** | **kWh/kg $\text{CO}_2$** | `84.42\text{ kWh/tonne}$ |
| **Specific Compression Energy** | **`0.3039`** | **MJ/kg $\text{CO}_2$** | Low parasitic loss |
| **Total Chilling Duty (`CHILL1-4`)** | **`522.09`** | **kW** | Refrigeration to $1.0\text{ }^\circ\text{C}$ |
| **Total Dissociation Duty (`DISS1-2`)** | **`411.80`** | **kW** | Low-grade heat at $12.0\text{ }^\circ\text{C}$ |
| **Fresh Solvent Make-up (`MAKEUP1` + `MAKEUP2`)** | **`6.14`** | **kg/h** | $0.28\text{ kmol/h}$ solvent replenishment |

---

## 2. Complete Stream Table (All 23 Streams)

| Stream | Description | Phase | $T$ [°C] | $P$ [bar] | Mole Flow [kmol/h] | Mass Flow [kg/h] | $\text{CO}_2$ mol% | $\text{H}_2$ mol% | $\text{H}_2\text{O}$ mol% | $\text{DIOX}$ mol% | $\text{CO}_2$ wt% | $\text{H}_2$ wt% |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`S1`** | Fresh Syngas Feed (Boundary Inlet) | Vapor | 30.0 | 22.0 | 100.00 | 1881.36 | 40.00% | 60.00% | 0.00% | 0.00% | 93.57% | 6.43% |
| **`MAKEUP1`** | Stage 1 Solvent Make-up (Boundary Inlet) | Liquid | 1.0 | 24.8 | 0.18 | 3.94 | 0.00% | 0.00% | 94.44% | 5.56% | 0.00% | 0.00% |
| **`S2`** | Feed Compressor Discharge | Vapor | 44.01 | 25.0 | 100.00 | 1881.36 | 40.00% | 60.00% | 0.00% | 0.00% | 93.57% | 6.43% |
| **`S3`** | Chilled Syngas Feed (to MIX1) | Vapor | 1.0 | 24.8 | 100.00 | 1881.36 | 40.00% | 60.00% | 0.00% | 0.00% | 93.57% | 6.43% |
| **`S11`** | Stage 1 Recycled Aqueous Solvent (to MIX1) | Liquid | 1.0 | 24.8 | 1132.92 | 20411.42 | 0.01% | 0.00% | 99.99% | 0.00% | 0.02% | 0.00% |
| **`S4`** | Stage 1 Combined Reactor Feed (B1 Inlet) | Two-Phase (V=0.081) | 0.98 | 24.8 | 1233.10 | 22296.73 | 3.25% | 4.87% | 91.88% | 0.00% | 7.91% | 0.54% |
| **`S5`** | Primary Clean H2 Overhead Product (Boundary Outlet) | Two-Phase (V=0.999) | 1.0 | 25.0 | 70.90 | 615.93 | 15.83% | 84.01% | 0.16% | 0.00% | 80.17% | 19.50% |
| **`S6`** | Stage 1 Hydrate Slurry (B1 Bottoms) | Two-Phase (V=0.025) | 1.0 | 25.0 | 1162.20 | 21680.81 | 2.48% | 0.04% | 97.48% | 0.00% | 5.86% | 0.00% |
| **`S7`** | Depressurized Slurry (VALV1 Outlet) | Two-Phase (V=0.025) | 1.13 | 4.0 | 1162.20 | 21680.81 | 2.48% | 0.04% | 97.48% | 0.00% | 5.86% | 0.00% |
| **`S8`** | Intermediate CO2 Flash Gas (to Stage 2) | Vapor | 12.0 | 4.0 | 29.30 | 1269.69 | 98.22% | 1.49% | 0.25% | 0.03% | 99.76% | 0.07% |
| **`S9`** | Stage 1 Regenerated Solvent Liquid (DISS1 Bottoms) | Liquid | 12.0 | 4.0 | 1132.90 | 20411.12 | 0.01% | 0.00% | 99.99% | 0.00% | 0.02% | 0.00% |
| **`S10`** | Stage 1 Pump Discharge | Liquid | 12.15 | 25.0 | 1132.90 | 20411.12 | 0.01% | 0.00% | 99.99% | 0.00% | 0.02% | 0.00% |
| **`MAKEUP2`** | Stage 2 Solvent Make-up (Boundary Inlet) | Liquid | 1.0 | 24.8 | 0.10 | 2.19 | 0.00% | 0.00% | 94.44% | 5.56% | 0.00% | 0.00% |
| **`S12`** | Intermediate Gas Compressor Discharge | Vapor | 196.99 | 25.0 | 29.30 | 1269.69 | 98.22% | 1.49% | 0.25% | 0.03% | 99.76% | 0.07% |
| **`S13`** | Chilled Intermediate Gas (to MIX2) | Two-Phase (V=0.998) | 1.0 | 24.8 | 29.30 | 1269.69 | 98.22% | 1.49% | 0.25% | 0.03% | 99.76% | 0.07% |
| **`S21`** | Stage 2 Recycled Aqueous Solvent (to MIX2) | Liquid | 1.0 | 24.8 | 378.07 | 6811.58 | 0.01% | 0.00% | 99.99% | 0.00% | 0.02% | 0.00% |
| **`S14`** | Stage 2 Combined Reactor Feed (B2 Inlet) | Two-Phase (V=0.072) | 0.97 | 24.8 | 407.47 | 8083.46 | 7.07% | 0.11% | 92.82% | 0.00% | 15.68% | 0.01% |
| **`S15`** | Secondary Tail Gas (Boundary Outlet) | Two-Phase (V=0.991) | 1.0 | 25.0 | 3.89 | 153.60 | 88.89% | 10.14% | 0.97% | 0.00% | 99.04% | 0.52% |
| **`S16`** | Stage 2 Hydrate Slurry (B2 Bottoms) | Two-Phase (V=0.063) | 1.0 | 25.0 | 403.58 | 7929.86 | 6.28% | 0.01% | 93.70% | 0.00% | 14.07% | 0.00% |
| **`S17`** | Depressurized Slurry (VALV2 Outlet) | Two-Phase (V=0.063) | 0.54 | 4.0 | 403.58 | 7929.86 | 6.28% | 0.01% | 93.70% | 0.00% | 14.07% | 0.00% |
| **`S18`** | Target CO2 Final Product Gas (Boundary Outlet) | Vapor | 12.0 | 4.0 | 25.45 | 1117.09 | 99.52% | 0.17% | 0.25% | 0.06% | 99.77% | 0.01% |
| **`S19`** | Stage 2 Regenerated Solvent Liquid (DISS2 Bottoms) | Liquid | 12.0 | 4.0 | 378.14 | 6812.77 | 0.01% | 0.00% | 99.99% | 0.00% | 0.02% | 0.00% |
| **`S20`** | Stage 2 Pump Discharge | Liquid | 12.15 | 25.0 | 378.14 | 6812.77 | 0.01% | 0.00% | 99.99% | 0.00% | 0.02% | 0.00% |

---

## 3. Unit Operation Performance (All 16 Blocks)

| Block ID | Category | Model Type | Thermal Duty [Gcal/hr] | Thermal Duty [kW] | Electrical Power [kW] |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **`B1`** | Hydrate Formation Reactor | Unit Operation | 0.0000 | 0.00 | 0.00 |
| **`B2`** | Hydrate Formation Reactor | Unit Operation | 0.0000 | 0.00 | 0.00 |
| **`CHILL1`** | Chilling / Refrigeration | Unit Operation | -0.0346 | 40.25 | 0.00 |
| **`CHILL2`** | Chilling / Refrigeration | Unit Operation | -0.2640 | 307.03 | 0.00 |
| **`CHILL3`** | Chilling / Refrigeration | Unit Operation | -0.0622 | 72.33 | 0.00 |
| **`CHILL4`** | Chilling / Refrigeration | Unit Operation | -0.0881 | 102.48 | 0.00 |
| **`COMP1`** | Compression | ISENTROPIC | 0.0000 | 0.00 | 12.57 |
| **`COMP2`** | Compression | ISENTROPIC | 0.0000 | 0.00 | 58.97 |
| **`DISS1`** | Hydrate Dissociation (Heating) | Unit Operation | 0.2605 | 302.94 | 0.00 |
| **`DISS2`** | Hydrate Dissociation (Heating) | Unit Operation | 0.0936 | 108.86 | 0.00 |
| **`MIX1`** | Feed Mixer | Unit Operation | 0.0000 | 0.00 | 0.00 |
| **`MIX2`** | Feed Mixer | Unit Operation | 0.0000 | 0.00 | 0.00 |
| **`PUMP1`** | Pumping | Unit Operation | 0.0000 | 0.00 | 16.90 |
| **`PUMP2`** | Pumping | Unit Operation | 0.0000 | 0.00 | 5.64 |
| **`VALV1`** | Depressurization Valve | Unit Operation | 0.0000 | 0.00 | 0.00 |
| **`VALV2`** | Depressurization Valve | Unit Operation | 0.0000 | 0.00 | 0.00 |

---

## 4. Overall Plant Material & Component Balances

### 4.1 Carbon Dioxide ($	ext{CO}_2$) Balance
* **Total In:** Syngas `S1` = **`40.0000 kmol/h`** (`1760.40 kg/h`)
* **Outlets:**
  * Primary Clean Gas (`S5`): `11.2204 kmol/h`
  * Secondary Tail Gas (`S15`): `3.4565 kmol/h`
  * Target $\text{CO}_2$ Product (`S18`): `25.3233 kmol/h` (`1114.48 kg/h`)
* **Total Out:** **`40.0002 kmol/h`** (Closure: **`100.00%`**)

### 4.2 Hydrogen ($	ext{H}_2$) Balance
* **Total In:** Syngas `S1` = **`60.0000 kmol/h`** (`120.96 kg/h`)
* **Outlets:**
  * Primary Clean $\text{H}_2$ Product (`S5`): `59.5620 kmol/h` (Recovery: **`99.27%`**)
  * Secondary Tail Gas (`S15`): `0.3942 kmol/h`
  * Target $\text{CO}_2$ Product (`S18`): `0.0438 kmol/h`
* **Total Out:** **`60.0000 kmol/h`** (Closure: **`100.00%`**)

---

## 5. Scope & Roadmap for Upcoming Sensitivity Analysis

The current baseline simulation is fully verified and converged. For the sensitivity analysis section of the thesis, the following parametric sweeps are ready to be conducted:

1. **Stage 1 & 2 Operating Pressure ($P = 15 - 35\text{ bar}$):**
   * *Impact on:* Hydrate equilibrium formation temperature, required refrigeration power vs. compression work.
2. **Operating Temperature ($T = 273.15 - 278.15\text{ K}$ / $0 - 5\text{ }^\circ\text{C}$):**
   * *Impact on:* Chilling duty, hydrate crystallizer volume, stability margin.
3. **1,4-Dioxane Promoter Concentration ($0.0 - 5.56\text{ mol}\%$):**
   * *Impact on:* Thermodynamic equilibrium shift from Structure I to Structure II, enabling mild operating pressures ($<25\text{ bar}$ vs. $>60\text{ bar}$ without promoter).
4. **Solvent-to-Feed Ratio ($L/G = 5 - 15\text{ mol/mol}$):**
   * *Impact on:* Slurry pump power, reactor kinetics, heat of formation absorption.
5. **Dissociator Pressure & Temperature ($P = 2 - 8\text{ bar}$, $T = 8 - 20\text{ }^\circ\text{C}$):**
   * *Impact on:* Intermediate compression ratio (`COMP2`), product $\text{CO}_2$ delivery pressure, dissociation heat requirement.
