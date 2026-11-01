# AmsterCHEM CAPE-OPEN Python Unit Operation Guide

This directory contains the standalone files for integrating **kdarnell's Gibbs/vdW-P Hydrate Flash Engine** directly into **Aspen Plus V14** via the **AmsterCHEM Python Unit Operation** CAPE-OPEN interface.

---

## 1. Directory Contents

* **`Two_Stage_Hydrate_Capture_CAPE_OPEN.apw`**  
  Aspen Plus compound workbook (copy of the complete 2-stage capture flowsheet).
* **`Two_Stage_Hydrate_Capture_CAPE_OPEN.bkp`**  
  Aspen Plus backup file.
* **`hydrate_unit_op.py`**  
  The unified CAPE-OPEN entry script defining `Configure(unit)` and `Calculate(unit)` hooks for AmsterCHEM.
* **Thermodynamic Engine Modules:**
  * `component_properties.py`: EOS parameters, critical constants, and Kihara potentials.
  * `flashalgorithm.py`: Multiphase flash controller.
  * `h_vdwpm_eos.py`: van der Waals–Platteeuw hydrate solid-solution chemical potential model.
  * `aq_hb_eos.py`: Holder-Bishnoi aqueous activity model.
  * `vlhc_srk_eos.py`: Soave-Redlich-Kwong gas-phase fugacity solver.

---

## 2. Verified Python Environment

The Python interpreter has been pre-configured with all required dependencies (`numpy`, `scipy`, `pandas`):
* **Interpreter Path:** `C:\Python312\python.exe`
* Tested standalone execution:
  ```powershell
  python hydrate_unit_op.py --co2 0.40 --h2 0.60 --diox 0.0556 -T 274.15 -P 25.0 --water-ratio 0.85
  ```
  *(Status: Converged, Hydrate Formed: Structure II with 97.76% CO2 selectivity)*

---

## 3. How to Connect the Python Unit Operation in Aspen Plus

### Step 1: Open the Flowsheet
Open `Two_Stage_Hydrate_Capture_CAPE_OPEN.apw` in Aspen Plus V14.

### Step 2: Add the CAPE-OPEN Block
1. In the bottom **Model Palette**, select the **`CAPE-OPEN`** tab.
2. Click and drag the **`Python Unit Operation`** icon onto the flowsheet canvas (e.g., name it `HYD-RX1`).

### Step 3: Configure the Script Path
1. Double-click the newly placed `HYD-RX1` block (or right-click $\rightarrow$ **Edit / Configure**).
2. The AmsterCHEM configuration window will open:
   * **Script File:** Browse and select:
     ```text
     C:\Users\hskus\Documents\MTP\BTP-II\hydrate_project\aspen_simulation\cape_open_simulation\hydrate_unit_op.py
     ```
   * **Python Executable / DLL:** Ensure it points to `C:\Python312\python.exe` (or `python312.dll`).
3. Click **Configure / Reload Script**.
   * AmsterCHEM will call `Configure(unit)` and automatically generate the 4 standard ports:
     * **Inlet:** `Feed`
     * **Outlets:** `Vapor_Out`, `Hydrate_Out`, `Liquid_Out`

### Step 4: Connect Streams
Connect the streams to replace `FORM1` (or test in parallel):
* **Inlet Stream:** Connect `S4` (mixed feed) $\rightarrow$ `Feed` port.
* **Vapor Outlet:** Connect `Vapor_Out` $\rightarrow$ `S5` (overhead clean gas).
* **Hydrate Slurry Outlet:** Connect `Hydrate_Out` (or combined `Hydrate_Out` + `Liquid_Out` via a Mixer) $\rightarrow$ `S6` (slurry stream leading to `VALV1`).

### Step 5: Run the Simulation
Click **Run (F5)**.
* Aspen Plus will pass stream thermodynamic state $(T, P, \text{Flow}, z_i)$ to `Calculate(unit)`.
* `hydrate_unit_op.py` solves the dynamic Gibbs/vdW-P flash algorithm and returns product stream states back to Aspen Plus in real time.
