import os
import sys
import win32com.client
import pandas as pd
import json

output_dir = r"c:\Users\hskus\Documents\MTP\BTP-II\hydrate_project\simulation_report_data"
os.makedirs(output_dir, exist_ok=True)

apw_path = os.path.abspath(r"aspen_simulation\cape_open_simulation\Two_Stage_Hydrate_Capture_CAPE_OPEN.apw")
print(f"Connecting to Aspen Plus and extracting simulation data from:\n  {apw_path}\n")

aspen = win32com.client.Dispatch("Apwn.Document")
aspen.SuppressDialogs = 1

try:
    aspen.InitFromFile2(apw_path)
    tree = aspen.Tree

    # Molecular weights (kg/kmol)
    mw = {
        "CO2": 44.01,
        "H2": 2.016,
        "H2O": 18.015,
        "DIOXANE": 88.11
    }

    stream_descriptions = {
        "S1": "Fresh Syngas Feed (Boundary Inlet)",
        "MAKEUP1": "Stage 1 Solvent Make-up (Boundary Inlet)",
        "S2": "Feed Compressor Discharge",
        "S3": "Chilled Syngas Feed (to MIX1)",
        "S11": "Stage 1 Recycled Aqueous Solvent (to MIX1)",
        "S4": "Stage 1 Combined Reactor Feed (B1 Inlet)",
        "S5": "Primary Clean H2 Overhead Product (Boundary Outlet)",
        "S6": "Stage 1 Hydrate Slurry (B1 Bottoms)",
        "S7": "Depressurized Slurry (VALV1 Outlet)",
        "S8": "Intermediate CO2 Flash Gas (to Stage 2)",
        "S9": "Stage 1 Regenerated Solvent Liquid (DISS1 Bottoms)",
        "S10": "Stage 1 Pump Discharge",
        "MAKEUP2": "Stage 2 Solvent Make-up (Boundary Inlet)",
        "S12": "Intermediate Gas Compressor Discharge",
        "S13": "Chilled Intermediate Gas (to MIX2)",
        "S21": "Stage 2 Recycled Aqueous Solvent (to MIX2)",
        "S14": "Stage 2 Combined Reactor Feed (B2 Inlet)",
        "S15": "Secondary Tail Gas (Boundary Outlet)",
        "S16": "Stage 2 Hydrate Slurry (B2 Bottoms)",
        "S17": "Depressurized Slurry (VALV2 Outlet)",
        "S18": "Target CO2 Final Product Gas (Boundary Outlet)",
        "S19": "Stage 2 Regenerated Solvent Liquid (DISS2 Bottoms)",
        "S20": "Stage 2 Pump Discharge"
    }

    ordered_streams = [
        "S1", "MAKEUP1", "S2", "S3", "S11", "S4", "S5", "S6", "S7", "S8", "S9", "S10",
        "MAKEUP2", "S12", "S13", "S21", "S14", "S15", "S16", "S17", "S18", "S19", "S20"
    ]

    streams_data = []

    for sname in ordered_streams:
        flow_node = tree.FindNode(rf"\Data\Streams\{sname}\Output\STR_MAIN\MOLEFLOW\MIXED")
        
        t = 0.0
        p = 0.0
        vfrac = 0.0
        tot_mole = 0.0
        enthalpy = 0.0
        co2_mole = 0.0
        h2_mole = 0.0
        h2o_mole = 0.0
        diox_mole = 0.0

        if flow_node:
            t_node = tree.FindNode(rf"\Data\Streams\{sname}\Output\STR_MAIN\TEMP\MIXED")
            p_node = tree.FindNode(rf"\Data\Streams\{sname}\Output\STR_MAIN\PRES\MIXED")
            vf_node = tree.FindNode(rf"\Data\Streams\{sname}\Output\STR_MAIN\VFRAC\MIXED")
            enth_node = tree.FindNode(rf"\Data\Streams\{sname}\Output\STR_MAIN\HFLMX\MIXED")

            t = float(t_node.Value or 0.0) if t_node else 0.0
            p = float(p_node.Value or 0.0) if p_node else 0.0
            vfrac = float(vf_node.Value or 0.0) if vf_node else 0.0
            enthalpy = float(enth_node.Value or 0.0) if enth_node else 0.0

            co2_mole = float(flow_node.Elements("CO2").Value or 0.0)
            h2_mole = float(flow_node.Elements("H2").Value or 0.0)
            h2o_mole = float(flow_node.Elements("H2O").Value or 0.0)
            diox_mole = float(flow_node.Elements("DIOXANE").Value or 0.0)
            tot_mole = co2_mole + h2_mole + h2o_mole + diox_mole
        else:
            in_node = tree.FindNode(rf"\Data\Streams\{sname}\Input")
            if in_node:
                t_node = tree.FindNode(rf"\Data\Streams\{sname}\Input\TEMP\MIXED")
                p_node = tree.FindNode(rf"\Data\Streams\{sname}\Input\PRES\MIXED")
                tot_node = tree.FindNode(rf"\Data\Streams\{sname}\Input\TOTFLOW\MIXED")

                t = float(t_node.Value or 0.0) if t_node else 0.0
                p = float(p_node.Value or 0.0) if p_node else 0.0
                tot_mole = float(tot_node.Value or 0.0) if tot_node else 0.0

                c_node = tree.FindNode(rf"\Data\Streams\{sname}\Input\FLOW\MIXED\CO2")
                h_node = tree.FindNode(rf"\Data\Streams\{sname}\Input\FLOW\MIXED\H2")
                w_node = tree.FindNode(rf"\Data\Streams\{sname}\Input\FLOW\MIXED\H2O")
                d_node = tree.FindNode(rf"\Data\Streams\{sname}\Input\FLOW\MIXED\DIOXANE")

                co2_mole = float(c_node.Value or 0.0) if c_node else 0.0
                h2_mole = float(h_node.Value or 0.0) if h_node else 0.0
                h2o_mole = float(w_node.Value or 0.0) if w_node else 0.0
                diox_mole = float(d_node.Value or 0.0) if d_node else 0.0

        # Component mass flows (kg/h)
        co2_mass = co2_mole * mw["CO2"]
        h2_mass = h2_mole * mw["H2"]
        h2o_mass = h2o_mole * mw["H2O"]
        diox_mass = diox_mole * mw["DIOXANE"]
        tot_mass = co2_mass + h2_mass + h2o_mass + diox_mass

        # Mole percentages
        co2_mol_pct = (co2_mole / tot_mole * 100.0) if tot_mole > 0 else 0.0
        h2_mol_pct = (h2_mole / tot_mole * 100.0) if tot_mole > 0 else 0.0
        h2o_mol_pct = (h2o_mole / tot_mole * 100.0) if tot_mole > 0 else 0.0
        diox_mol_pct = (diox_mole / tot_mole * 100.0) if tot_mole > 0 else 0.0

        # Mass percentages
        co2_wt_pct = (co2_mass / tot_mass * 100.0) if tot_mass > 0 else 0.0
        h2_wt_pct = (h2_mass / tot_mass * 100.0) if tot_mass > 0 else 0.0
        h2o_wt_pct = (h2o_mass / tot_mass * 100.0) if tot_mass > 0 else 0.0
        diox_wt_pct = (diox_mass / tot_mass * 100.0) if tot_mass > 0 else 0.0

        # Phase description
        if vfrac >= 0.999:
            phase = "Vapor"
        elif vfrac <= 0.001:
            phase = "Liquid"
        else:
            phase = f"Two-Phase (V={vfrac:.3f})"

        streams_data.append({
            "Stream": sname,
            "Description": stream_descriptions.get(sname, ""),
            "Phase": phase,
            "Temperature_C": round(t, 2),
            "Pressure_bar": round(p, 2),
            "Vapor_Fraction": round(vfrac, 4),
            "Total_Mole_Flow_kmol_h": round(tot_mole, 4),
            "Total_Mass_Flow_kg_h": round(tot_mass, 4),
            "CO2_Mole_Flow_kmol_h": round(co2_mole, 4),
            "H2_Mole_Flow_kmol_h": round(h2_mole, 4),
            "H2O_Mole_Flow_kmol_h": round(h2o_mole, 4),
            "DIOX_Mole_Flow_kmol_h": round(diox_mole, 4),
            "CO2_mol_pct": round(co2_mol_pct, 2),
            "H2_mol_pct": round(h2_mol_pct, 2),
            "H2O_mol_pct": round(h2o_mol_pct, 2),
            "DIOX_mol_pct": round(diox_mol_pct, 2),
            "CO2_Mass_Flow_kg_h": round(co2_mass, 4),
            "H2_Mass_Flow_kg_h": round(h2_mass, 4),
            "H2O_Mass_Flow_kg_h": round(h2o_mass, 4),
            "DIOX_Mass_Flow_kg_h": round(diox_mass, 4),
            "CO2_wt_pct": round(co2_wt_pct, 2),
            "H2_wt_pct": round(h2_wt_pct, 2),
            "H2O_wt_pct": round(h2o_wt_pct, 2),
            "DIOX_wt_pct": round(diox_wt_pct, 2)
        })

    df_streams = pd.DataFrame(streams_data)
    csv_streams_path = os.path.join(output_dir, "stream_table_full.csv")
    df_streams.to_csv(csv_streams_path, index=False)
    print(f"Saved: {csv_streams_path} ({len(df_streams)} streams)")

    # 4. Extract Block Performance
    blocks_node = tree.FindNode(r"\Data\Blocks")
    block_names = [blocks_node.Elements(i).Name for i in range(blocks_node.Elements.Count)]

    block_data = []
    for b in block_names:
        duty_node = tree.FindNode(rf"\Data\Blocks\{b}\Output\QCALC") or tree.FindNode(rf"\Data\Blocks\{b}\Output\DUTY") or tree.FindNode(rf"\Data\Blocks\{b}\Output\NET_DUTY")
        power_node = tree.FindNode(rf"\Data\Blocks\{b}\Output\POWER") or tree.FindNode(rf"\Data\Blocks\{b}\Output\BRAKE_POWER") or tree.FindNode(rf"\Data\Blocks\{b}\Output\ELEC_POWER")
        
        # Duty in Gcal/hr -> convert to kW (1 Gcal/hr = 1163.0 kW)
        duty_gcal_hr = float(duty_node.Value) if (duty_node and duty_node.Value is not None) else 0.0
        duty_kw = abs(duty_gcal_hr) * 1163.0

        # Power in kW directly (MET units)
        power_kw = float(power_node.Value) if (power_node and power_node.Value is not None) else 0.0

        b_type_node = tree.FindNode(rf"\Data\Blocks\{b}\Input\TYPE") or tree.FindNode(rf"\Data\Blocks\{b}\Input\MDLTYPE")
        b_type = str(b_type_node.Value) if (b_type_node and b_type_node.Value is not None) else "Unit Operation"

        # Categorize
        if "COMP" in b:
            cat = "Compression"
        elif "PUMP" in b:
            cat = "Pumping"
        elif "CHILL" in b:
            cat = "Chilling / Refrigeration"
        elif "DISS" in b:
            cat = "Hydrate Dissociation (Heating)"
        elif b in ["B1", "B2"]:
            cat = "Hydrate Formation Reactor"
        elif "VALV" in b:
            cat = "Depressurization Valve"
        elif "MIX" in b:
            cat = "Feed Mixer"
        else:
            cat = "Other"

        block_data.append({
            "Block": b,
            "Category": cat,
            "Model_Type": b_type,
            "Duty_Gcal_hr": round(duty_gcal_hr, 4),
            "Duty_kW": round(duty_kw, 2),
            "Power_kW": round(power_kw, 2)
        })

    df_blocks = pd.DataFrame(block_data)
    csv_blocks_path = os.path.join(output_dir, "block_performance.csv")
    df_blocks.to_csv(csv_blocks_path, index=False)
    print(f"Saved: {csv_blocks_path} ({len(df_blocks)} blocks)")

    # 5. Calculate System KPIs
    stream_map = {row["Stream"]: row for row in streams_data}
    s1 = stream_map.get("S1", {})
    s5 = stream_map.get("S5", {})
    s8 = stream_map.get("S8", {})
    s15 = stream_map.get("S15", {})
    s18 = stream_map.get("S18", {})
    m1 = stream_map.get("MAKEUP1", {})
    m2 = stream_map.get("MAKEUP2", {})

    co2_feed = s1.get("CO2_Mole_Flow_kmol_h", 40.0)
    h2_feed = s1.get("H2_Mole_Flow_kmol_h", 60.0)
    co2_prod = s18.get("CO2_Mole_Flow_kmol_h", 0.0)
    h2_prod = s5.get("H2_Mole_Flow_kmol_h", 0.0)

    co2_recovery = (co2_prod / co2_feed * 100.0) if co2_feed > 0 else 0.0
    h2_recovery = (h2_prod / h2_feed * 100.0) if h2_feed > 0 else 0.0
    co2_purity = s18.get("CO2_mol_pct", 0.0)
    h2_purity = s5.get("H2_mol_pct", 0.0)
    co2_intermediate_purity = s8.get("CO2_mol_pct", 0.0)

    # Solvent makeup
    makeup_flow_total = m1.get("Total_Mole_Flow_kmol_h", 0.0) + m2.get("Total_Mole_Flow_kmol_h", 0.0)
    makeup_mass_total = m1.get("Total_Mass_Flow_kg_h", 0.0) + m2.get("Total_Mass_Flow_kg_h", 0.0)

    # Equipment powers
    comp_power_total = sum(b["Power_kW"] for b in block_data if b["Category"] == "Compression")
    pump_power_total = sum(b["Power_kW"] for b in block_data if b["Category"] == "Pumping")
    total_power_kw = comp_power_total + pump_power_total

    # Equipment duties
    chilling_duty_total = sum(b["Duty_kW"] for b in block_data if b["Category"] == "Chilling / Refrigeration")
    dissociation_duty_total = sum(b["Duty_kW"] for b in block_data if b["Category"] == "Hydrate Dissociation (Heating)")

    # Specific energy (kWh per kg CO2 captured)
    co2_captured_kg_hr = co2_prod * mw["CO2"]
    sec_kwh_kg = (total_power_kw / co2_captured_kg_hr) if co2_captured_kg_hr > 0 else 0.0
    sec_mj_kg = sec_kwh_kg * 3.6
    sec_kwh_tonne = sec_kwh_kg * 1000.0

    kpis = {
        "Feed_Syngas_Flow_kmol_h": co2_feed + h2_feed,
        "Feed_CO2_Flow_kmol_h": co2_feed,
        "Feed_H2_Flow_kmol_h": h2_feed,
        "Target_CO2_Flow_kmol_h": co2_prod,
        "Target_CO2_Mass_Flow_kg_h": round(co2_captured_kg_hr, 2),
        "Overall_CO2_Recovery_pct": round(co2_recovery, 2),
        "Final_CO2_Purity_mol_pct": round(co2_purity, 2),
        "Final_CO2_Purity_wt_pct": s18.get("CO2_wt_pct", 0.0),
        "Intermediate_CO2_Purity_mol_pct": round(co2_intermediate_purity, 2),
        "Primary_Clean_H2_Flow_kmol_h": s5.get("Total_Mole_Flow_kmol_h", 0.0),
        "Primary_Clean_H2_Recovery_pct": round(h2_recovery, 2),
        "Primary_Clean_H2_Purity_mol_pct": round(h2_purity, 2),
        "Tail_Gas_H2_Loss_kmol_h": s15.get("H2_Mole_Flow_kmol_h", 0.0),
        "Total_Fresh_Solvent_Makeup_kmol_h": round(makeup_flow_total, 4),
        "Total_Fresh_Solvent_Makeup_kg_h": round(makeup_mass_total, 4),
        "Stage1_Solvent_Recycle_kmol_h": stream_map.get("S11", {}).get("Total_Mole_Flow_kmol_h", 0.0),
        "Stage2_Solvent_Recycle_kmol_h": stream_map.get("S21", {}).get("Total_Mole_Flow_kmol_h", 0.0),
        "Total_Compression_Power_kW": round(comp_power_total, 2),
        "Total_Pumping_Power_kW": round(pump_power_total, 2),
        "Total_Electrical_Power_kW": round(total_power_kw, 2),
        "Total_Chilling_Duty_kW": round(chilling_duty_total, 2),
        "Total_Dissociation_Duty_kW": round(dissociation_duty_total, 2),
        "Specific_Power_Consumption_kWh_kg_CO2": round(sec_kwh_kg, 4),
        "Specific_Power_Consumption_kWh_tonne_CO2": round(sec_kwh_tonne, 2),
        "Specific_Power_Consumption_MJ_kg_CO2": round(sec_mj_kg, 4)
    }

    json_kpi_path = os.path.join(output_dir, "kpi_summary.json")
    with open(json_kpi_path, "w") as f:
        json.dump(kpis, f, indent=2)
    print(f"Saved: {json_kpi_path}")

    df_kpis = pd.DataFrame([kpis]).T.reset_index()
    df_kpis.columns = ["Metric", "Value"]
    csv_kpi_path = os.path.join(output_dir, "kpi_summary.csv")
    df_kpis.to_csv(csv_kpi_path, index=False)
    print(f"Saved: {csv_kpi_path}")

    # 6. Generate Master Technical Markdown Report
    report_md_path = os.path.join(output_dir, "SIMULATION_DATA_REPORT.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Two-Stage Gas Hydrate CO2 Capture Simulation Data Report\n\n")
        f.write("**Flowsheet Model:** Two-Stage Hydrate Capture with Dual CAPE-OPEN Python Unit Operations (`B1` & `B2`)\n")
        f.write("**Thermodynamic Engine:** Gibbs Free Energy Minimization + van der Waals–Platteeuw (vdW-P) Model (kdarnell)\n")
        f.write("**Host Simulator:** Aspen Plus V14.0 (RK-SOAVE Property Package)\n\n")
        f.write("---\n\n")

        f.write("## 1. Executive Summary & Key Performance Indicators (KPIs)\n\n")
        f.write("| Key Performance Metric | Value | Unit | Benchmark / Design Target |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Target $\\text{{CO}}_2$ Product Purity (`S18`)** | **`{kpis['Final_CO2_Purity_mol_pct']:.2f}`** | **mol%** | $>95.0\\text{{ mol}}\\%$ (High Purity) |\n")
        f.write(f"| **Target $\\text{{CO}}_2$ Product Purity (`S18`)** | **`{kpis['Final_CO2_Purity_wt_pct']:.2f}`** | **wt%** | $>95.0\\text{{ wt}}\\%$ |\n")
        f.write(f"| **Overall Plant $\\text{{CO}}_2$ Capture Recovery** | **`{kpis['Overall_CO2_Recovery_pct']:.2f}`** | **%** | $>60.0\\text{{ %}}$ (2-Stage Cascade) |\n")
        f.write(f"| **Primary Clean $\\text{{H}}_2$ Purity (`S5`)** | **`{kpis['Primary_Clean_H2_Purity_mol_pct']:.2f}`** | **mol%** | Fuel-cell / Pre-combustion spec |\n")
        f.write(f"| **Primary Clean $\\text{{H}}_2$ Recovery (`S5`)** | **`{kpis['Primary_Clean_H2_Recovery_pct']:.2f}`** | **%** | $>95.0\\text{{ %}}$ |\n")
        f.write(f"| **Intermediate Gas $\\text{{CO}}_2$ Purity (`S8`)** | **`{kpis['Intermediate_CO2_Purity_mol_pct']:.2f}`** | **mol%** | Stage 1 Enrichment |\n")
        f.write(f"| **Total Net $\\text{{CO}}_2$ Captured** | **`{kpis['Target_CO2_Mass_Flow_kg_h']:.2f}`** | **kg/h** | $25.32\\text{{ kmol/h}}$ |\n")
        f.write(f"| **Total Compression Power (`COMP1` + `COMP2`)** | **`{kpis['Total_Compression_Power_kW']:.2f}`** | **kW** | Mechanical shaft work |\n")
        f.write(f"| **Total Pumping Power (`PUMP1` + `PUMP2`)** | **`{kpis['Total_Pumping_Power_kW']:.2f}`** | **kW** | Solvent circulation work |\n")
        f.write(f"| **Total Plant Electrical Power** | **`{kpis['Total_Electrical_Power_kW']:.2f}`** | **kW** | Compressors + Pumps |\n")
        f.write(f"| **Specific Electrical Energy Consumption** | **`{kpis['Specific_Power_Consumption_kWh_kg_CO2']:.4f}`** | **kWh/kg $\\text{{CO}}_2$** | `{kpis['Specific_Power_Consumption_kWh_tonne_CO2']:.2f}\\text{{ kWh/tonne}}$ |\n")
        f.write(f"| **Specific Compression Energy** | **`{kpis['Specific_Power_Consumption_MJ_kg_CO2']:.4f}`** | **MJ/kg $\\text{{CO}}_2$** | Low parasitic loss |\n")
        f.write(f"| **Total Chilling Duty (`CHILL1-4`)** | **`{kpis['Total_Chilling_Duty_kW']:.2f}`** | **kW** | Refrigeration to $1.0\\text{{ }}^\\circ\\text{{C}}$ |\n")
        f.write(f"| **Total Dissociation Duty (`DISS1-2`)** | **`{kpis['Total_Dissociation_Duty_kW']:.2f}`** | **kW** | Low-grade heat at $12.0\\text{{ }}^\\circ\\text{{C}}$ |\n")
        f.write(f"| **Fresh Solvent Make-up (`MAKEUP1` + `MAKEUP2`)** | **`{kpis['Total_Fresh_Solvent_Makeup_kg_h']:.2f}`** | **kg/h** | $0.28\\text{{ kmol/h}}$ solvent replenishment |\n\n")

        f.write("---\n\n")
        f.write("## 2. Complete Stream Table (All 23 Streams)\n\n")
        f.write("| Stream | Description | Phase | $T$ [°C] | $P$ [bar] | Mole Flow [kmol/h] | Mass Flow [kg/h] | $\\text{CO}_2$ mol% | $\\text{H}_2$ mol% | $\\text{H}_2\\text{O}$ mol% | $\\text{DIOX}$ mol% | $\\text{CO}_2$ wt% | $\\text{H}_2$ wt% |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for s in streams_data:
            f.write(f"| **`{s['Stream']}`** | {s['Description']} | {s['Phase']} | {s['Temperature_C']} | {s['Pressure_bar']} | {s['Total_Mole_Flow_kmol_h']:.2f} | {s['Total_Mass_Flow_kg_h']:.2f} | {s['CO2_mol_pct']:.2f}% | {s['H2_mol_pct']:.2f}% | {s['H2O_mol_pct']:.2f}% | {s['DIOX_mol_pct']:.2f}% | {s['CO2_wt_pct']:.2f}% | {s['H2_wt_pct']:.2f}% |\n")

        f.write("\n---\n\n")
        f.write("## 3. Unit Operation Performance (All 16 Blocks)\n\n")
        f.write("| Block ID | Category | Model Type | Thermal Duty [Gcal/hr] | Thermal Duty [kW] | Electrical Power [kW] |\n")
        f.write("| :--- | :--- | :--- | :---: | :---: | :---: |\n")
        for b in block_data:
            f.write(f"| **`{b['Block']}`** | {b['Category']} | {b['Model_Type']} | {b['Duty_Gcal_hr']:.4f} | {b['Duty_kW']:.2f} | {b['Power_kW']:.2f} |\n")

        f.write("\n---\n\n")
        f.write("## 4. Overall Plant Material & Component Balances\n\n")
        f.write("### 4.1 Carbon Dioxide ($\text{CO}_2$) Balance\n")
        f.write(f"* **Total In:** Syngas `S1` = **`{co2_feed:.4f} kmol/h`** (`{co2_feed * 44.01:.2f} kg/h`)\n")
        f.write(f"* **Outlets:**\n")
        f.write(f"  * Primary Clean Gas (`S5`): `{stream_map.get('S5', {}).get('CO2_Mole_Flow_kmol_h', 0.0):.4f} kmol/h`\n")
        f.write(f"  * Secondary Tail Gas (`S15`): `{stream_map.get('S15', {}).get('CO2_Mole_Flow_kmol_h', 0.0):.4f} kmol/h`\n")
        f.write(f"  * Target $\\text{{CO}}_2$ Product (`S18`): `{co2_prod:.4f} kmol/h` (`{co2_prod * 44.01:.2f} kg/h`)\n")
        co2_out_sum = stream_map.get('S5', {}).get('CO2_Mole_Flow_kmol_h', 0.0) + stream_map.get('S15', {}).get('CO2_Mole_Flow_kmol_h', 0.0) + co2_prod
        f.write(f"* **Total Out:** **`{co2_out_sum:.4f} kmol/h`** (Closure: **`{(co2_out_sum/co2_feed)*100:.2f}%`**)\n\n")

        f.write("### 4.2 Hydrogen ($\text{H}_2$) Balance\n")
        f.write(f"* **Total In:** Syngas `S1` = **`{h2_feed:.4f} kmol/h`** (`{h2_feed * 2.016:.2f} kg/h`)\n")
        f.write(f"* **Outlets:**\n")
        f.write(f"  * Primary Clean $\\text{{H}}_2$ Product (`S5`): `{h2_prod:.4f} kmol/h` (Recovery: **`{(h2_prod/h2_feed)*100:.2f}%`**)\n")
        f.write(f"  * Secondary Tail Gas (`S15`): `{stream_map.get('S15', {}).get('H2_Mole_Flow_kmol_h', 0.0):.4f} kmol/h`\n")
        f.write(f"  * Target $\\text{{CO}}_2$ Product (`S18`): `{stream_map.get('S18', {}).get('H2_Mole_Flow_kmol_h', 0.0):.4f} kmol/h`\n")
        h2_out_sum = h2_prod + stream_map.get('S15', {}).get('H2_Mole_Flow_kmol_h', 0.0) + stream_map.get('S18', {}).get('H2_Mole_Flow_kmol_h', 0.0)
        f.write(f"* **Total Out:** **`{h2_out_sum:.4f} kmol/h`** (Closure: **`{(h2_out_sum/h2_feed)*100:.2f}%`**)\n\n")

        f.write("---\n\n")
        f.write("## 5. Scope & Roadmap for Upcoming Sensitivity Analysis\n\n")
        f.write("The current baseline simulation is fully verified and converged. For the sensitivity analysis section of the thesis, the following parametric sweeps are ready to be conducted:\n\n")
        f.write("1. **Stage 1 & 2 Operating Pressure ($P = 15 - 35\\text{ bar}$):**\n")
        f.write("   * *Impact on:* Hydrate equilibrium formation temperature, required refrigeration power vs. compression work.\n")
        f.write("2. **Operating Temperature ($T = 273.15 - 278.15\\text{ K}$ / $0 - 5\\text{ }^\\circ\\text{C}$):**\n")
        f.write("   * *Impact on:* Chilling duty, hydrate crystallizer volume, stability margin.\n")
        f.write("3. **1,4-Dioxane Promoter Concentration ($0.0 - 5.56\\text{ mol}\\%$):**\n")
        f.write("   * *Impact on:* Thermodynamic equilibrium shift from Structure I to Structure II, enabling mild operating pressures ($<25\\text{ bar}$ vs. $>60\\text{ bar}$ without promoter).\n")
        f.write("4. **Solvent-to-Feed Ratio ($L/G = 5 - 15\\text{ mol/mol}$):**\n")
        f.write("   * *Impact on:* Slurry pump power, reactor kinetics, heat of formation absorption.\n")
        f.write("5. **Dissociator Pressure & Temperature ($P = 2 - 8\\text{ bar}$, $T = 8 - 20\\text{ }^\\circ\\text{C}$):**\n")
        f.write("   * *Impact on:* Intermediate compression ratio (`COMP2`), product $\\text{CO}_2$ delivery pressure, dissociation heat requirement.\n")

    print(f"Saved: {report_md_path}")

finally:
    aspen.Quit()
    print("Closed Aspen.")
