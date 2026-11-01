"""
compare_models.py
------------------
Compares streams and block performance metrics between Model 1 (Native) and Model 2 (CAPE-OPEN).
"""

import os
import sys
import win32com.client

def read_streams_from(sim_path, name):
    print(f"\n=======================================================")
    print(f"   {name}")
    print(f"   Path: {sim_path}")
    print(f"=======================================================")
    
    aspen = win32com.client.Dispatch("Apwn.Document")
    aspen.SuppressDialogs = 1
    stream_results = {}
    
    try:
        abs_path = os.path.abspath(sim_path)
        if abs_path.endswith(".bkp"):
            aspen.InitFromArchive2(abs_path)
        else:
            aspen.InitFromFile2(abs_path)
            
        tree = aspen.Tree
        
        # Check run status
        stat_node = tree.FindNode(r"\Data\Results Summary\Run-Status\Output\RUN_STATUS")
        if stat_node:
            print(f"Run Status: {stat_node.Value}")
            
        key_streams = [
            ("S1", "Fresh Syngas Feed"),
            ("S2", "Feed Compressor Discharge"),
            ("S3", "Chilled Feed Gas (1.0 C)"),
            ("S4", "Stage 1 Combined Reactor Feed"),
            ("S5", "CLEAN-1 (Primary H2 Product)"),
            ("S6", "Stage 1 Hydrate Slurry"),
            ("S7", "Depressurized Slurry (4 bar)"),
            ("S8", "GAS-INT (Intermediate CO2)"),
            ("S9", "Stage 1 Regenerated Solvent"),
            ("S10", "Pump 1 Discharge"),
            ("S11", "Stage 1 Recycled Solvent"),
            ("S12", "Stage 2 Compressed Gas"),
            ("S13", "Stage 2 Chilled Gas"),
            ("S14", "Stage 2 Combined Reactor Feed"),
            ("S15", "CLEAN-2 (Secondary Tail Gas)"),
            ("S16", "Stage 2 Hydrate Slurry"),
            ("S17", "Depressurized Slurry (4 bar)"),
            ("S18", "TARGET-CO2 (Final Product)"),
            ("S19", "Stage 2 Regenerated Solvent"),
            ("S20", "Pump 2 Discharge"),
            ("S21", "Stage 2 Recycled Solvent"),
        ]
        
        hdr = f"{'Stream':<6} {'Description':<32} {'T [C]':<8} {'P [bar]':<8} {'Flow [kmol/h]':<15} {'CO2 %':<10} {'H2 %':<10} {'H2O %':<10} {'DIOX %':<10}"
        print(hdr)
        print("-" * len(hdr))
        
        for sname, desc in key_streams:
            flow_node = tree.FindNode(rf"\Data\Streams\{sname}\Output\STR_MAIN\MOLEFLOW\MIXED")
            t, p, tot, co2, h2, h2o, diox = 0, 0, 0, 0, 0, 0, 0
            if flow_node:
                try:
                    t = float(tree.FindNode(rf"\Data\Streams\{sname}\Output\STR_MAIN\TEMP\MIXED").Value or 0.0)
                    p = float(tree.FindNode(rf"\Data\Streams\{sname}\Output\STR_MAIN\PRES\MIXED").Value or 0.0)
                    co2 = float(flow_node.Elements("CO2").Value or 0.0)
                    h2 = float(flow_node.Elements("H2").Value or 0.0)
                    h2o = float(flow_node.Elements("H2O").Value or 0.0)
                    diox = float(flow_node.Elements("DIOXANE").Value or 0.0)
                    tot = co2 + h2 + h2o + diox
                except Exception as e:
                    pass
            else:
                try:
                    t = float(tree.FindNode(rf"\Data\Streams\{sname}\Input\TEMP\MIXED").Value or 0.0)
                    p = float(tree.FindNode(rf"\Data\Streams\{sname}\Input\PRES\MIXED").Value or 0.0)
                    tot = float(tree.FindNode(rf"\Data\Streams\{sname}\Input\TOTFLOW\MIXED").Value or 0.0)
                    co2 = float(tree.FindNode(rf"\Data\Streams\{sname}\Input\FLOW\MIXED\CO2").Value or 0.0)
                    h2 = float(tree.FindNode(rf"\Data\Streams\{sname}\Input\FLOW\MIXED\H2").Value or 0.0)
                    h2o = float(tree.FindNode(rf"\Data\Streams\{sname}\Input\FLOW\MIXED\H2O").Value or 0.0)
                    diox = float(tree.FindNode(rf"\Data\Streams\{sname}\Input\FLOW\MIXED\DIOXANE").Value or 0.0)
                except Exception as e:
                    pass
                    
            c_pct = (co2 / tot * 100) if tot > 0 else 0
            h_pct = (h2 / tot * 100) if tot > 0 else 0
            w_pct = (h2o / tot * 100) if tot > 0 else 0
            d_pct = (diox / tot * 100) if tot > 0 else 0
            
            stream_results[sname] = {
                "desc": desc, "T": t, "P": p, "tot": tot,
                "co2": co2, "h2": h2, "h2o": h2o, "diox": diox,
                "co2_pct": c_pct, "h2_pct": h_pct, "h2o_pct": w_pct, "diox_pct": d_pct
            }
            print(f"{sname:<6} {desc:<32} {t:<8.1f} {p:<8.1f} {tot:<15.2f} {c_pct:<10.2f}% {h_pct:<10.2f}% {w_pct:<10.2f}% {d_pct:<10.2f}%")
            
        # Key metrics
        s1 = stream_results.get("S1")
        s5 = stream_results.get("S5")
        s8 = stream_results.get("S8")
        s18 = stream_results.get("S18")
        if s1 and s5 and s18:
            purity_co2 = s18["co2_pct"]
            r_co2 = (s18["co2"] / s1["co2"]) * 100 if s1["co2"] > 0 else 0
            purity_h2 = s5["h2_pct"]
            r_h2 = (s5["h2"] / s1["h2"]) * 100 if s1["h2"] > 0 else 0
            print("-" * 65)
            print(f"Target CO2 Purity (S18) : {purity_co2:.2f} mol%")
            print(f"Overall CO2 Recovery    : {r_co2:.2f} %")
            print(f"Stage 1 H2 Recovery (S5): {r_h2:.2f} %")
            print(f"Clean H2 Purity (S5)    : {purity_h2:.2f} mol%")
            print(f"Intermediate CO2 (S8)   : {s8['co2_pct']:.2f} mol%")
            print("-" * 65)
            
    finally:
        aspen.Quit()
        
    return stream_results

if __name__ == "__main__":
    m1 = read_streams_from("aspen_simulation/Two_Stage_Hydrate_Capture_Final.bkp", "Model 1 (Native MVP)")
    m2 = read_streams_from("aspen_simulation/cape_open_simulation/Two_Stage_Hydrate_Capture_CAPE_OPEN.bkp", "Model 2 (CAPE-OPEN)")
