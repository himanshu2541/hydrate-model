"""
generate_thesis_tables.py
--------------------------
Extracts stream and performance data from both Model 1 (Native MVP) and Model 2 (CAPE-OPEN)
and formats publication-quality comparison tables for the Master's Thesis defense.
"""

import os
import sys
import win32com.client
import pandas as pd

def extract_model_data(sim_path, model_type="archive"):
    aspen = win32com.client.Dispatch("Apwn.Document")
    aspen.SuppressDialogs = 1
    abs_path = os.path.abspath(sim_path)
    
    try:
        if model_type == "archive":
            aspen.InitFromArchive2(abs_path)
        else:
            aspen.InitFromFile2(abs_path)
            
        tree = aspen.Tree
        
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
        
        streams_dict = {}
        for sname, desc in key_streams:
            flow_node = tree.FindNode(rf"\Data\Streams\{sname}\Output\STR_MAIN\MOLEFLOW\MIXED")
            t, p, tot, co2, h2, h2o, diox = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            if flow_node:
                try:
                    t = float(tree.FindNode(rf"\Data\Streams\{sname}\Output\STR_MAIN\TEMP\MIXED").Value or 0.0)
                    p = float(tree.FindNode(rf"\Data\Streams\{sname}\Output\STR_MAIN\PRES\MIXED").Value or 0.0)
                    co2 = float(flow_node.Elements("CO2").Value or 0.0)
                    h2 = float(flow_node.Elements("H2").Value or 0.0)
                    h2o = float(flow_node.Elements("H2O").Value or 0.0)
                    diox = float(flow_node.Elements("DIOXANE").Value or 0.0)
                    tot = co2 + h2 + h2o + diox
                except Exception:
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
                except Exception:
                    pass
                    
            c_pct = (co2 / tot * 100.0) if tot > 0 else 0.0
            h_pct = (h2 / tot * 100.0) if tot > 0 else 0.0
            w_pct = (h2o / tot * 100.0) if tot > 0 else 0.0
            d_pct = (diox / tot * 100.0) if tot > 0 else 0.0
            
            streams_dict[sname] = {
                "desc": desc, "T": t, "P": p, "tot": tot,
                "co2": co2, "h2": h2, "h2o": h2o, "diox": diox,
                "co2_pct": c_pct, "h2_pct": h_pct, "h2o_pct": w_pct, "diox_pct": d_pct
            }
            
        # Equipment work/duty
        def get_work(bname):
            for path in [rf"\Data\Blocks\{bname}\Output\POWER", rf"\Data\Blocks\{bname}\Output\BRAKE_PWR", rf"\Data\Blocks\{bname}\Output\ELEC_PWR"]:
                n = tree.FindNode(path)
                if n and n.Value is not None:
                    return float(n.Value)
            return 0.0
            
        def get_duty(bname):
            for path in [rf"\Data\Blocks\{bname}\Output\QCALC", rf"\Data\Blocks\{bname}\Output\DUTY"]:
                n = tree.FindNode(path)
                if n and n.Value is not None:
                    return float(n.Value)
            return 0.0
            
        equipment = {
            "COMP1_kW": get_work("COMP1"),
            "COMP2_kW": get_work("COMP2"),
            "PUMP1_kW": get_work("PUMP1"),
            "PUMP2_kW": get_work("PUMP2"),
            "CHILL1_Gcal": get_duty("CHILL1"),
            "CHILL2_Gcal": get_duty("CHILL2"),
            "CHILL3_Gcal": get_duty("CHILL3"),
            "CHILL4_Gcal": get_duty("CHILL4"),
        }
        
        return streams_dict, equipment
        
    finally:
        aspen.Quit()

if __name__ == "__main__":
    print("Helper module ready.")
