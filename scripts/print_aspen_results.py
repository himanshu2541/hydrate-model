"""
print_aspen_results.py
-----------------------
Extracts converged stream data and performance metrics from Two_Stage_Hydrate_Capture_MVP.bkp.
"""

import os
import win32com.client

def main():
    sim_dir = os.path.abspath("aspen_simulation")
    bkp_path = os.path.join(sim_dir, "Two_Stage_Hydrate_Capture_MVP.bkp")
    if not os.path.exists(bkp_path):
        print("Backup file not found:", bkp_path)
        return

    print("Opening Aspen Plus converged simulation...")
    aspen = win32com.client.Dispatch("Apwn.Document")
    aspen.SuppressDialogs = 1
    try:
        aspen.InitFromArchive2(bkp_path)
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
        
        def read_stream(sname):
            flow_node = tree.FindNode(rf"\Data\Streams\{sname}\Output\STR_MAIN\MOLEFLOW\MIXED")
            if flow_node:
                try:
                    t = float(tree.FindNode(rf"\Data\Streams\{sname}\Output\STR_MAIN\TEMP\MIXED").Value or 0.0)
                    p = float(tree.FindNode(rf"\Data\Streams\{sname}\Output\STR_MAIN\PRES\MIXED").Value or 0.0)
                    co2 = float(flow_node.Elements("CO2").Value or 0.0)
                    h2 = float(flow_node.Elements("H2").Value or 0.0)
                    h2o = float(flow_node.Elements("H2O").Value or 0.0)
                    diox = float(flow_node.Elements("DIOXANE").Value or 0.0)
                    tot = co2 + h2 + h2o + diox
                    return t, p, tot, co2, h2, h2o, diox
                except Exception as e:
                    pass
                
            try:
                t = float(tree.FindNode(rf"\Data\Streams\{sname}\Input\TEMP\MIXED").Value or 0.0)
                p = float(tree.FindNode(rf"\Data\Streams\{sname}\Input\PRES\MIXED").Value or 0.0)
                tot = float(tree.FindNode(rf"\Data\Streams\{sname}\Input\TOTFLOW\MIXED").Value or 0.0)
                co2 = float(tree.FindNode(rf"\Data\Streams\{sname}\Input\FLOW\MIXED\CO2").Value or 0.0)
                h2 = float(tree.FindNode(rf"\Data\Streams\{sname}\Input\FLOW\MIXED\H2").Value or 0.0)
                h2o = float(tree.FindNode(rf"\Data\Streams\{sname}\Input\FLOW\MIXED\H2O").Value or 0.0)
                diox = float(tree.FindNode(rf"\Data\Streams\{sname}\Input\FLOW\MIXED\DIOXANE").Value or 0.0)
                return t, p, tot, co2, h2, h2o, diox
            except Exception as e:
                return None

        print("\n" + "="*95)
        print(f"{'Stream':<6} {'Description':<32} {'T [C]':<8} {'P [bar]':<8} {'Flow [kmol/h]':<15} {'CO2 mol%':<10} {'H2 mol%':<10}")
        print("-"*95)
        
        stream_results = {}
        for sname, desc in key_streams:
            res = read_stream(sname)
            if res:
                t, p, tot, co2, h2, h2o, diox = res
                stream_results[sname] = res
                co2_pct = (co2 / tot * 100) if tot > 0 else 0
                h2_pct = (h2 / tot * 100) if tot > 0 else 0
                print(f"{sname:<6} {desc:<32} {t:<8.1f} {p:<8.1f} {tot:<15.2f} {co2_pct:<10.2f}% {h2_pct:<10.2f}%")
        print("="*95 + "\n")
        
        # Calculate Key Performance Indicators (KPIs)
        s1 = stream_results.get("S1")
        s5 = stream_results.get("S5")
        s8 = stream_results.get("S8")
        s18 = stream_results.get("S18")
        
        if s1 and s5 and s18:
            n_co2_feed = s1[3]
            n_co2_target = s18[3]
            purity_co2 = (s18[3] / s18[2]) * 100.0
            r_co2 = (n_co2_target / n_co2_feed) * 100.0
            
            n_h2_feed = s1[4]
            n_h2_clean1 = s5[4]
            r_h2 = (n_h2_clean1 / n_h2_feed) * 100.0
            purity_h2 = (s5[4] / s5[2]) * 100.0
            
            print("="*65)
            print("         KEY THESIS DEFENSE DELIVERABLES & BENCHMARKS")
            print("="*65)
            print(f"Target CO2 Product Purity (Stream S18) : {purity_co2:6.2f} mol%   (Req: >= 98.0%)")
            print(f"Overall CO2 Recovery across 2 Stages   : {r_co2:6.2f} %      (Req: >= 90.0%)")
            print(f"Primary Stage 1 H2 Recovery (Stream S5): {r_h2:6.2f} %      (Req: >= 95.0%)")
            print(f"Clean Hydrogen Gas Purity (Stream S5)  : {purity_h2:6.2f} mol%   (Enriched Syngas)")
            print("="*65)
            
    finally:
        aspen.Quit()

if __name__ == "__main__":
    main()
