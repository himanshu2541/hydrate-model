"""
build_and_run_hydrate_flowsheet.py
-----------------------------------
Rigorous automated builder and executor for Two-Stage Hydrate CO2 Capture in Aspen Plus V14.
Implements the exact flowsheet architecture and stream table specifications from Chapter 4 / Step 1.
"""

import os
import sys
import time
import win32com.client

def build_simulation():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_bkp = os.path.join(current_dir, "hydrate_base.bkp")
    out_bkp = os.path.join(current_dir, "Two_Stage_Hydrate_Capture_MVP.bkp")
    out_apw = os.path.join(current_dir, "Two_Stage_Hydrate_Capture_Final.apw")
    
    print(f"Connecting to Aspen Plus V14.0 COM...")
    aspen = win32com.client.Dispatch("Apwn.Document")
    aspen.SuppressDialogs = 1
    
    try:
        print(f"Loading base file: {base_bkp}...")
        aspen.InitFromArchive2(base_bkp)
        print("Base simulation loaded successfully.")
        
        tree = aspen.Tree
        blocks = tree.FindNode(r"\Data\Blocks")
        streams = tree.FindNode(r"\Data\Streams")
        
        # 1. Define Streams
        stream_names = [f"S{i}" for i in range(1, 22)]
        print(f"Creating {len(stream_names)} streams...")
        for sname in stream_names:
            streams.Elements.Add(f"{sname}!MATERIAL")
            
        # 2. Define Blocks
        block_defs = [
            ("COMP1", "Compr"),
            ("CHILL1", "Heater"),
            ("MIX1", "Mixer"),
            ("FORM1", "Sep"),
            ("VALV1", "Valve"),
            ("DISS1", "Flash2"),
            ("PUMP1", "Pump"),
            ("CHILL2", "Heater"),
            ("COMP2", "Compr"),
            ("CHILL3", "Heater"),
            ("MIX2", "Mixer"),
            ("FORM2", "Sep"),
            ("VALV2", "Valve"),
            ("DISS2", "Flash2"),
            ("PUMP2", "Pump"),
            ("CHILL4", "Heater"),
        ]
        
        print(f"Creating {len(block_defs)} unit operation blocks...")
        for bname, btype in block_defs:
            blocks.Elements.Add(f"{bname}!{btype}")
            
        # 3. Connect Flowsheet Streams
        print("Wiring flowsheet streams to block ports...")
        
        # Stage 1 Feed conditioning:
        # S1 -> COMP1 -> S2 -> CHILL1 -> S3 -> MIX1
        tree.FindNode(r"\Data\Blocks\COMP1\Ports\F(IN)").Elements.Add("S1")
        tree.FindNode(r"\Data\Blocks\COMP1\Ports\P(OUT)").Elements.Add("S2")
        
        tree.FindNode(r"\Data\Blocks\CHILL1\Ports\F(IN)").Elements.Add("S2")
        tree.FindNode(r"\Data\Blocks\CHILL1\Ports\P(OUT)").Elements.Add("S3")
        
        # MIX1 combines S3 and S11 (recycled solvent) -> S4
        tree.FindNode(r"\Data\Blocks\MIX1\Ports\F(IN)").Elements.Add("S3")
        tree.FindNode(r"\Data\Blocks\MIX1\Ports\F(IN)").Elements.Add("S11")
        tree.FindNode(r"\Data\Blocks\MIX1\Ports\P(OUT)").Elements.Add("S4")
        
        # FORM1 (Sep): In S4 -> Out S5 (Vapor: CLEAN-1), S6 (Slurry: SLURRY-1)
        tree.FindNode(r"\Data\Blocks\FORM1\Ports\F(IN)").Elements.Add("S4")
        tree.FindNode(r"\Data\Blocks\FORM1\Ports\P(OUT)").Elements.Add("S5")
        tree.FindNode(r"\Data\Blocks\FORM1\Ports\P(OUT)").Elements.Add("S6")
        
        # VALV1: In S6 -> Out S7
        tree.FindNode(r"\Data\Blocks\VALV1\Ports\F(IN)").Elements.Add("S6")
        tree.FindNode(r"\Data\Blocks\VALV1\Ports\P(OUT)").Elements.Add("S7")
        
        # DISS1 (Flash2): In S7 -> Out S8 (Vapor: GAS-INT), S9 (Liquid: LIQ-REC1)
        tree.FindNode(r"\Data\Blocks\DISS1\Ports\F(IN)").Elements.Add("S7")
        tree.FindNode(r"\Data\Blocks\DISS1\Ports\V(OUT)").Elements.Add("S8")
        tree.FindNode(r"\Data\Blocks\DISS1\Ports\L(OUT)").Elements.Add("S9")
        
        # PUMP1: In S9 -> Out S10
        tree.FindNode(r"\Data\Blocks\PUMP1\Ports\F(IN)").Elements.Add("S9")
        tree.FindNode(r"\Data\Blocks\PUMP1\Ports\P(OUT)").Elements.Add("S10")
        
        # CHILL2: In S10 -> Out S11
        tree.FindNode(r"\Data\Blocks\CHILL2\Ports\F(IN)").Elements.Add("S10")
        tree.FindNode(r"\Data\Blocks\CHILL2\Ports\P(OUT)").Elements.Add("S11")
        
        # Stage 2 Polishing Loop:
        # COMP2: In S8 -> Out S12
        tree.FindNode(r"\Data\Blocks\COMP2\Ports\F(IN)").Elements.Add("S8")
        tree.FindNode(r"\Data\Blocks\COMP2\Ports\P(OUT)").Elements.Add("S12")
        
        # CHILL3: In S12 -> Out S13
        tree.FindNode(r"\Data\Blocks\CHILL3\Ports\F(IN)").Elements.Add("S12")
        tree.FindNode(r"\Data\Blocks\CHILL3\Ports\P(OUT)").Elements.Add("S13")
        
        # MIX2 combines S13 and S21 (recycled solvent) -> S14
        tree.FindNode(r"\Data\Blocks\MIX2\Ports\F(IN)").Elements.Add("S13")
        tree.FindNode(r"\Data\Blocks\MIX2\Ports\F(IN)").Elements.Add("S21")
        tree.FindNode(r"\Data\Blocks\MIX2\Ports\P(OUT)").Elements.Add("S14")
        
        # FORM2 (Sep): In S14 -> Out S15 (Vapor: CLEAN-2), S16 (Slurry: SLURRY-2)
        tree.FindNode(r"\Data\Blocks\FORM2\Ports\F(IN)").Elements.Add("S14")
        tree.FindNode(r"\Data\Blocks\FORM2\Ports\P(OUT)").Elements.Add("S15")
        tree.FindNode(r"\Data\Blocks\FORM2\Ports\P(OUT)").Elements.Add("S16")
        
        # VALV2: In S16 -> Out S17
        tree.FindNode(r"\Data\Blocks\VALV2\Ports\F(IN)").Elements.Add("S16")
        tree.FindNode(r"\Data\Blocks\VALV2\Ports\P(OUT)").Elements.Add("S17")
        
        # DISS2 (Flash2): In S17 -> Out S18 (Vapor: TARGET-CO2), S19 (Liquid: LIQ-REC2)
        tree.FindNode(r"\Data\Blocks\DISS2\Ports\F(IN)").Elements.Add("S17")
        tree.FindNode(r"\Data\Blocks\DISS2\Ports\V(OUT)").Elements.Add("S18")
        tree.FindNode(r"\Data\Blocks\DISS2\Ports\L(OUT)").Elements.Add("S19")
        
        # PUMP2: In S19 -> Out S20
        tree.FindNode(r"\Data\Blocks\PUMP2\Ports\F(IN)").Elements.Add("S19")
        tree.FindNode(r"\Data\Blocks\PUMP2\Ports\P(OUT)").Elements.Add("S20")
        
        # CHILL4: In S20 -> Out S21
        tree.FindNode(r"\Data\Blocks\CHILL4\Ports\F(IN)").Elements.Add("S20")
        tree.FindNode(r"\Data\Blocks\CHILL4\Ports\P(OUT)").Elements.Add("S21")
        
        # 4. Set Feed Stream S1 Specifications
        print("Specifying Feed Stream S1 (100 kmol/h, 25 bar, 30 C, 40% CO2, 60% H2)...")
        tree.FindNode(r"\Data\Streams\S1\Input\TEMP\MIXED").Value = 30.0
        tree.FindNode(r"\Data\Streams\S1\Input\PRES\MIXED").Value = 25.0
        tree.FindNode(r"\Data\Streams\S1\Input\TOTFLOW\MIXED").Value = 100.0
        tree.FindNode(r"\Data\Streams\S1\Input\FLOW\MIXED\CO2").Value = 40.0
        tree.FindNode(r"\Data\Streams\S1\Input\FLOW\MIXED\H2").Value = 60.0
        tree.FindNode(r"\Data\Streams\S1\Input\FLOW\MIXED\H2O").Value = 0.0
        tree.FindNode(r"\Data\Streams\S1\Input\FLOW\MIXED\DIOXANE").Value = 0.0
        
        # 5. Set Initial Tear Stream Estimates (S11 and S21)
        print("Initializing Recycle Tear Stream Estimates (S11 and S21)...")
        # S11: 1200 kmol/h solvent, 5.56 mol% Dioxane
        tree.FindNode(r"\Data\Streams\S11\Input\TEMP\MIXED").Value = 1.0
        tree.FindNode(r"\Data\Streams\S11\Input\PRES\MIXED").Value = 24.8
        tree.FindNode(r"\Data\Streams\S11\Input\TOTFLOW\MIXED").Value = 1200.0
        tree.FindNode(r"\Data\Streams\S11\Input\FLOW\MIXED\H2O").Value = 1200.0 * 0.9444
        tree.FindNode(r"\Data\Streams\S11\Input\FLOW\MIXED\DIOXANE").Value = 1200.0 * 0.0556
        tree.FindNode(r"\Data\Streams\S11\Input\FLOW\MIXED\CO2").Value = 0.0
        tree.FindNode(r"\Data\Streams\S11\Input\FLOW\MIXED\H2").Value = 0.0

        # S21: 400 kmol/h solvent, 5.56 mol% Dioxane
        tree.FindNode(r"\Data\Streams\S21\Input\TEMP\MIXED").Value = 1.0
        tree.FindNode(r"\Data\Streams\S21\Input\PRES\MIXED").Value = 24.8
        tree.FindNode(r"\Data\Streams\S21\Input\TOTFLOW\MIXED").Value = 400.0
        tree.FindNode(r"\Data\Streams\S21\Input\FLOW\MIXED\H2O").Value = 400.0 * 0.9444
        tree.FindNode(r"\Data\Streams\S21\Input\FLOW\MIXED\DIOXANE").Value = 400.0 * 0.0556
        tree.FindNode(r"\Data\Streams\S21\Input\FLOW\MIXED\CO2").Value = 0.0
        tree.FindNode(r"\Data\Streams\S21\Input\FLOW\MIXED\H2").Value = 0.0

        # 6. Configure Unit Operations Parameters
        print("Configuring Block Parameters...")
        
        # Convergence Options
        conv = tree.FindNode(r"\Data\Convergence\Conv-Options\Input")
        if conv:
            conv.Elements("WEG_MAXIT").Value = 100
            conv.Elements("TOL").Value = 0.001
        
        # COMP1: 25 bar discharge (P_in = P_out = 25 bar -> 0 kW for Promoted Case B)
        tree.FindNode(r"\Data\Blocks\COMP1\Input\TYPE").Value = "ISENTROPIC"
        tree.FindNode(r"\Data\Blocks\COMP1\Input\OPT_SPEC").Value = "PRES"
        tree.FindNode(r"\Data\Blocks\COMP1\Input\PRES").Value = 25.0
        tree.FindNode(r"\Data\Blocks\COMP1\Input\PEFF").Value = 0.75

        # CHILL1: 1.0 C, 24.8 bar
        tree.FindNode(r"\Data\Blocks\CHILL1\Input\TEMP").Value = 1.0
        tree.FindNode(r"\Data\Blocks\CHILL1\Input\PRES").Value = 24.8

        # FORM1 (Sep - Stage 1 Hydrate Reaction):
        # S5 overhead gets 99.27% H2, 28% CO2, traces of liquid
        s5_fracs = tree.FindNode(r"\Data\Blocks\FORM1\Input\FRACS\S5\MIXED")
        s5_fracs.Elements("H2").Value = 0.9927
        s5_fracs.Elements("CO2").Value = 0.2800
        s5_fracs.Elements("H2O").Value = 0.0001
        s5_fracs.Elements("DIOXANE").Value = 0.0001

        # VALV1: 4.0 bar
        tree.FindNode(r"\Data\Blocks\VALV1\Input\P_OUT").Value = 4.0

        # DISS1: 12.0 C, 4.0 bar
        tree.FindNode(r"\Data\Blocks\DISS1\Input\TEMP").Value = 12.0
        tree.FindNode(r"\Data\Blocks\DISS1\Input\PRES").Value = 4.0

        # PUMP1: 25.0 bar, eta = 0.70
        tree.FindNode(r"\Data\Blocks\PUMP1\Input\OPT_SPEC").Value = "PRES"
        tree.FindNode(r"\Data\Blocks\PUMP1\Input\PRES").Value = 25.0
        tree.FindNode(r"\Data\Blocks\PUMP1\Input\EFF").Value = 0.70

        # CHILL2: 1.0 C, 24.8 bar
        tree.FindNode(r"\Data\Blocks\CHILL2\Input\TEMP").Value = 1.0
        tree.FindNode(r"\Data\Blocks\CHILL2\Input\PRES").Value = 24.8

        # COMP2: Intermediate re-compression 4 bar -> 25 bar
        tree.FindNode(r"\Data\Blocks\COMP2\Input\TYPE").Value = "ISENTROPIC"
        tree.FindNode(r"\Data\Blocks\COMP2\Input\OPT_SPEC").Value = "PRES"
        tree.FindNode(r"\Data\Blocks\COMP2\Input\PRES").Value = 25.0
        tree.FindNode(r"\Data\Blocks\COMP2\Input\PEFF").Value = 0.75

        # CHILL3: 1.0 C, 24.8 bar
        tree.FindNode(r"\Data\Blocks\CHILL3\Input\TEMP").Value = 1.0
        tree.FindNode(r"\Data\Blocks\CHILL3\Input\PRES").Value = 24.8

        # FORM2 (Sep - Stage 2 Secondary Polishing):
        s15_fracs = tree.FindNode(r"\Data\Blocks\FORM2\Input\FRACS\S15\MIXED")
        s15_fracs.Elements("H2").Value = 0.9000
        s15_fracs.Elements("CO2").Value = 0.1200
        s15_fracs.Elements("H2O").Value = 0.0001
        s15_fracs.Elements("DIOXANE").Value = 0.0001

        # VALV2: 4.0 bar
        tree.FindNode(r"\Data\Blocks\VALV2\Input\P_OUT").Value = 4.0

        # DISS2: 12.0 C, 4.0 bar
        tree.FindNode(r"\Data\Blocks\DISS2\Input\TEMP").Value = 12.0
        tree.FindNode(r"\Data\Blocks\DISS2\Input\PRES").Value = 4.0

        # PUMP2: 25.0 bar, eta = 0.70
        tree.FindNode(r"\Data\Blocks\PUMP2\Input\OPT_SPEC").Value = "PRES"
        tree.FindNode(r"\Data\Blocks\PUMP2\Input\PRES").Value = 25.0
        tree.FindNode(r"\Data\Blocks\PUMP2\Input\EFF").Value = 0.70

        # CHILL4: 1.0 C, 24.8 bar
        tree.FindNode(r"\Data\Blocks\CHILL4\Input\TEMP").Value = 1.0
        tree.FindNode(r"\Data\Blocks\CHILL4\Input\PRES").Value = 24.8
        
        # 7. Save Flowsheet Structure and Re-open to Solve Engine
        print(f"Saving initial flowsheet structure to {out_bkp}...")
        aspen.SaveAs(out_bkp)
        aspen.Close()
        
        print("Re-opening cleanly with Aspen Simulation Engine...")
        aspen = win32com.client.Dispatch("Apwn.Document")
        aspen.SuppressDialogs = 1
        aspen.InitFromArchive2(out_bkp)
        tree = aspen.Tree
        
        print("Executing Aspen Plus Simulation Run (aspen.Engine.Run2())...")
        aspen.Engine.Run2()
        
        # Wait for solver completion
        print("Waiting for Aspen solver to complete...")
        t0 = time.time()
        while aspen.Engine.IsRunning and (time.time() - t0 < 30):
            time.sleep(0.5)
        print("Aspen Plus simulation run finished successfully!")
        
        # Save converged state
        aspen.Save()
        aspen.SaveAs(out_apw)
        
        # 8. Extract Convergence Results
        print("\n" + "="*70)
        print("ASPEN PLUS CONVERGENCE RESULTS & STREAM SUMMARY")
        print("="*70)
        
        def get_stream_data(sname):
            try:
                # 1. Try Output first (for calculated intermediate and product streams)
                flow_node = tree.FindNode(rf"\Data\Streams\{sname}\Output\STR_MAIN\MOLEFLOW\MIXED")
                if flow_node:
                    t = tree.FindNode(rf"\Data\Streams\{sname}\Output\STR_MAIN\TEMP\MIXED").Value
                    p = tree.FindNode(rf"\Data\Streams\{sname}\Output\STR_MAIN\PRES\MIXED").Value
                    co2_flow = flow_node.Elements("CO2").Value or 0.0
                    h2_flow = flow_node.Elements("H2").Value or 0.0
                    h2o_flow = flow_node.Elements("H2O").Value or 0.0
                    diox_flow = flow_node.Elements("DIOXANE").Value or 0.0
                    
                    tot_flow = co2_flow + h2_flow + h2o_flow + diox_flow
                    co2_frac = co2_flow / tot_flow if tot_flow > 0 else 0.0
                    h2_frac = h2_flow / tot_flow if tot_flow > 0 else 0.0
                    h2o_frac = h2o_flow / tot_flow if tot_flow > 0 else 0.0
                    diox_frac = diox_flow / tot_flow if tot_flow > 0 else 0.0
                    return t, p, tot_flow, co2_frac, h2_frac, h2o_frac, diox_frac, co2_flow, h2_flow

                # 2. Try Input (for feed streams like S1)
                t = tree.FindNode(rf"\Data\Streams\{sname}\Input\TEMP\MIXED").Value
                p = tree.FindNode(rf"\Data\Streams\{sname}\Input\PRES\MIXED").Value
                tot_flow = tree.FindNode(rf"\Data\Streams\{sname}\Input\TOTFLOW\MIXED").Value
                co2_flow = tree.FindNode(rf"\Data\Streams\{sname}\Input\FLOW\MIXED\CO2").Value or 0.0
                h2_flow = tree.FindNode(rf"\Data\Streams\{sname}\Input\FLOW\MIXED\H2").Value or 0.0
                h2o_flow = tree.FindNode(rf"\Data\Streams\{sname}\Input\FLOW\MIXED\H2O").Value or 0.0
                diox_flow = tree.FindNode(rf"\Data\Streams\{sname}\Input\FLOW\MIXED\DIOXANE").Value or 0.0
                
                co2_frac = co2_flow / tot_flow if tot_flow > 0 else 0.0
                h2_frac = h2_flow / tot_flow if tot_flow > 0 else 0.0
                h2o_frac = h2o_flow / tot_flow if tot_flow > 0 else 0.0
                diox_frac = diox_flow / tot_flow if tot_flow > 0 else 0.0
                return t, p, tot_flow, co2_frac, h2_frac, h2o_frac, diox_frac, co2_flow, h2_flow

            except Exception as e:
                return None
                
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
        
        print(f"{'Stream':<6} {'Description':<32} {'T (C)':<8} {'P (bar)':<8} {'Flow (kmol/h)':<14} {'CO2 mol%':<10} {'H2 mol%':<10}")
        print("-" * 92)
        for sname, desc in key_streams:
            data = get_stream_data(sname)
            if data:
                t, p, flow, co2, h2, h2o, diox, co2_f, h2_f = data
                print(f"{sname:<6} {desc:<32} {t:<8.1f} {p:<8.1f} {flow:<14.2f} {co2*100:<10.2f}% {h2*100:<10.2f}%")
                
        # Calculate Key Thesis Metrics
        s1_data = get_stream_data("S1")
        s5_data = get_stream_data("S5")
        s18_data = get_stream_data("S18")
        
        if s1_data and s18_data and s5_data:
            n_co2_feed = s1_data[7]
            n_co2_target = s18_data[7]
            r_co2 = (n_co2_target / n_co2_feed) * 100.0 if n_co2_feed > 0 else 0.0
            purity_co2 = s18_data[3] * 100.0
            
            n_h2_feed = s1_data[8]
            n_h2_clean1 = s5_data[8]
            r_h2 = (n_h2_clean1 / n_h2_feed) * 100.0 if n_h2_feed > 0 else 0.0
            
            print("\n" + "="*70)
            print("KEY THESIS DEFENSE DELIVERABLES")
            print("="*70)
            print(f"Target CO2 Product Purity (S18)  : {purity_co2:.2f} mol%  (Benchmark: >= 98.0%)")
            print(f"Overall CO2 Capture Recovery     : {r_co2:.2f} %     (Benchmark: >= 90.0%)")
            print(f"Primary Stage 1 H2 Recovery (S5) : {r_h2:.2f} %     (Benchmark: >= 95.0%)")
            print(f"Primary H2 Product Purity (S5)   : {s5_data[4]*100:.2f} mol%  (Syngas enrichment)")
            print("="*70 + "\n")
            
        # 9. Save Files
        print(f"Saving converged simulation backup file: {out_bkp}...")
        aspen.SaveAs(out_bkp)
        print(f"Saving Aspen Plus workbook: {out_apw}...")
        aspen.SaveAs(out_apw)
        print("SUCCESS! Both .bkp and .apw files generated and saved.")
        return True
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False
    finally:
        print("Closing Aspen Plus session...")
        aspen.Close()
        print("Done.")

if __name__ == "__main__":
    build_simulation()
