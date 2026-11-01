"""
check_tear_streams.py
"""
import os
import win32com.client

apw_path = os.path.abspath("aspen_simulation/cape_open_simulation/Two_Stage_Hydrate_Capture_CAPE_OPEN.apw")
aspen = win32com.client.Dispatch("Apwn.Document")
aspen.SuppressDialogs = 1

try:
    aspen.InitFromFile2(apw_path)
    tree = aspen.Tree

    for sname in ["S11", "S21"]:
        print(f"\n=== Stream {sname} ===")
        # Input
        in_t = tree.FindNode(rf"\Data\Streams\{sname}\Input\TEMP\MIXED")
        in_p = tree.FindNode(rf"\Data\Streams\{sname}\Input\PRES\MIXED")
        in_tot = tree.FindNode(rf"\Data\Streams\{sname}\Input\TOTFLOW\MIXED")
        in_comps = {}
        for c in ["CO2", "H2", "H2O", "DIOXANE"]:
            cn = tree.FindNode(rf"\Data\Streams\{sname}\Input\FLOW\MIXED\{c}")
            in_comps[c] = cn.Value if cn else 0.0
        print(f"  Input Spec: T={in_t.Value if in_t else None} C, P={in_p.Value if in_p else None} bar, Tot={in_tot.Value if in_tot else None} kmol/h")
        print(f"  Input Comps: {in_comps}")

        # Output
        out_t = tree.FindNode(rf"\Data\Streams\{sname}\Output\STR_MAIN\TEMP\MIXED")
        out_p = tree.FindNode(rf"\Data\Streams\{sname}\Output\STR_MAIN\PRES\MIXED")
        out_flow = tree.FindNode(rf"\Data\Streams\{sname}\Output\STR_MAIN\MOLEFLOW\MIXED")
        out_comps = {}
        out_tot = 0.0
        if out_flow:
            for c in ["CO2", "H2", "H2O", "DIOXANE"]:
                cn = out_flow.Elements(c)
                val = float(cn.Value or 0.0)
                out_comps[c] = val
                out_tot += val
        print(f"  Output Calc: T={out_t.Value if out_t else None} C, P={out_p.Value if out_p else None} bar, Tot={out_tot:.4f} kmol/h")
        print(f"  Output Comps: {out_comps}")

finally:
    aspen.Quit()
