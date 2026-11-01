"""
print_all_apw_streams.py
"""
import os
import win32com.client

apw_path = os.path.abspath("aspen_simulation/cape_open_simulation/Two_Stage_Hydrate_Capture_CAPE_OPEN.apw")
aspen = win32com.client.Dispatch("Apwn.Document")
aspen.SuppressDialogs = 1

try:
    aspen.InitFromFile2(apw_path)
    tree = aspen.Tree

    print(f"{'Stream':<6} {'T [C]':<8} {'P [bar]':<8} {'Flow [kmol/h]':<15} {'CO2 [kmol/h]':<14} {'H2 [kmol/h]':<14} {'H2O [kmol/h]':<14} {'DIOX [kmol/h]':<14}")
    print("-" * 90)

    for i in range(1, 22):
        sname = f"S{i}"
        flow_node = tree.FindNode(rf"\Data\Streams\{sname}\Output\STR_MAIN\MOLEFLOW\MIXED")
        if flow_node:
            t = float(tree.FindNode(rf"\Data\Streams\{sname}\Output\STR_MAIN\TEMP\MIXED").Value or 0.0)
            p = float(tree.FindNode(rf"\Data\Streams\{sname}\Output\STR_MAIN\PRES\MIXED").Value or 0.0)
            co2 = float(flow_node.Elements("CO2").Value or 0.0)
            h2 = float(flow_node.Elements("H2").Value or 0.0)
            h2o = float(flow_node.Elements("H2O").Value or 0.0)
            diox = float(flow_node.Elements("DIOXANE").Value or 0.0)
            tot = co2 + h2 + h2o + diox
            print(f"{sname:<6} {t:<8.1f} {p:<8.1f} {tot:<15.2f} {co2:<14.4f} {h2:<14.4f} {h2o:<14.4f} {diox:<14.4f}")
        else:
            in_node = tree.FindNode(rf"\Data\Streams\{sname}\Input")
            if in_node:
                t = float(tree.FindNode(rf"\Data\Streams\{sname}\Input\TEMP\MIXED").Value or 0.0)
                p = float(tree.FindNode(rf"\Data\Streams\{sname}\Input\PRES\MIXED").Value or 0.0)
                tot = float(tree.FindNode(rf"\Data\Streams\{sname}\Input\TOTFLOW\MIXED").Value or 0.0)
                co2 = float(tree.FindNode(rf"\Data\Streams\{sname}\Input\FLOW\MIXED\CO2").Value or 0.0)
                h2 = float(tree.FindNode(rf"\Data\Streams\{sname}\Input\FLOW\MIXED\H2").Value or 0.0)
                h2o = float(tree.FindNode(rf"\Data\Streams\{sname}\Input\FLOW\MIXED\H2O").Value or 0.0)
                diox = float(tree.FindNode(rf"\Data\Streams\{sname}\Input\FLOW\MIXED\DIOXANE").Value or 0.0)
                print(f"{sname:<6} (INPUT ONLY) {t:<8.1f} {p:<8.1f} {tot:<15.2f} {co2:<14.4f} {h2:<14.4f} {h2o:<14.4f} {diox:<14.4f}")
            else:
                print(f"{sname:<6} NOT FOUND")

finally:
    aspen.Quit()
