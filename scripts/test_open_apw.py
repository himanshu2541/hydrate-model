"""
test_open_apw.py
"""
import os
import win32com.client

apw_path = os.path.abspath("aspen_simulation/cape_open_simulation/Two_Stage_Hydrate_Capture_CAPE_OPEN.apw")
print(f"Testing opening: {apw_path}")
aspen = win32com.client.Dispatch("Apwn.Document")
aspen.SuppressDialogs = 1

try:
    aspen.InitFromFile2(apw_path)
    print("Successfully opened .apw file via InitFromFile2!")
    tree = aspen.Tree
    blocks = tree.FindNode(r"\Data\Blocks")
    print("Blocks count:", blocks.Elements.Count)
    for i in range(blocks.Elements.Count):
        print(f"  {blocks.Elements(i).Name}")
        
    # Check results in S18 and S5
    s18_flow = tree.FindNode(r"\Data\Streams\S18\Output\STR_MAIN\MOLEFLOW\MIXED")
    if s18_flow:
        co2 = float(s18_flow.Elements("CO2").Value or 0.0)
        tot = sum(float(s18_flow.Elements(c).Value or 0.0) for c in ["CO2", "H2", "H2O", "DIOXANE"])
        print(f"S18 Total: {tot:.2f} kmol/h, CO2: {co2:.2f} ({co2/tot*100:.2f}%)")
    else:
        print("S18 has no output flow node")

    s5_flow = tree.FindNode(r"\Data\Streams\S5\Output\STR_MAIN\MOLEFLOW\MIXED")
    if s5_flow:
        h2 = float(s5_flow.Elements("H2").Value or 0.0)
        tot = sum(float(s5_flow.Elements(c).Value or 0.0) for c in ["CO2", "H2", "H2O", "DIOXANE"])
        print(f"S5 Total: {tot:.2f} kmol/h, H2: {h2:.2f} ({h2/tot*100:.2f}%)")
    else:
        print("S5 has no output flow node")

    s6_flow = tree.FindNode(r"\Data\Streams\S6\Output\STR_MAIN\MOLEFLOW\MIXED")
    if s6_flow:
        tot = sum(float(s6_flow.Elements(c).Value or 0.0) for c in ["CO2", "H2", "H2O", "DIOXANE"])
        print(f"S6 Total: {tot:.2f} kmol/h")
        
finally:
    aspen.Quit()
    print("Closed.")
