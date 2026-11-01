"""
run_model1_check.py
"""
import os
import win32com.client
import time

bkp_path = os.path.abspath("aspen_simulation/Two_Stage_Hydrate_Capture_Final.bkp")
print("Connecting to Aspen...")
aspen = win32com.client.Dispatch("Apwn.Document")
aspen.SuppressDialogs = 1

try:
    print(f"Loading {bkp_path}...")
    aspen.InitFromArchive2(bkp_path)
    tree = aspen.Tree

    # Check Run Status before
    stat = tree.FindNode(r"\Data\Results Summary\Run-Status\Output\RUN_STATUS")
    print(f"Initial Run Status: {stat.Value if stat else 'None'}")

    print("Running Model 1 Engine...")
    aspen.Engine.Run2()
    t0 = time.time()
    while aspen.Engine.IsRunning and (time.time() - t0 < 60):
        time.sleep(1)

    print(f"Run finished in {time.time()-t0:.2f} s")
    
    # Check Run Status after
    stat = tree.FindNode(r"\Data\Results Summary\Run-Status\Output\RUN_STATUS")
    print(f"Final Run Status: {stat.Value if stat else 'None'}")
    errtxt = tree.FindNode(r"\Data\Results Summary\Run-Status\Output\ERRTXT")
    if errtxt:
        print(f"ERRTXT: {errtxt.Value}")

    # Read S18 and S5
    s18_flow = tree.FindNode(r"\Data\Streams\S18\Output\STR_MAIN\MOLEFLOW\MIXED")
    if s18_flow:
        co2 = float(s18_flow.Elements("CO2").Value or 0.0)
        tot = sum(float(s18_flow.Elements(c).Value or 0.0) for c in ["CO2", "H2", "H2O", "DIOXANE"])
        print(f"S18 Total Flow: {tot:.2f} kmol/h, CO2 purity: {co2/tot*100:.2f}%")

    s5_flow = tree.FindNode(r"\Data\Streams\S5\Output\STR_MAIN\MOLEFLOW\MIXED")
    if s5_flow:
        h2 = float(s5_flow.Elements("H2").Value or 0.0)
        tot = sum(float(s5_flow.Elements(c).Value or 0.0) for c in ["CO2", "H2", "H2O", "DIOXANE"])
        print(f"S5 Total Flow: {tot:.2f} kmol/h, H2 purity: {h2/tot*100:.2f}%")

finally:
    aspen.Quit()
    print("Closed Aspen.")
