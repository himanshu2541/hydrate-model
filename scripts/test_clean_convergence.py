"""
test_clean_convergence.py
"""
import os
import win32com.client
import time

apw_path = os.path.abspath("aspen_simulation/cape_open_simulation/Two_Stage_Hydrate_Capture_CAPE_OPEN.apw")
print("Connecting to Aspen...")
aspen = win32com.client.Dispatch("Apwn.Document")
aspen.SuppressDialogs = 1

try:
    print(f"Loading {apw_path}...")
    aspen.InitFromFile2(apw_path)
    tree = aspen.Tree

    # 1. Update initial tear stream estimates to near-exact converged solution
    print("Updating S11 and S21 initial estimates...")
    tree.FindNode(r"\Data\Streams\S11\Input\TEMP\MIXED").Value = 1.0
    tree.FindNode(r"\Data\Streams\S11\Input\PRES\MIXED").Value = 24.8
    tree.FindNode(r"\Data\Streams\S11\Input\TOTFLOW\MIXED").Value = 1032.80
    tree.FindNode(r"\Data\Streams\S11\Input\FLOW\MIXED\H2O").Value = 1032.737
    tree.FindNode(r"\Data\Streams\S11\Input\FLOW\MIXED\CO2").Value = 0.0663
    tree.FindNode(r"\Data\Streams\S11\Input\FLOW\MIXED\H2").Value = 0.0
    tree.FindNode(r"\Data\Streams\S11\Input\FLOW\MIXED\DIOXANE").Value = 0.0

    tree.FindNode(r"\Data\Streams\S21\Input\TEMP\MIXED").Value = 1.0
    tree.FindNode(r"\Data\Streams\S21\Input\PRES\MIXED").Value = 24.8
    tree.FindNode(r"\Data\Streams\S21\Input\TOTFLOW\MIXED").Value = 361.86
    tree.FindNode(r"\Data\Streams\S21\Input\FLOW\MIXED\H2O").Value = 361.833
    tree.FindNode(r"\Data\Streams\S21\Input\FLOW\MIXED\CO2").Value = 0.0235
    tree.FindNode(r"\Data\Streams\S21\Input\FLOW\MIXED\H2").Value = 0.0
    tree.FindNode(r"\Data\Streams\S21\Input\FLOW\MIXED\DIOXANE").Value = 0.0

    # 2. Adjust Convergence Options
    conv_opt = tree.FindNode(r"\Data\Convergence\Conv-Options\Input")
    if conv_opt:
        conv_opt.Elements("WEG_MAXIT").Value = 200
        conv_opt.Elements("TOL").Value = 0.0005
        # Damping or wait iterations
        conv_opt.Elements("WEG_WAIT").Value = 2
        conv_opt.Elements("WEG_QMIN").Value = -2.0

    # Reinit simulation
    print("Reinitializing flowsheet...")
    aspen.Reinit()

    print("Running Engine...")
    aspen.Engine.Run2()

    t0 = time.time()
    while aspen.Engine.IsRunning and (time.time() - t0 < 60):
        time.sleep(1)

    print(f"Run completed in {time.time()-t0:.2f} s")

    # Check status
    stat_node = tree.FindNode(r"\Data\Results Summary\Run-Status\Output\RUN_STATUS")
    print(f"RUN_STATUS: {stat_node.Value if stat_node else 'None'}")
    
    errtxt = tree.FindNode(r"\Data\Results Summary\Run-Status\Output\ERRTXT")
    if errtxt and errtxt.Value:
        print(f"ERRTXT:\n{errtxt.Value}")

    # Check $OLVER01 and $OLVER02
    for c in ["$OLVER01", "$OLVER02"]:
        rowstat = tree.FindNode(rf"\Data\Convergence\Convergence\{c}\Output\ROWSTAT")
        totiter = tree.FindNode(rf"\Data\Convergence\Convergence\{c}\Output\TOT_ITER")
        blkstat = tree.FindNode(rf"\Data\Convergence\Convergence\{c}\Output\BLKSTAT")
        print(f"{c}: BLKSTAT={blkstat.Value if blkstat else None}, Iters={totiter.Value if totiter else None}, Status={rowstat.Value if rowstat else None}")

finally:
    aspen.Quit()
    print("Closed Aspen.")
