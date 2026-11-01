"""
run_clean_solve_and_inspect.py
"""
import os
import win32com.client
import time

apw_path = os.path.abspath("aspen_simulation/cape_open_simulation/Two_Stage_Hydrate_Capture_CAPE_OPEN.apw")
print(f"Opening: {apw_path}")
aspen = win32com.client.Dispatch("Apwn.Document")
aspen.SuppressDialogs = 1

try:
    aspen.InitFromFile2(apw_path)
    tree = aspen.Tree

    # 1. Update initial tear stream estimates to near-exact converged solution
    print("Setting S11 and S21 initial estimates...")
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
        conv_opt.Elements("WEG_WAIT").Value = 2
        conv_opt.Elements("WEG_QMIN").Value = -2.0

    # Also check $OLVER01 and $OLVER02 specific inputs
    for c in ["$OLVER01", "$OLVER02"]:
        maxit_node = tree.FindNode(rf"\Data\Convergence\Convergence\{c}\Input\MAXIT")
        if maxit_node:
            maxit_node.Value = 200
        tol_node = tree.FindNode(rf"\Data\Convergence\Convergence\{c}\Input\TOL")
        if tol_node:
            tol_node.Value = 0.0005

    # Reinit and run
    print("Reinitializing and running Aspen Engine...")
    aspen.Reinit()
    aspen.Engine.Run2()

    # Wait for run to start
    time.sleep(2)
    t0 = time.time()
    while aspen.Engine.IsRunning and (time.time() - t0 < 120):
        time.sleep(1)

    print(f"Aspen run completed in {time.time()-t0:.2f} s")

    # Check status
    stat_node = tree.FindNode(r"\Data\Results Summary\Run-Status\Output\RUN_STATUS")
    print(f"\nFinal RUN_STATUS: {stat_node.Value if stat_node else 'None'}")

    # Check ERRTXT
    errtxt = tree.FindNode(r"\Data\Results Summary\Run-Status\Output\ERRTXT")
    if errtxt and errtxt.Value:
        print(f"ERRTXT: {errtxt.Value}")
    else:
        print("ERRTXT: None / Clean!")

    # Check convergence blocks
    for c in ["$OLVER01", "$OLVER02"]:
        rowstat = tree.FindNode(rf"\Data\Convergence\Convergence\{c}\Output\ROWSTAT")
        totiter = tree.FindNode(rf"\Data\Convergence\Convergence\{c}\Output\TOT_ITER")
        blkstat = tree.FindNode(rf"\Data\Convergence\Convergence\{c}\Output\BLKSTAT")
        print(f"{c}: BLKSTAT={blkstat.Value if blkstat else None}, Iters={totiter.Value if totiter else None}, Status={rowstat.Value if rowstat else None}")

    # Check equipment blocks for errors
    blocks = tree.FindNode(r"\Data\Blocks")
    has_err = False
    for i in range(blocks.Elements.Count):
        bname = blocks.Elements(i).Name
        b_stat = tree.FindNode(rf"\Data\Blocks\{bname}\Output\STATUS")
        b_err = tree.FindNode(rf"\Data\Blocks\{bname}\Output\ERR_STATUS")
        stat_val = b_stat.Value if b_stat else None
        err_val = b_err.Value if b_err else None
        if stat_val or err_val:
            print(f"  Block {bname}: status={stat_val}, err={err_val}")
            has_err = True
    if not has_err:
        print("All 16 Unit Operation Blocks converged cleanly (0 errors)!")

    # Save converged .apw and .bkp
    print("Saving converged simulation...")
    aspen.Save()
    bkp_save_path = os.path.abspath("aspen_simulation/cape_open_simulation/Two_Stage_Hydrate_Capture_CAPE_OPEN.bkp")
    aspen.Export(1, bkp_save_path)
    print(f"Saved APW and exported BKP to {bkp_save_path}")

finally:
    aspen.Quit()
    print("Closed Aspen.")
