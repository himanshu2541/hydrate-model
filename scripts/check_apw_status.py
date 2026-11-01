"""
check_apw_status.py
"""
import os
import win32com.client

apw_path = os.path.abspath("aspen_simulation/cape_open_simulation/Two_Stage_Hydrate_Capture_CAPE_OPEN.apw")
aspen = win32com.client.Dispatch("Apwn.Document")
aspen.SuppressDialogs = 1

try:
    aspen.InitFromFile2(apw_path)
    tree = aspen.Tree

    # Check status
    stat_node = tree.FindNode(r"\Data\Results Summary\Run-Status\Output")
    if stat_node:
        for i in range(stat_node.Elements.Count):
            el = stat_node.Elements(i)
            print(f"Status {el.Name}: {el.Value}")

    # Check ERRTXT
    err = tree.FindNode(r"\Data\Results Summary\Run-Status\Output\ERRTXT")
    if err:
        print(f"ERRTXT: {err.Value}")

    # Check Convergence blocks status
    for c in ["$OLVER01", "$OLVER02"]:
        c_stat = tree.FindNode(rf"\Data\Convergence\Convergence\{c}\Output")
        if c_stat:
            print(f"\n{c} Output:")
            for j in range(c_stat.Elements.Count):
                el = c_stat.Elements(j)
                print(f"  {el.Name} = {el.Value}")

    # Check Blocks status
    blocks = tree.FindNode(r"\Data\Blocks")
    print("\nBlocks status:")
    for i in range(blocks.Elements.Count):
        bname = blocks.Elements(i).Name
        b_stat = tree.FindNode(rf"\Data\Blocks\{bname}\Output\STATUS")
        b_err = tree.FindNode(rf"\Data\Blocks\{bname}\Output\ERR_STATUS")
        print(f"  {bname:<8}: status={b_stat.Value if b_stat else None}, err={b_err.Value if b_err else None}")

finally:
    aspen.Quit()
