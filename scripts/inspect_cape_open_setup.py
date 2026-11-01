"""
inspect_cape_open_setup.py
"""
import os
import win32com.client

bkp_path = os.path.abspath("aspen_simulation/cape_open_simulation/Two_Stage_Hydrate_Capture_CAPE_OPEN.bkp")
print("Connecting to Aspen...")
aspen = win32com.client.Dispatch("Apwn.Document")
aspen.SuppressDialogs = 1

try:
    print(f"Loading {bkp_path}...")
    aspen.InitFromArchive2(bkp_path)
    tree = aspen.Tree

    # Inspect COMP2
    print("\n--- COMP2 Input ---")
    comp2_in = tree.FindNode(r"\Data\Blocks\COMP2\Input")
    if comp2_in:
        for i in range(comp2_in.Elements.Count):
            el = comp2_in.Elements(i)
            if el.Value is not None:
                print(f"  {el.Name} = {el.Value}")

    # Inspect CHILL4
    print("\n--- CHILL4 Input ---")
    chill4_in = tree.FindNode(r"\Data\Blocks\CHILL4\Input")
    if chill4_in:
        for i in range(chill4_in.Elements.Count):
            el = chill4_in.Elements(i)
            if el.Value is not None:
                print(f"  {el.Name} = {el.Value}")

    # Inspect CHILL2
    print("\n--- CHILL2 Input ---")
    chill2_in = tree.FindNode(r"\Data\Blocks\CHILL2\Input")
    if chill2_in:
        for i in range(chill2_in.Elements.Count):
            el = chill2_in.Elements(i)
            if el.Value is not None:
                print(f"  {el.Name} = {el.Value}")

    # Inspect Convergence blocks
    print("\n--- Convergence Blocks ---")
    conv_blocks = tree.FindNode(r"\Data\Convergence\Convergence")
    if conv_blocks:
        for i in range(conv_blocks.Elements.Count):
            cb = conv_blocks.Elements(i)
            print(f"Convergence Block: {cb.Name}")
            c_in = tree.FindNode(rf"\Data\Convergence\Convergence\{cb.Name}\Input")
            if c_in:
                for j in range(c_in.Elements.Count):
                    el = c_in.Elements(j)
                    if el.Value is not None:
                        print(f"    {el.Name} = {el.Value}")
            tear = tree.FindNode(rf"\Data\Convergence\Convergence\{cb.Name}\Input\TEAR")
            if tear:
                print(f"    Tear streams: {[tear.Elements(j).Name for j in range(tear.Elements.Count)]}")

    # Inspect Conv-Options
    print("\n--- Conv-Options ---")
    conv_opt = tree.FindNode(r"\Data\Convergence\Conv-Options\Input")
    if conv_opt:
        for i in range(conv_opt.Elements.Count):
            el = conv_opt.Elements(i)
            if el.Value is not None:
                print(f"  {el.Name} = {el.Value}")

    # Sequence / order
    print("\n--- Flowsheet Sequence ---")
    seq = tree.FindNode(r"\Data\Convergence\Sequence")
    if seq:
        for i in range(seq.Elements.Count):
            print(f"  Seq {i}: {seq.Elements(i).Name} = {seq.Elements(i).Value}")

finally:
    aspen.Quit()
    print("Closed Aspen.")
