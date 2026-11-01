"""
check_olver02_warning.py
"""
import os
import glob

# Find the most recently modified .his file in cape_open_simulation
his_files = glob.glob("aspen_simulation/cape_open_simulation/*.his")
his_files.sort(key=os.path.getmtime, reverse=True)

if his_files:
    latest_his = his_files[0]
    print(f"Latest his file: {latest_his}")
    with open(latest_his, "r", encoding="latin1") as f:
        lines = f.readlines()
    print("Total lines:", len(lines))
    print("--- Tail lines ---")
    for l in lines[-100:]:
        print(l, end="")
else:
    print("No his file found.")
