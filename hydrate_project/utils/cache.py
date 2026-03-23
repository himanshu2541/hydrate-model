import sqlite3
import hashlib
import json
import pandas as pd

DB_NAME = "hydrate_results.db"

def get_run_hash(eos_name, gas_comp, liq_comp, T_range):
    """Generates a unique fingerprint for the simulation parameters."""
    run_params = {
        "eos": eos_name,
        "gas": dict(sorted(gas_comp.items())),
        "liq": dict(sorted(liq_comp.items())),
        "T_min": float(min(T_range)),
        "T_max": float(max(T_range)),
        "T_step": float(T_range[1] - T_range[0]) if len(T_range) > 1 else 0.0
    }
    param_str = json.dumps(run_params, sort_keys=True)
    return hashlib.md5(param_str.encode('utf-8')).hexdigest()

def load_from_cache(run_hash):
    """Attempts to load a previous run from the SQLite database."""
    try:
        with sqlite3.connect(DB_NAME) as conn:
            query = f"SELECT * FROM results WHERE run_hash = '{run_hash}'"
            df = pd.read_sql(query, conn)
            if not df.empty:
                return df.drop(columns=['run_hash'])
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        # Catches both raw SQLite missing table errors AND Pandas missing table errors
        pass 
    return None

def save_to_cache(df, run_hash):
    """Saves a successfully calculated dataframe to the SQLite database."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        try:
            # Delete old records with the same hash just in case of an override
            cursor.execute(f"DELETE FROM results WHERE run_hash = '{run_hash}'")
        except sqlite3.OperationalError:
            pass # Table doesn't exist yet, which is fine
            
        df_to_save = df.copy()
        df_to_save['run_hash'] = run_hash
        df_to_save.to_sql('results', conn, if_exists='append', index=False)