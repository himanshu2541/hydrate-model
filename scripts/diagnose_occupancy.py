from hydrate_project.core.database import Database
from hydrate_project.thermo_model.klauda_sandler import KlaudaSandlerModel
from hydrate_project.solvers.equilibrium import EquilibriumSolver
from hydrate_project.eos_model.pr_eos import PREOS

db = Database()
gas_comp = {'CO2': 0.4, 'H2': 0.6}
liq_comp = {'H2O': 0.9444, 'DIOX': 0.0556}
eos = PREOS(gas_comp, db)
km = KlaudaSandlerModel(db)
solver = EquilibriumSolver(liq_comp, db, km, eos)

T = 285.0
for P_MPa in [1.0, 5.0, 10.0, 20.0, 50.0]:
    P = P_MPa * 1e6
    f_dict, _, aw, _ = solver._get_liquid_and_fugacities(T, P)
    occ_s = km.calc_cage_occupancy(T, f_dict, 'sII', 'small')
    occ_l = km.calc_cage_occupancy(T, f_dict, 'sII', 'large')
    print(f"P={P_MPa:4.1f} MPa | small: CO2={occ_s.get('CO2',0):.4f}, H2={occ_s.get('H2',0):.4f} | large: DIOX={occ_l.get('DIOX',0):.4f}, CO2={occ_l.get('CO2',0):.4f}")
