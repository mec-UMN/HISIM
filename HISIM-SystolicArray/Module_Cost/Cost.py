#Cost Model have been adopted from following works:
# Ref [1]: A. Graening, J. Talukdar, S. Pal, K. Chakrabarty and P. Gupta, "CATCH: A Cost Analysis Tool for Co-Optimization of Chiplet-Based Heterogeneous systems," in IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems
# Ref [2]: M. Ahmad, J. DeLaCruz and A. Ramamurthy, "Heterogeneous Integration of Chiplets: Cost and Yield Tradeoff Analysis," 2022 23rd International Conference on Thermal, Mechanical and Multi-Physics Simulation and Experiments in Microelectronics and Microsystems (EuroSimE),
# Ref [3]: Tianqi Tang and Yuan Xie , “Cost-Aware Exploration for Chiplet-Based Architecture with Advanced Packaging Technologies”  HiPChips Chiplet Workshop, ISCA 2022
# Ref [4]: Yinxiao Feng and Kaisheng Ma. Chiplet actuary: a quantitative cost model and multi-chiplet architecture exploration. In Proceedings of the 59th ACM/IEEE Design Automation Conference 2022

import math

# Interposer Cost models from Ref [1]
def negative_binomial(D0, A, alpha):
    yield_nb = (1+D0*A/alpha)**(-alpha)
    return yield_nb

def cost_die(k_silicon, k_exposures, A_chip, A_reticle, D0_chip, alpha_chip):
    Y_die =negative_binomial(D0_chip, A_chip, alpha_chip)
    C_die = k_die/Y_die
    k_die = k_silicon * A_chip + k_exposures / math.floor(A_reticle/A_chip)
    return C_die, k_die

def cost_substrate(P_unit_sub, Demand_sub, D0_sub, A_sub, alpha_sub):
    Y_sub = negative_binomial(D0_sub, A_sub, alpha_sub)
    C_sub = P_unit_sub*Demand_sub/Y_sub
    return C_sub, Y_sub

def cost_assembly(k_place, T_place, k_bond, T_bond, Y_alignment, Y_bond, Y_TSV, N_die, N_pins, N_TSV, A_bond, D0_HB):
    C_assembly = k_place*T_place+k_bond*T_bond
    Y_assembly = Y_alignment**N_die*Y_bond**N_pins*Y_TSV**N_TSV*(1/(1+D0_HB*A_bond))
    return C_assembly, Y_assembly

# Interposer Cost models from Ref [3]
def cost_interposer_silicon(C_wafer, A_interposer, N_interposers, r_wafer, D0_interposer,   alpha_interposer):
    N_interposer = math.pi*r_wafer**2/A_interposer-math.pi*r_wafer/math.sqrt(2*A_interposer)
    Y_interposer = negative_binomial(D0_interposer, A_interposer, alpha_interposer)
    C_interposer = C_wafer/(N_interposer*Y_interposer)
    return C_interposer, Y_interposer

def cost_interposer_organic(C_panel, A_interposer, A_panel, D0_interposer, alpha_interposer):
    Y_interposer = negative_binomial(D0_interposer, A_interposer, alpha_interposer)
    C_interposer = C_panel*A_interposer/A_panel/Y_interposer
    return C_interposer, Y_interposer

#Test Cost models from Ref [2]
def cost_wafer_probe(C_ate_ws, C_probe_ws, C_site_ws, Y_die, cover_die_ws, T_per_insert_list_ws, T_reattempt_list_ws):
    C_test_probe = (C_ate_ws + C_probe_ws)/3600/C_site_ws*(sum(T_per_insert_list_ws)+((1-Y_die)*cover_die_ws)*sum(T_per_insert_list_ws[i] for i in T_reattempt_list_ws))
    return C_test_probe

def cost_ftl(C_ate_ftl, C_probe_ftl, C_site_ftl, Y_die, cover_die_ftl, cover_die_ws, T_per_insert_list_ftl, T_reattempt_list_ftl):
    C_test_ftl = (C_ate_ftl + C_probe_ftl)/3600/C_site_ftl*(sum(T_per_insert_list_ftl)+((1-Y_die)*(cover_die_ws-cover_die_ftl))*sum(T_per_insert_list_ftl[i] for i in T_reattempt_list_ftl))
    return C_test_ftl

def cost_wafer_stl(C_ate_stl, C_probe_stl, C_site_stl, Y_die, cover_die_stl, T_per_insert_list_stl):
    C_test_stl = (C_ate_stl + C_probe_stl)/3600/C_site_stl*(sum(T_per_insert_list_stl)*(Y_die+(1-Y_die)*(1-cover_die_stl)))
    return C_test_stl

#NRE Cost model from Ref [4]
def cost_nre(A_mod_nre_list, A_chip_nre_list, A_pkg_nre_list, C_pkg_nre_fixed, C_chip_nre_fixed, C_D2D_nre, k_pkg_nre, k_chip_nre, k_mod_nre):
    C_nre = sum(A_pkg * k_pkg + C_pkg for A_pkg, k_pkg, C_pkg in zip(A_pkg_nre_list, k_pkg_nre, C_pkg_nre_fixed))
    C_nre += sum(A_chip * k_chip + A_mod * k_mod + C_chip for A_chip, k_chip, C_chip, A_mod, k_mod in zip(A_chip_nre_list, k_chip_nre, C_chip_nre_fixed, A_mod_nre_list, k_mod_nre))
    C_nre += C_D2D_nre
    return C_nre