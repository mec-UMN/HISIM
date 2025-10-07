#Cost Model have been adopted from following works:
# Ref [1]: A. Graening, J. Talukdar, S. Pal, K. Chakrabarty and P. Gupta, "CATCH: A Cost Analysis Tool for Co-Optimization of Chiplet-Based Heterogeneous systems," in IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems
# Ref [2]: M. Ahmad, J. DeLaCruz and A. Ramamurthy, "Heterogeneous Integration of Chiplets: Cost and Yield Tradeoff Analysis," 2022 23rd International Conference on Thermal, Mechanical and Multi-Physics Simulation and Experiments in Microelectronics and Microsystems (EuroSimE),
# Ref [3]: Tianqi Tang and Yuan Xie , “Cost-Aware Exploration for Chiplet-Based Architecture with Advanced Packaging Technologies”  HiPChips Chiplet Workshop, ISCA 2022
# Ref [4]: Yinxiao Feng and Kaisheng Ma. Chiplet actuary: a quantitative cost model and multi-chiplet architecture exploration. In Proceedings of the 59th ACM/IEEE Design Automation Conference 2022

#Most values in the cost.json are taken from github repos of ref [1] and ref [4]: 
#https://github.com/nanocad-lab/CATCH/tree/main
#https://github.com/Yinxiao-Feng/chiplet-actuary/tree/dac2022
import math

# Interposer Cost models from Ref [1]
def negative_binomial(D0, A, alpha):
    yield_nb = (1+D0*A/alpha)**(-alpha)
    return yield_nb

def cost_die(k_silicon, A_chip, A_reticle_unit, D0_chip, alpha_chip,litho_percent, manuf_vol):
    Y_die =negative_binomial(D0_chip, A_chip, alpha_chip)
    k_die = k_silicon * A_chip 
    A_reticle = A_reticle_unit
    # If the chip area is larger than the reticle area, this requires stitching.
    # The model assumes multiple reticles are used, effectively increasing the "total" reticle area.
    while A_reticle < A_chip:
        A_reticle += A_reticle_unit
    # Calculate how many full chips can be patterned within one reticle exposure area.
    if A_chip == 0:
        return 0.0, 1.0
    number_chips_in_reticle = A_reticle//A_chip 
    unutilized_reticle = (A_reticle) - number_chips_in_reticle*A_chip
    reticle_utilization = (A_reticle - unutilized_reticle)/(A_reticle)
    k_die= k_die*(1-litho_percent)+ (k_die*litho_percent)/reticle_utilization
    C_die = k_die/Y_die
    #cost for 1 million units 
    C_die*=manuf_vol
    return C_die, Y_die

def cost_substrate(k_sub, Demand_sub, D0_sub, A_sub, alpha_sub, manuf_vol):
    P_unit_sub = k_sub * A_sub
    Y_sub = negative_binomial(D0_sub, A_sub, alpha_sub)
    C_sub = P_unit_sub*Demand_sub/Y_sub
    #cost for 1 million units
    C_sub*=manuf_vol
    return C_sub, Y_sub

def compute_machine_cost_per_second(k_machine, T_machine_lifetime, k_machine_technician, T_machine_uptime):
    k_machine_year = k_machine/T_machine_lifetime + k_machine_technician
    # Convert yearly cost to cost per second, accounting for machine uptime.
    k_machine_sec = k_machine_year/(365*24*60*60)*T_machine_uptime
    return k_machine_sec

def cost_assembly(k_place, T_place, k_bond, T_bond, 
                    T_place_lifetime, k_place_technician, T_place_uptime, T_bond_lifetime, k_bond_technician, T_bond_uptime, 
                    Y_alignment, Y_bond, Y_TSV, N_die, N_bonds, N_TSV, A_bond, D0_HB, manuf_vol):
    T_place_total = N_die*T_place
    T_bond_total = N_die*T_bond
    k_place_sec = compute_machine_cost_per_second(k_place, T_place_lifetime, k_place_technician, T_place_uptime)
    k_bond_sec = compute_machine_cost_per_second(k_bond, T_bond_lifetime, k_bond_technician, T_bond_uptime)
    k_assembly = k_place_sec*T_place_total+k_bond_sec*T_bond_total
    Y_assembly = Y_alignment**N_die*Y_bond**N_bonds*Y_TSV**N_TSV*(1/(1+D0_HB*A_bond))
    C_assembly = k_assembly/Y_assembly
    #cost for 1 million units
    C_assembly*=manuf_vol
    #import pdb; pdb.set_trace()
    return C_assembly, Y_assembly

# Interposer Cost models from Ref [3]
def cost_interposer_silicon(C_wafer, A_interposer, N_interposers, r_wafer, D0_interposer,   alpha_interposer, manuf_vol):
    N_interposer_wafer = math.ceil(math.pi*r_wafer**2/A_interposer-math.pi*r_wafer/math.sqrt(2*A_interposer))
    Y_interposer = negative_binomial(D0_interposer, A_interposer, alpha_interposer)
    C_interposer = C_wafer*N_interposers/(N_interposer_wafer*Y_interposer)
    #cost for 1 million units
    C_interposer*=manuf_vol
    #import pdb; pdb.set_trace()
    return C_interposer, Y_interposer

def cost_interposer_organic(C_panel, A_interposer, A_panel, D0_interposer, alpha_interposer, manuf_vol):
    Y_interposer = negative_binomial(D0_interposer, A_interposer, alpha_interposer)
    C_interposer = C_panel*A_interposer/A_panel/Y_interposer
    #cost for 1 million units
    C_interposer*=manuf_vol
    return C_interposer, Y_interposer

#Test Cost models from Ref [2] and params from OCP-ODSA open source model - https://drive.google.com/drive/folders/1kphneR4UElaZTmnOYgh2za3fXg4v66_b
def cost_wafer_probe(C_ate_ws, C_probe_ws, C_site_ws, Y_die, cover_die_ws, T_per_insert_list_ws, T_reattempt_list_ws, manuf_vol):
    C_test_probe = (C_ate_ws + C_probe_ws)/3600/C_site_ws*(sum(T_per_insert_list_ws)+((1-Y_die)*cover_die_ws)*sum(T_reattempt_list_ws))
    #cost for 1 million units
    C_test_probe*=manuf_vol
    return C_test_probe

def cost_ftl(C_ate_ftl, C_probe_ftl, C_site_ftl, Y_die, cover_die_ftl, cover_die_ws, T_per_insert_list_ftl, T_reattempt_list_ftl, manuf_vol):
    C_test_ftl = (C_ate_ftl + C_probe_ftl)/3600/C_site_ftl*(sum(T_per_insert_list_ftl)+((1-Y_die)*(cover_die_ws-cover_die_ftl))*sum(T_reattempt_list_ftl))
    #cost for 1 million units
    C_test_ftl*=manuf_vol
    return C_test_ftl

def cost_wafer_stl(C_ate_stl, C_probe_stl, C_site_stl, Y_die, cover_die_ftl, T_per_insert_list_stl, manuf_vol):
    C_test_stl = (C_ate_stl + C_probe_stl)/3600/C_site_stl*(sum(T_per_insert_list_stl)*(Y_die+(1-Y_die)*(1-cover_die_ftl)))
    #cost for 1 million units
    C_test_stl*=manuf_vol
    return C_test_stl

#NRE Cost model from Ref [4]
def cost_nre_chiplet(A_chip, k_chip_nre, A_mod, k_mod_nre, C_chip_nre_fixed):
    C_nre_chiplet = A_chip * k_chip_nre + A_mod * k_mod_nre + C_chip_nre_fixed
    return C_nre_chiplet

def cost_nre_pkg(A_interposer, k_interposer_nre, A_pkg, k_pkg_nre, C_pkg_nre_fixed):
    C_nre_pkg = A_interposer * k_interposer_nre + A_pkg * k_pkg_nre + C_pkg_nre_fixed
    return C_nre_pkg

def cost_nre_d2d(A_d2d, C_D2D_nre):
    C_nre_d2d = C_D2D_nre * A_d2d
    return C_nre_d2d