import config
import json
import matplotlib.pyplot as plt
import os
from Module_3_Cost.Cost_Models import *
import math

aimodel=config.aimodel
main_dir = config.main_dir
DEBUG=config.DEBUG
current_dir = os.path.dirname(__file__)
def cost_main_fn(G_chip, G_sys, nw_df, stack_area, stack_ids, mesh_size, ip_list, n_signal_IO):
    total_cost = 0
    manuf_vol=1e6
    cost_breakdown = {"die": {}, "assembly": {}, "interposer": {}, "substrate": {},  "test": {"wafer_probe": {}, "final_package": {}, "system_level": {}}, "nre": {}}
    yield_breakdown = {"die": {}, "assembly": {}, "interposer": {}, "substrate": {}}
    #read parameters from Cost.json
    with open(current_dir+'/Cost.json') as f:
        cost_params = json.load(f)
        #All cost parameters are extracted at 28nm node
    #Calculate cost of each component using cost models from Cost_Models.py
    #import pdb; pdb.set_trace()
    chip_ip_dict={}
    for chip_idx in G_sys.nodes():
        if G_sys.nodes[chip_idx]["Chiplet ID"]!="Empty":
            A_chip = G_sys.nodes[chip_idx]["Chiplet Actual Area"]
            cost_breakdown["die"][chip_idx] , yield_breakdown["die"][chip_idx] = cost_die(cost_params["k_silicon"], A_chip, cost_params["A_reticle_unit"], cost_params["D0_chip"], cost_params["alpha_chip"], cost_params["litho_percent"], cost_params["Y_perstitch"], cost_params["Y_wafer_process"], manuf_vol)
            cost_breakdown["test"]["wafer_probe"][chip_idx] = cost_wafer_probe(cost_params["C_ate_ws"], cost_params["C_probe_ws"], cost_params["C_site_ws"], yield_breakdown["die"][chip_idx], cost_params["cover_die_ws"], cost_params["T_per_insert_list_ws"], cost_params["T_reattempt_list_ws"], manuf_vol)
            ip_set_chip = set([ip for ip in ip_list if any(chip_tile_idx.split('_')[0]==chip_idx for chip_tile_idx in ip_list[ip])])
            A_ip = sum([G_chip.nodes[ip_list[ip][0]]["Tile_Area"]  for ip in ip_set_chip])
            #import pdb; pdb.set_trace()
            if tuple(sorted(ip_set_chip)) not in chip_ip_dict :
                C_nre = cost_nre_chiplet(A_chip, cost_params["k_chip_nre"], A_ip, cost_params["k_mod_nre"], cost_params["C_chip_nre_fixed"])
                A_d2d = G_sys.nodes[chip_idx].get("2.5D link Area", 0)
                C_nre += cost_nre_d2d(A_d2d, cost_params["C_D2D_nre"])* 4 #Assuming 4 different D2D connections per chiplet - North, South, East, West
                cost_breakdown["nre"][chip_idx] = C_nre
                chip_ip_dict[tuple(sorted(ip_set_chip))] = [chip_idx]
            else:
                chip_ip_dict[tuple(sorted(ip_set_chip))].append(chip_idx)
        #print(chip_ip_dict)
        #import pdb; pdb.set_trace()
    for stack_idx, A_bond in stack_area.items():
        if A_bond != 0:
            N_pins = math.ceil(n_signal_IO[stack_idx]/0.7) #70% signal and 30% power/ground
            N_dies = len(stack_ids[stack_idx]) 
            #import pdb; pdb.set_trace()
            if N_dies>1:
                N_pins_3d = nw_df.loc[nw_df["Stack ID"]==stack_idx, "N_3D_Links_per_tile"].values[0] # individual bonding of die to die
                N_pins_3d= math.ceil(N_pins_3d/0.7) #70% signal and 30% power/ground
                N_pins_3d*=mesh_size[next(iter(stack_ids[stack_idx]))]**2
                N_pins_3d*=(N_dies -1)
                N_pins+= N_pins_3d
                N_TSV = N_pins
            else:
                N_TSV = 0
            #print("Npins for stack ", stack_idx, " is ", N_pins)
            cost_breakdown["assembly"][stack_idx] , yield_breakdown["assembly"][stack_idx]  = cost_assembly(cost_params["k_place"], cost_params["T_place"], cost_params["k_bond"], cost_params["T_bond"], 
                                    cost_params["T_place_lifetime"], cost_params["k_place_technician"], cost_params["T_place_uptime"], cost_params["T_bond_lifetime"], cost_params["k_bond_technician"], cost_params["T_bond_uptime"],
                                    cost_params["Y_alignment"], cost_params["Y_bond"], cost_params["Y_TSV"], N_dies , N_pins, N_TSV, A_bond, cost_params["D0_HB"], manuf_vol)
            Y_stack = min(list(yield_breakdown["die"][chip_idx] for chip_idx in stack_ids[stack_idx]))
            #import pdb; pdb.set_trace()
            cost_breakdown["test"]["final_package"][stack_idx]= cost_ftl(cost_params["C_ate_ftl"], cost_params["C_probe_ftl"], cost_params["C_site_ftl"], Y_stack , cost_params["cover_die_ftl"], cost_params["cover_die_ws"], cost_params["T_per_insert_list_ftl"], cost_params["T_reattempt_list_ftl"], manuf_vol)

    A_sub =1
    A_interposer =0
    if len(stack_area)>1:
        A_interposer = sum(stack_area.values())*cost_params["si_scaling_factor"]
        cost_breakdown["interposer"]["Si0"], yield_breakdown["interposer"]["Si0"] = cost_interposer_silicon(cost_params["C_wafer"], A_interposer, 1, cost_params["r_wafer"], cost_params["D0_interposer_silicon"], cost_params["alpha_interposer_silicon"], manuf_vol)
        A_sub= cost_params["si_scaling_factor"]

    A_sub *= sum(stack_area.values())*cost_params["sub_scaling_factor"]
    N_sub = 1
    cost_breakdown["substrate"]["Sub0"] , yield_breakdown["substrate"]["Y_sub"] = cost_substrate(cost_params["k_substrate"], N_sub, cost_params["D0_sub"], A_sub, cost_params["alpha_sub"], manuf_vol)
    cost_breakdown["test"]["system_level"]["System0"] = cost_wafer_stl(cost_params["C_ate_stl"], cost_params["C_probe_stl"], cost_params["C_site_stl"], min(list(yield_breakdown["die"].values())) if len(yield_breakdown["die"].values())>0 else 0, cost_params["cover_die_ftl"], cost_params["T_per_insert_list_stl"], manuf_vol)
    cost_breakdown["nre"]["Packaging"]= cost_nre_pkg(A_interposer, cost_params["k_si_nre"], A_sub, cost_params["k_pkg_nre"], cost_params["C_pkg_nre_fixed"])
    cost_breakdown["nre"]["test"] = cost_params["C_test_nre_ftl"]+cost_params["C_test_nre_stl"]
    total_cost = sum(cost_breakdown["die"].values()) + sum(cost_breakdown["assembly"].values()) + sum(cost_breakdown["interposer"].values()) + sum(cost_breakdown["substrate"].values()) + sum(cost_breakdown["nre"].values()) + sum(cost_breakdown["test"]["wafer_probe"].values()) + sum(cost_breakdown["test"]["final_package"].values()) + sum(cost_breakdown["test"]["system_level"].values())
    print("----------Cost Summary---------------")
    print("Total Recurring Cost for ", manuf_vol, " units ($): ", sum(cost_breakdown["die"].values()) + sum(cost_breakdown["assembly"].values()) + sum(cost_breakdown["interposer"].values()) + sum(cost_breakdown["substrate"].values()) + sum(cost_breakdown["test"]["wafer_probe"].values()) + sum(cost_breakdown["test"]["final_package"].values()) + sum(cost_breakdown["test"]["system_level"].values()))
    print("Total NRE Cost ($): ", sum(cost_breakdown["nre"].values()))
    print("Total Cost per Die ($): ", total_cost/manuf_vol)
    print("----------------------------------------")
    if DEBUG:
        # plot bar chart of cost breakdown
        plt.figure(figsize=(10, 6))
        labels = list(cost_breakdown.keys())
        die_cost = sum(cost_breakdown["die"].values())
        assembly_cost = sum(cost_breakdown["assembly"].values())
        interposer_cost = sum(cost_breakdown["interposer"].values())
        substrate_cost = sum(cost_breakdown["substrate"].values())
        test_cost = sum([sum(cost_breakdown["test"][param].values()) for param in cost_breakdown["test"]])
        nre_cost = sum(cost_breakdown["nre"].values())
        costs = [die_cost, assembly_cost, interposer_cost, substrate_cost, test_cost, nre_cost]
        plt.bar(labels, costs)
        plt.ylabel('Cost ($)')
        plt.title('Cost Breakdown')
        plt.yscale('log')
        #plt.ylim(0, 3E6)
        for i, v in enumerate(costs):
            plt.text(i, v , f"${v:,.0f}", ha='center', va='bottom')
        os.makedirs(os.path.join(main_dir, "Results"), exist_ok=True)
        plt.savefig(os.path.join(main_dir, "Results", f"Cost_Breakdown_{aimodel}.png"))

        #plot bar chart of yield breakdown
        plt.figure(figsize=(10, 6))
        labels = list(yield_breakdown.keys())
        die_yield = min(list(yield_breakdown["die"].values())) if len(yield_breakdown["die"].values())>0 else 0
        assembly_yield = min(list(yield_breakdown["assembly"].values())) if len(yield_breakdown["assembly"].values())>0 else 0
        interposer_yield = min(list(yield_breakdown["interposer"].values())) if len(yield_breakdown["interposer"].values())>0 else 0
        substrate_yield = min(list(yield_breakdown["substrate"].values())) if len(yield_breakdown["substrate"].values())>0 else 0
        yields = [die_yield, assembly_yield, interposer_yield, substrate_yield]
        plt.bar(labels, yields)
        plt.ylabel('Yield')
        plt.title('Yield Breakdown')
        #plt.ylim(0, 1)
        for i, v in enumerate(yields):
            plt.text(i, v , f"{v:.10f}", ha='center', va='bottom')
        os.makedirs(os.path.join(main_dir, "Results"), exist_ok=True)
        plt.savefig(os.path.join(main_dir, "Results", f"Yield_Breakdown_{aimodel}.png"))
        #plt.show()
        
    #import pdb; pdb.set_trace()
    return total_cost, cost_breakdown