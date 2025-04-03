import logging

import pandas as pd

from sampling import scenario_sampling
from simulation import NetworkSim

logger = logging.getLogger()

# class Simulation:
#     def __init__(self, data):
#         self.data = data

#     # def __repr__(self):
#     #     return "Simulation Model"
    
def initialize_scenarios(self):
    """
    """
    if len(self.slist) == 0:
        self.slist = pd.Index(["nominal"])
    else:
        logger.info(f"Sampled scenarios found, so scenarios = {self.slist}")

    if len(self.demand) == 0:
        self.demand = self.nom.reset_index()
        self.demand["slist"] = "nominal"
        self.demand = self.demand.set_index(["products", "periods", "slist"])["nom"]
    logger.info(f"Demand table expanded for all scenarios.")

def update_items(self):
    """
    Updates FG list to drop FG with no demand, and components that are used to build updated FG.
    """
    forecast_FG = self.demand.index.get_level_values(0).unique()
    intermediate_components = set(self.Qty_Per.index.get_level_values(0)) & set(
        self.Qty_Per.index.get_level_values(1)
    )

    self.products = self.products[self.products.isin(forecast_FG)]
    bom_products = list(self.products) + list(intermediate_components)

    self.Qty_Per = self.Qty_Per[
        self.Qty_Per.index.get_level_values(1).isin(bom_products)
    ]
    self.components = self.components[
        self.components.isin(self.Qty_Per.index.get_level_values(0))
    ]
    self.items = self.products.union(self.components)

def simulate_main(self):

    hierarchy_df = pd.concat(
        [
            self.Platform[self.products],
            self.Family[self.products],
            self.Item[self.products],
        ],
        axis=1,
    )

    relation_df = pd.concat(
        [self.relation_from, self.relation_to, self.relation], axis=1
    ).rename(
        columns={
            "relation_from": "From",
            "relation_to": "To",
            "relation": "Relation",
        }
    )

    demand_df = (
        self.nom.reset_index()
        .rename(columns={"nom": "Demand"})
        .set_index(["Product", "Period"])
    )

    instructions_df = pd.concat(
        [
            self.Item_Sim,
            self.Uncertainty_Type,
            self.Distribution,
            self.Min,
            self.Max,
        ],
        axis=1,
    )

    ns = NetworkSim(hierarchy_df, demand_df, instructions_df, relation_df)
    #! TODO: From input
    self.max_simulations = 500
    self.auto_simulations = "No"
    self.NUMBER_OF_CLUSTERS = 20

    self.simulations = ns.simulate(
        self.max_simulations, auto_stop=self.auto_simulations
    )
    self.simulations, self.simulation_probability = ns.cluster_simulations(
        self.NUMBER_OF_CLUSTERS
    )
    self.Simulation_Status = 1

    sampled_demand, samples = scenario_sampling(
        self.simulations,
        self.simulation_probability,
        self.SELECTED_UNCERTAINTY_COVERAGE,
        self.SCENARIO_SELECTION_MODE,
    )
    self.slist = self.slist.append(pd.Index(samples))

    # update self.demand by concat nominal and sampled
    # make sure column orders before concat
    ordered_cols = ["Product", "Period", "Scenario", "Demand"]
    self.demand = self.nom.copy(deep = True)
    self.demand = pd.MultiIndex.from_product([self.demand.index.levels[0], self.demand.index.levels[1], ['nominal']],
                                        names=['Product', 'Period', 'Scenario'])
    self.demand = pd.Series(self.nom, index = self.demand)
    nominal = self.demand.reset_index().rename(
        columns={"nom": "Demand", "demand": "Demand"}
    )[ordered_cols]
    sampled = sampled_demand.reset_index()[ordered_cols]
    self.demand = pd.concat([nominal, sampled]).set_index(
        ["Product", "Period", "Scenario"]
    )["Demand"]

    # # peresist results on insight

    SIMULATION_OUTPUT = "Polaris_Simulation_Output.xlsx"
    sim_df = pd.DataFrame(
        {
            "Simulation Report": ["Simulation Status"],
            "Value": [self.Simulation_Status],
        }
    )
    slist = pd.DataFrame(self.slist, index=self.slist, columns=["slist"])
    with pd.ExcelWriter(SIMULATION_OUTPUT) as writer:
        sim_df.to_excel(writer, sheet_name="Summary", index=False)
        slist.to_excel(writer, sheet_name="Scenarios List", index=False)
        self.demand.reset_index().to_excel(
            writer, sheet_name="Demand", index=False
        )
        self.simulations.reset_index().to_excel(
            writer, sheet_name="Simulations", index=False
        )
        self.simulation_probability.reset_index().to_excel(
            writer, sheet_name="Simulation Probability", index=False
        )
    print("Finishing the Simulation")

    return sim_df, slist, self.demand, self.simulations, self.simulation_probability
