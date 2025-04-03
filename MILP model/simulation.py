import collections
import logging
import math

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.traversal.depth_first_search import dfs_tree
from sklearn_extra.cluster import KMedoids

from simhelper import (bass_scale, enough_simulations, get_random_number,
                       shift_demand, sort_instructions)

logging.basicConfig(level=logging.DEBUG)


class NetworkSim(nx.DiGraph):
    def __init__(self, hierarchy, demand, instructions=None, relations=None):
        """
        NetworkSim constructor
        """
        super().__init__(self)

        instructions = sort_instructions(instructions)

        # Create nodes and edges from hierachy
        for j in range(hierarchy.shape[1] - 1):
            self.add_edges_from(
                list(zip(hierarchy.iloc[:, j], hierarchy.iloc[:, j + 1]))
            )

        # Identify and save leaf nodes
        self.leaf_nodes = set(n for n in self.nodes if self.out_degree(n) == 0)
        # Identify and save root nodes
        self.root_nodes = set(n for n in self.nodes if self.in_degree(n) == 0)

        # Save time periods
        self.periods = list(demand.reset_index()["Period"].unique())

        # Save Items
        self.items = hierarchy.iloc[:, -1].unique()

        # Save simulation instructions
        self.simulation_instructions = instructions

        # Save nodes relationships
        self.relations = relations

        # NOTE: It could lead to errors if the names of the attributed is repeated with different sections of the network
        # TODO: Create a single parent node containing the Total

        # Save demand on leaf nodes
        for item in self.items:
            self.nodes[item]["Demand"] = np.array(
                demand.loc[item, :]["Demand"], dtype=float
            )

        # Set initial fraction for root nodes
        for item in self.root_nodes:
            self.nodes[item]["Fraction"] = np.ones(len(self.periods))

        for node in self.nodes:
            self.nodes[node]["Relations"] = []

        for node in self.nodes:
            self.nodes[node]["Relations_type"] = []

        for idx, row in relations.iterrows():
            self.nodes[row["Item_From"]]["Relations"].append(row["Item_To"])

        for idx, row in relations.iterrows():
            self.nodes[row["Item_From"]]["Relations_type"].append(row["Relation"])

        # Propagate demand upstream
        self.propagate_upstream(self.leaf_nodes)

        # Identify Uncertain Nodes
        self.uncertain_nodes = set()
        for item in set(self.simulation_instructions["Item"]):
            G = dfs_tree(self, item)
            leaf_nodes = set([x for x in G.nodes() if G.out_degree(x) == 0])
            self.uncertain_nodes = self.uncertain_nodes.union(leaf_nodes)
        logging.debug(f"Uncertain Nodes: {self.uncertain_nodes}")

        # Save initial demand and fractions
        self.initial_demand = {
            node: self.nodes[node]["Demand"].copy() for node in self.nodes
        }
        self.initial_fraction = {
            node: self.nodes[node]["Fraction"].copy() for node in self.nodes
        }

    def propagate_upstream(self, starting_nodes):
        """
        Given a change in a particular node in the network, this function propagates the changes upstream
        """
        # If called with single node convert in list
        starting_nodes = (
            [starting_nodes] if type(starting_nodes) == str else starting_nodes
        )
        # Identify parents of those leaf nodes
        parents_list = [self.predecessors(n) for n in starting_nodes]
        parents = set(item for sublist in parents_list for item in sublist)
        if len(parents) == 0:
            logging.debug(f"Reached the parent node. No more upstream propagation")
            return
        else:
            logging.debug(f"Propagating upstream for {parents}")
        # Propagate demand
        for p in parents:
            self.nodes[p]["Demand"] = sum(
                self.nodes[e[1]]["Demand"] for e in self.out_edges(p)
            )
            # Update fractions
            for e in self.out_edges(p):
                child = self.nodes[e[1]]
                child["Fraction"] = child["Demand"] / self.nodes[p]["Demand"]
                child["Fraction"][np.isnan(child["Fraction"])] = 0
        # Move one level up
        self.propagate_upstream(parents)

    def propagate_downstream(self, starting_node, shift=0):
        logging.debug(f"Starting downstream propagation from {starting_node}")
        if self.out_degree(starting_node) == 0:
            return
        else:
            for e in self.out_edges(starting_node):
                child_name = e[1]
                child = self.nodes[child_name]
                child["Fraction"] = shift_demand(child["Fraction"], shift)
                child["Demand"] = (
                    child["Fraction"] * self.nodes[starting_node]["Demand"]
                )
                self.propagate_downstream(child_name, shift)

    def propagate_parallel(self, starting_node, od, shift=0):
        logging.debug(f"Starting parallel propagation from {starting_node}")

        inverse, direct = self.direct_inverse_lists(starting_node)

        if len(inverse) != 0:
            delta, factor_prop = self.propagation_factor(
                starting_node, od, relation=inverse
            )
            self.inverse_method(starting_node, inverse, delta, factor_prop, shift)

        if len(direct) != 0:
            delta, factor_prop = self.propagation_factor(
                starting_node, od, relation=direct
            )
            self.direct_method(starting_node, direct, delta, factor_prop, shift)

    def direct_inverse_lists(self, starting_node):
        relation = self.nodes[starting_node]["Relations"]
        relation_type = self.nodes[starting_node]["Relations_type"]

        inverse = [
            relation[idx]
            for idx in range(len(relation))
            if relation_type[idx].upper() == "INVERSE"
        ]
        direct = [
            relation[idx]
            for idx in range(len(relation))
            if relation_type[idx].upper() == "DIRECT"
        ]

        return inverse, direct

    def direct_method(self, starting_node, relation, delta, factor_prop, shift):
        for r in relation:
            self.nodes[r]["Demand"] = self.nodes[r]["Demand"] + (factor_prop[r] * delta)

        self.propagate_upstream_downstream(starting_node, shift)

    def propagate_upstream_downstream(self, starting_node, shift):
        # Obtain demand of the parent node
        predecessor = [pred for pred in self.predecessors(starting_node)]

        # For each node in relation and the current node, update fractions
        if len(predecessor) == 0:
            # There is no parent node at this point
            relation = self.nodes[starting_node]["Relations"]
            for rel in relation:
                self.propagate_downstream(rel)
            self.propagate_downstream(starting_node)
        else:
            successors = [suc for suc in self.successors(predecessor[0])]

            # Total demand of predecessor
            sum_successors = sum(self.nodes[i]["Demand"] for i in successors)

            # self.nodes[predecessor[0]]["Demand"] = sum_successors

            # Update fractions for each node in relation
            for s in successors:
                self.nodes[s]["Fraction"] = np.nan_to_num(
                    self.nodes[s]["Demand"] / sum_successors
                )
                self.propagate_downstream(s, shift=shift)

            self.propagate_upstream(starting_node)

    def inverse_method(self, starting_node, relation, delta, factor_prop, shift):
        for r in relation:
            self.nodes[r]["Demand"] = self.nodes[r]["Demand"] - (factor_prop[r] * delta)

            if any(n < 0 for n in self.nodes[r]["Demand"]):
                demand_node = self.nodes[r]["Demand"]
                demand_node[demand_node < 0] = 0
                self.propagate_upstream_downstream(starting_node, shift)

        # Obtain demand of the parent node
        predecessor = [pred for pred in self.predecessors(starting_node)]

        if len(predecessor) == 0:
            # There is no parent node at this point
            relation = self.nodes[starting_node]["Relations"]
            for rel in relation:
                self.propagate_downstream(rel)
            self.propagate_downstream(starting_node)
        else:
            parent_demand = self.nodes[predecessor[0]]["Demand"]

            # For each node in relation and the current node, update fractions
            successors = [suc for suc in self.successors(predecessor[0])]

            # Update fractions for each node in relation
            for s in successors:
                self.nodes[s]["Fraction"] = np.nan_to_num(
                    self.nodes[s]["Demand"] / parent_demand
                )
                self.propagate_downstream(s, shift=shift)

    def propagation_factor(self, starting_node, od, relation):
        # Obtain simulated demand of starting node
        current_demand = self.nodes[starting_node]["Demand"]

        # Create a list named delta with the difference between the simulated demand and the original demand of the starting node
        delta = current_demand - od

        # Sum up all the demands of the nodes in relation
        sum_dem_relation_nodes = sum(self.nodes[i]["Demand"] for i in relation)

        # Create a dictionary with the division of Demand by sum_dem_relation_nodes for each node in relation
        factor_prop = {
            rel: self.nodes[rel]["Demand"] / sum_dem_relation_nodes for rel in relation
        }
        #
        for rel in factor_prop:
            factor_prop[rel] = np.nan_to_num(factor_prop[rel])

        return delta, factor_prop

    def propagate(self, starting_node, uncertainty_type, od, shift=0):
        if uncertainty_type.upper() == "DELAY" or uncertainty_type.upper() == "LAUNCH":
            self.propagate_upstream(starting_node)
            self.propagate_downstream(starting_node, shift=shift)
        else:
            # Checking if starting_node has a relation.
            if len(self.nodes[starting_node]["Relations"]) > 0:
                self.propagate_parallel(starting_node, od)
            # If starting_node has no relation, propagate upstream and downstream
            else:
                self.propagate_upstream(starting_node)
                self.propagate_downstream(starting_node, shift=shift)

    def process_instruction(self, item, uncertainty_type, distribution, umin, umax):
        od = self.nodes[item]["Demand"]
        shift = 0
        random_number = 0
        # If the sum of item's demand is not 0, then self.propagate old demand
        if sum(od) != 0:
            # if uncertainty_type is different from delay, then get a random number
            if uncertainty_type.upper() != "DELAY":
                random_number = get_random_number(
                    distribution, umin / 100 + 1, umax / 100 + 1
                )
            if uncertainty_type.upper() == "MARKET POTENTIAL":
                self.scale_node_demand(item, mf=random_number)
            elif uncertainty_type.upper() == "ADOPTION RATE":
                self.scale_node_demand(item, qf=random_number)
            # if uncertainty_type is launch, then compare the probability of launching in every simulation
            elif uncertainty_type.upper() == "LAUNCH":
                launch_probability = get_random_number(
                    distribution, umin / 100, umax / 100
                )
                toss = np.random.random()

                logging.debug(
                    f"Comparing Launch Probability: {launch_probability} with Toss: {toss}"
                )
                # if toss is greater than launch probability, then item's demand is 0
                random_number = 1 * (toss < launch_probability)
                if toss > launch_probability:
                    self.nodes[item]["Demand"] *= 0
                logging.debug(
                    f'Item {item} total demand: {sum(self.nodes[item]["Demand"])}'
                )
            elif uncertainty_type.upper() == "DELAY":
                delay_probability = umin / 100
                toss = np.random.random()

                logging.debug(
                    f"Comparing Delay Probability: {delay_probability} with Toss: {toss}"
                )
                # if toss is less than delay probability, then item's demand is shifted for the number of periods
                # stated in max column
                random_number = 1 * (toss < delay_probability)
                if toss < delay_probability:
                    shift = int(umax)
                    self.nodes[item]["Demand"] = shift_demand(
                        self.nodes[item]["Demand"], shift
                    )
            else:
                # error message if uncertainty_type is not valid
                raise ValueError(f"Unrecognized uncertainty type {uncertainty_type}")

            # Propagate the new demand
            self.propagate(item, uncertainty_type, od, shift)

        # If the sum of item's demand is 0, then self.propagate old demand
        else:
            # Propagate the new demand
            self.propagate(item, uncertainty_type, od, shift)

        return random_number

    def scale_node_demand(self, item, mf=1, qf=1):
        """
        Scale the demand of a single node
        """
        demand = self.nodes[item]["Demand"].copy()
        self.nodes[item]["Demand"] = bass_scale(demand, mf, qf)

    def get_demand(self):
        """
        Extract the demand from the leaf nodes to generate a Pandas Series
        """
        cols = ["Product"] + self.periods
        ls = [
            [node] + [float(n) for n in self.nodes[node]["Demand"]]
            for node in self.leaf_nodes
        ]
        df = pd.DataFrame(ls, columns=cols)
        df = df.melt("Product", var_name="Period", value_name="Demand")
        return df.set_index(["Product", "Period"])["Demand"]

    def simulate(self, max_simulations, auto_stop=False, step=20, threshold=0.01):
        # Initialize the simulation
        # Create a dataframe to store the results
        df = pd.DataFrame()
        random_numbers = pd.DataFrame(
            columns=["Simulation", "Instruction", "Random_Number"]
        )
        # Execute the simulation with the number of simulations between 1 and max_simulations
        for sim in range(1, max_simulations + 1):
            logging.debug(f"Running Simulation {sim}")
            # For each simulation instruction, execute process_instruction function
            for idx, row in self.simulation_instructions.iterrows():
                random_number = self.process_instruction(
                    row["Item"],
                    row["Uncertainty_Type"],
                    row["Distribution"],
                    row["Min"],
                    row["Max"],
                )
                random_numbers.loc[len(random_numbers.index)] = [
                    int(sim),
                    idx,
                    random_number,
                ]
            sim_demand = self.get_demand().reset_index()
            sim_demand["Simulation"] = sim
            df = pd.concat([df, sim_demand])

            # Restitue demand to the initial value
            for node in self.initial_demand:
                self.nodes[node]["Demand"] = self.initial_demand[node].copy()
                self.nodes[node]["Fraction"] = self.initial_fraction[node].copy()
            # If auto_stop is True, then stop the simulation if the change in demand is less than threshold
            if auto_stop and math.remainder(sim, step) == 0:
                mid_period = self.periods[int(0.5 * len(self.periods))]
                if enough_simulations(
                    df[
                        (df.Period == mid_period)
                        & (df.Product.isin(self.uncertain_nodes))
                    ],
                    threshold,
                ):
                    break
        # Return the dataframe with the results of the simulation
        self.simulation = df.set_index(["Product", "Period", "Simulation"])["Demand"]
        self.random_numbers = random_numbers.pivot(
            index="Simulation", columns="Instruction", values="Random_Number"
        )
        return self.simulation, self.random_numbers

    def cluster_simulations(self, num_clusters=20):
        # apply K-Medoids clustering with 20 clusters
        kmedoids = KMedoids(n_clusters=num_clusters, random_state=0).fit(
            self.random_numbers
        )

        simulations = self.simulation.reset_index()
        clusters = pd.DataFrame(kmedoids.labels_, columns=["Cluster"]).reset_index()
        clusters.rename(columns={"index": "Simulation"}, inplace=True)
        clusters["Simulation"] += 1
        clusters["Probability"] = clusters.Cluster.map(
            collections.Counter(kmedoids.labels_)
        ) / len(clusters.index)
        simulations = simulations.merge(clusters, how="right")

        filtered_simulations = simulations[
            simulations["Simulation"].isin(kmedoids.medoid_indices_ + 1)
        ]

        simulation_probabilities = simulations.groupby('Simulation').max()['Probability']
        filtered_simulations = filtered_simulations.set_index(['Product', 'Period', 'Simulation'])['Demand']
    
        return filtered_simulations, simulation_probabilities

