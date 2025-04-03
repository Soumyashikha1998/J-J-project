import logging
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition

from compute_inventory_investment import *
from optimization.mip_model import MIPModel
from optimization.opt_helper import (append_bounds, get_options,
                                     log_obj_values, save_bounds)

# gap when solving model with objective of minimizing missed demand (%)
GAP_SOLVE_MISSED_DEMAND = 1
logger = logging.getLogger()

class SCMSModel:
    def __init__(self, data):
        # default decimal number when rounding solution
        self.DECIMAL_NUMBER = 2
        self.data = data
        self.data.DECIMAL_NUMBER = self.DECIMAL_NUMBER
        self.epsilon = 1 / (10**self.DECIMAL_NUMBER)
        self.results = None
        self.objective = 0
        self.mip_model = MIPModel(data)
        self.instances = []

    def __repr__(self):
        return "SCMS Model"

    def pre_compute(self):
        self.demand = self.data.nom.copy(deep = True)
        self.demand = pd.MultiIndex.from_product([self.demand.index.levels[0], self.demand.index.levels[1], ['nominal']],
                                            names=['Product', 'Period', 'Scenario'])
        self.demand = pd.Series(self.data.nom, index = self.demand)
        self.data.kj_set = self.data.Qty_Per.index
        st_df = self.demand.reset_index()[["Period", "Scenario"]].drop_duplicates()
        st_df["Valid"] = 1
        self.data.st_set = st_df.set_index(["Period", "Scenario"]).Valid
    
    def solve(self, solver="cbc"):
        self.pre_compute()
        if self.data.forecast_considered == "All Sampled Scenarios":
            print(
                f"Running the model for totally {len(self.data.slist)} scenarios"
            )
            MinBounds, TargetBounds, MaxBounds, AvDemand, TargetReq, TargetDiff = (
                [] for i in range(6)
            )

            # 1.1 Build mip models for simulated scenarios
            for sce in self.data.slist:
                demand = self.data.demand[:, :, sce]
                # for testing time
                instance = self.mip_model.build(demand)
                self.instances.append((instance, sce))
            print(f"Built {len(self.data.slist)} scenarios")
            
            #! TODO: this info should be aggregated to the input file. In the app, is something that is collected in a UI
            self.data.number_of_first_stage_years = 0
            self.data.solve_two_step = True
            # 1.2 Solve each mip model indivisually
            if self.data.number_of_first_stage_years <= 0:
                for instance, sce in self.instances:
                    self.results = self.stable_solve(
                        instance,
                        stable_solve=self.data.solve_two_step,
                        solver=solver,
                        time=self.data.time_limit,
                        gap=self.data.optimality_gap / 100,
                    )
                    # Extract this as a function passing the logger and termination condition as parameters, and apply to the stochastic model in line 78 as well
                    if self.results["Solver"][0]["Termination condition"] == "optimal":
                        print(
                            f"Optimization done!  Objective value is {value(instance.obj)}"
                        )
                    else:
                        print("Model not solved to optimal!")
                    (
                        MinBounds,
                        TargetBounds,
                        MaxBounds,
                        AvDemand,
                        TargetReq,
                        TargetDiff,
                    ) = append_bounds(
                        self.mip_model,
                        MinBounds,
                        TargetBounds,
                        MaxBounds,
                        AvDemand,
                        TargetReq,
                        TargetDiff,
                        sce,
                    )

            # 1.3 Solve 2-stage stochastic model
            else:
                # 1.3.0 Initialize 2-stage stochastic model
                sm = pyo.ConcreteModel()
                probability = scale_scenario_probability(
                    self.data.simulations,
                    self.data.simulation_probability,
                    self.data.demand,
                    self.data.slist,
                )
                print(f"Sampled scenario probability scaled: {probability}.")

                # 1.3.1 include each individual model to stochastic model
                for instance, sce in self.instances:
                    setattr(sm, sce, instance)  # sm.sce = instance
                    instance.obj.deactivate()
                    instance.obj_attended_demand.deactivate()

                # 1.3.2 Add non-anticipativity constraints
                add_non_anticipativity_constraints(
                    sm,
                    self.instances,
                    self.data.number_of_first_stage_years,
                    periods_per_year=self.data.PERIODS_PER_YEAR,
                )

                # 1.3.3 Add stochastic model objective
                sm.obj = pyo.Objective(
                    expr=sum(probability[i[1]] * i[0].obj.expr for i in self.instances),
                    sense=pyo.maximize,
                )

                # 1.3.4 Solve stochastic model
                self.results = self.stable_solve(
                    sm,
                    stable_solve=False,
                    solver=solver,
                    time=self.data.time_limit,
                    gap=self.data.optimality_gap / 100,
                )

                # 1.3.5 Results
                if self.results["Solver"][0]["Termination condition"] == "optimal":
                    print(
                        f"2-stage Stochastic Optimization done!  Objective value is {value(instance.obj)}"
                    )
                else:
                    logger.error("Model not solved to optimal!")

                for instance, sce in self.instances:
                    (
                        MinBounds,
                        TargetBounds,
                        MaxBounds,
                        AvDemand,
                        TargetReq,
                        TargetDiff,
                    ) = append_bounds(
                        self.mip_model,
                        MinBounds,
                        TargetBounds,
                        MaxBounds,
                        AvDemand,
                        TargetReq,
                        TargetDiff,
                        sce,
                    )

            # 1.4 Save bounds
            save_bounds(
                self,
                MinBounds,
                TargetBounds,
                MaxBounds,
                AvDemand,
                TargetReq,
                TargetDiff,
            )
            return self.instances
        else:
            print("Solving nominal scenario...")
            MinBounds, TargetBounds, MaxBounds, AvDemand, TargetReq, TargetDiff = (
                [] for i in range(6)
            )
            instance = self.mip_model.build(self.data.nom)
            print("mip_model built...")

            #! TODO: this info should be aggregated to the input file. In the app, is something that is collected in a UI
            self.data.solve_two_step = True

            self.results = self.stable_solve(
                instance,
                stable_solve=self.data.solve_two_step,
                solver=solver,
                time=self.data.time_limit,
                gap=self.data.optimality_gap / 100,
            )
            if self.results["Solver"][0]["Termination condition"] == "optimal":
                print(
                    f"Optimization done! Objective value is {value(instance.obj)}"
                )
            else:
                print("Model not solved to optimal!")
            print('Appending bounds')
            self.instances = [(instance, "nominal")]
            (
                MinBounds,
                TargetBounds,
                MaxBounds,
                AvDemand,
                TargetReq,
                TargetDiff,
            ) = append_bounds(
                self.mip_model,
                MinBounds,
                TargetBounds,
                MaxBounds,
                AvDemand,
                TargetReq,
                TargetDiff,
                "nominal",
            )
            print('Saving bounds')
            save_bounds(
                self,
                MinBounds,
                TargetBounds,
                MaxBounds,
                AvDemand,
                TargetReq,
                TargetDiff,
            )
            return self.instances

    def stable_solve(
        self,
        instance,
        stable_solve=True,
        solver="cbc",
        time=300,
        gap=0.01,
        solver_log=True,
        stochastic=False,
    ):
        """
        Return stable solution:
        1. Run model of minimizing demand fulfillment
        2. Get solution of missed demand to fix d variables
        3. Run regular model with fixed d variables
        """
        # # Config Solver
        # regular config
        options_complete = get_options(solver=solver, time=time, gap=gap)
        mip = pyo.SolverFactory(solver, options=options_complete)
        self.mip = mip

        # minimizing missed demand config
        options_demand_attendance = get_options(
            solver=solver, time=time, gap=GAP_SOLVE_MISSED_DEMAND / 100
        )
        mip_missed_demand = pyo.SolverFactory(solver, options=options_demand_attendance)

        # if scenario_evaluation is activated, solve_two_step must be False
        #! TODO: this info should be aggregated to the input file. In the app, is something that is collected in a UI
        
        if self.data.evaluate_solutions:
            stable_solve = False

        # # Stable solve mode
        if stable_solve:
            # 1. Run model of minimizing demand fulfillment
            instance.obj.deactivate()
            instance.obj_attended_demand.activate()
            # solve model minimizing missed demand
            mip_missed_demand.solve(instance, tee=solver_log)
            print("Model with Demand Attendance Objective ON - Finished")

            # 2. Get solution of missed demand to fix d variables
            # fix d variable with the d solution of minimizing missing demand
            if not stochastic:
                self.fix_d_var(instance)
            else:
                for model, name in self.instances:
                    self.fix_d_var(model)
                    self.fix_dlt_var(model)

            # 3. Run regular model with fixed d variables
            instance.obj.activate()
            instance.obj_attended_demand.deactivate()
            # solve model
            results = mip.solve(instance, tee=solver_log)
            log_obj_values(instance, logger)
            print("Model with Cost-Benefit Efficiency ON - Finished")
        # Direct mode
        else:
            # final results
            results = mip.solve(instance, tee=solver_log)
            log_obj_values(instance, logger)
            print("Model with Cost-Benefit Efficiency ON - Finished")
            
        # Code to export the variables and parameters
        
        data_to_export = {}

        # Extract Variable Values
        for var_name, var_obj in instance.component_map(Var, active=True).items():
            var_data = {}
            for index in var_obj:
                var_data[index] = value(var_obj[index])
            data_to_export[var_name] = var_data

        # Extract Parameter Values
        for param_name, param_obj in instance.component_map(Param, active=True).items():
            param_data = {}
            for index in param_obj:
                param_data[index] = value(param_obj[index])
            data_to_export[param_name] = param_data

        # Create a Pandas DataFrame from the data
        df_data_to_export = pd.DataFrame.from_dict(data_to_export, orient="index")

        # Write DataFrame to Excel
        file_name = "variables_and_parameters.xlsx"
        with pd.ExcelWriter(file_name) as writer:
            for sheet_name in df_data_to_export.index.unique():
                df_data_to_export.loc[sheet_name].to_excel(writer, sheet_name=sheet_name)
        return results

    def fix_d_var(self, instance):
        for j, t in instance.d:
            if j in self.data.products:
                if instance.d[j, t].value >= self.epsilon:
                    instance.d[j, t].fix(
                        round(instance.d[j, t].value, self.DECIMAL_NUMBER)
                        - self.epsilon
                    )
                else:
                    instance.d[j, t].fix(0.0)

 
    def store_solution(self, app=None):
        print('Storing solution')
        # app link
        self.app = app
        # data should be attached to the app
        app = self.app if self.app else self
        self.data.output = {}
        columns_j = ["Item", "Scenario", "Solution"]
        columns_jt = ["Item", "Period", "Scenario", "Solution"]
        columns_it = ["Work Center", "Period", "Scenario", "Solution"]
        columns_ijt = ["Work Center", "Item", "Period", "Scenario", "Solution"]
        x_hat_name = self.mip_model.variable_names["x_hat"]
        z_hat_name = self.mip_model.variable_names["z_hat"]
        u_name = self.mip_model.variable_names["u"]
        v_name = self.mip_model.variable_names["v"]
        ad_name = self.mip_model.variable_names["ad"]
        wc_s_name = self.mip_model.variable_names["wc_s"]
        wc_e_name = self.mip_model.variable_names["wc_e"]
        wc_dev_s_name = self.mip_model.variable_names["wc_dev_s"]
        wc_dev_e_name = self.mip_model.variable_names["wc_dev_e"]
        for i in self.mip_model.variable_names.values():
            self.data.output[i] = {}
            if i in [x_hat_name, z_hat_name, u_name, v_name, ad_name]:
                columns_list = columns_ijt
            elif i in [wc_s_name, wc_e_name, wc_dev_s_name, wc_dev_e_name]:
                columns_list = columns_it
            else:
                columns_list = columns_jt
            for j in columns_list:
                self.data.output[i][j] = []
        # collect solutions by scenarios
        print(f'Collecting scenario solutions for {len(self.instances)} instances')
        for instance, sce in self.instances:
            for v in instance.component_objects(Var, active=True):
                varName = v.name.split(".")[-1]
                varObject = getattr(instance, varName)
                if self.mip_model.variable_names[varName] in [
                    x_hat_name,
                    z_hat_name,
                    u_name,
                    v_name,
                    ad_name,
                ]:
                    for workcenter, item, period in varObject:
                        self.data.output[self.mip_model.variable_names[varName]][
                            "Work Center"
                        ].append(workcenter)
                        self.data.output[self.mip_model.variable_names[varName]][
                            "Item"
                        ].append(item)
                        self.data.output[self.mip_model.variable_names[varName]][
                            "Period"
                        ].append(period)
                        self.data.output[self.mip_model.variable_names[varName]][
                            "Scenario"
                        ].append(sce)
                        self.data.output[self.mip_model.variable_names[varName]][
                            "Solution"
                        ].append(varObject[(workcenter, item, period)].value)
                elif self.mip_model.variable_names[varName] in [
                    wc_s_name,
                    wc_e_name,
                    wc_dev_s_name,
                    wc_dev_e_name,
                ]:
                    for workcenter, period in varObject:
                        self.data.output[self.mip_model.variable_names[varName]][
                            "Work Center"
                        ].append(workcenter)
                        self.data.output[self.mip_model.variable_names[varName]][
                            "Period"
                        ].append(period)
                        self.data.output[self.mip_model.variable_names[varName]][
                            "Scenario"
                        ].append(sce)
                        self.data.output[self.mip_model.variable_names[varName]][
                            "Solution"
                        ].append(varObject[(workcenter, period)].value)
                else:
                    for item, period in varObject:
                        self.data.output[self.mip_model.variable_names[varName]][
                            "Item"
                        ].append(item)
                        self.data.output[self.mip_model.variable_names[varName]][
                            "Period"
                        ].append(period)
                        self.data.output[self.mip_model.variable_names[varName]][
                            "Scenario"
                        ].append(sce)
                        self.data.output[self.mip_model.variable_names[varName]][
                            "Solution"
                        ].append(varObject[(item, period)].value)
        for itm in self.data.output:
            if itm in [x_hat_name, z_hat_name, u_name, v_name, ad_name]:
                columns_list = columns_ijt
            elif itm in [wc_s_name, wc_e_name, wc_dev_s_name, wc_dev_e_name]:
                columns_list = columns_it
            else:
                columns_list = columns_jt
            # initialize result
            result = pd.DataFrame(columns=columns_list)
            for col in columns_list:
                result[col] = self.data.output[itm][col]
            if itm == self.mip_model.variable_names["x"]:
                self.data.x_jts = (
                    result.sort_values(["Scenario", "Period", "Item"])
                    .set_index(["Item", "Period", "Scenario"])
                    .round(self.DECIMAL_NUMBER)
                )
            if itm == self.mip_model.variable_names["y"]:
                self.data.y_pts = (
                    result.rename(columns={"Item": "Project"})
                    .sort_values(["Scenario", "Period", "Project"])
                    .set_index(["Project", "Period", "Scenario"])
                    .round()
                    .astype(int)
                )
            if itm == self.mip_model.variable_names["y_s"]:
                self.data.y_s_pts = (
                    result.rename(columns={"Item": "Project"})
                    .sort_values(["Scenario", "Period", "Project"])
                    .set_index(["Project", "Period", "Scenario"])
                    .round()
                    .astype(int)
                )
            if itm == self.mip_model.variable_names["y_a"]:
                self.data.y_a_pts = (
                    result.rename(columns={"Item": "Project"})
                    .sort_values(["Scenario", "Period", "Project"])
                    .set_index(["Project", "Period", "Scenario"])
                    .round()
                    .astype(int)
                )
            if itm == self.mip_model.variable_names["y_d"]:
                self.data.y_d_pts = (
                    result.rename(columns={"Item": "Project"})
                    .sort_values(["Scenario", "Period", "Project"])
                    .set_index(["Project", "Period", "Scenario"])
                    .round()
                    .astype(int)
                )
            if itm == self.mip_model.variable_names["y_f"]:
                self.data.y_f_pts = (
                    result.rename(columns={"Item": "Project"})
                    .sort_values(["Scenario", "Period", "Project"])
                    .set_index(["Project", "Period", "Scenario"])
                    .round()
                    .astype(int)
                )
            if itm == self.mip_model.variable_names["s"]:
                self.data.s_jts = (
                    result.sort_values(["Scenario", "Period", "Item"])
                    .set_index(["Item", "Period", "Scenario"])
                    .round(self.DECIMAL_NUMBER)
                )
            if itm == self.mip_model.variable_names["z"]:
                self.data.z_its = (
                    result.rename(columns={"Item": "Work Center"})
                    .sort_values(["Scenario", "Period", "Work Center"])
                    .set_index(["Work Center", "Period", "Scenario"])
                    .round(self.DECIMAL_NUMBER)
                )
            if itm == self.mip_model.variable_names["d"]:
                self.data.d_jts = (
                    result.sort_values(["Scenario", "Period", "Item"])
                    .set_index(["Item", "Period", "Scenario"])
                    .round(self.DECIMAL_NUMBER)
                )
            if itm == self.mip_model.variable_names["dlt"]:
                self.data.dlt_jts = (
                    result.sort_values(["Scenario", "Period", "Item"])
                    .set_index(["Item", "Period", "Scenario"])
                    .round(self.DECIMAL_NUMBER)
                )
            if itm == self.mip_model.variable_names["s_e"]:
                self.data.s_e_jts = (
                    result.sort_values(["Scenario", "Period", "Item"])
                    .set_index(["Item", "Period", "Scenario"])
                    .round(self.DECIMAL_NUMBER)
                )
            if itm == self.mip_model.variable_names["s_s"]:
                self.data.s_s_jts = (
                    result.sort_values(["Scenario", "Period", "Item"])
                    .set_index(["Item", "Period", "Scenario"])
                    .round(self.DECIMAL_NUMBER)
                )
            if itm == self.mip_model.variable_names["s_bin"]:
                self.data.s_bin_jts = (
                    result.sort_values(["Scenario", "Period", "Item"])
                    .set_index(["Item", "Period", "Scenario"])
                    .round(self.DECIMAL_NUMBER)
                )
            if itm == self.mip_model.variable_names["s_l"]:
                self.data.s_l_jts = (
                    result.sort_values(["Scenario", "Period", "Item"])
                    .set_index(["Item", "Period", "Scenario"])
                    .round(self.DECIMAL_NUMBER)
                )
            if itm == self.mip_model.variable_names["s_u"]:
                self.data.s_u_jts = (
                    result.sort_values(["Scenario", "Period", "Item"])
                    .set_index(["Item", "Period", "Scenario"])
                    .round(self.DECIMAL_NUMBER)
                )
            if itm == self.mip_model.variable_names["x_hat"]:
                self.data.x_hat_ijts = (
                    result.sort_values(["Scenario", "Period", "Item", "Work Center"])
                    .set_index(["Work Center", "Item", "Period", "Scenario"])
                    .fillna(0)
                    .round(self.DECIMAL_NUMBER)
                )
            if itm == self.mip_model.variable_names["z_hat"]:
                self.data.z_hat_ijts = (
                    result.sort_values(["Scenario", "Period", "Item", "Work Center"])
                    .set_index(["Work Center", "Item", "Period", "Scenario"])
                    .fillna(0)
                    .round(self.DECIMAL_NUMBER)
                )
            if itm == self.mip_model.variable_names["u"]:
                self.data.u_ijts = (
                    result.sort_values(["Scenario", "Period", "Item", "Work Center"])
                    .set_index(["Work Center", "Item", "Period", "Scenario"])
                    .round()
                    .astype(bool)
                )
            if itm == self.mip_model.variable_names["v"]:
                self.data.v_ijts = (
                    result.sort_values(["Scenario", "Period", "Item", "Work Center"])
                    .set_index(["Work Center", "Item", "Period", "Scenario"])
                    .round()
                    .astype(bool)
                )
            if itm == self.mip_model.variable_names["ad"]:
                self.data.ad_ijts = (
                    result.sort_values(["Scenario", "Period", "Item", "Work Center"])
                    .set_index(["Work Center", "Item", "Period", "Scenario"])
                    .fillna(0)
                    .round(self.DECIMAL_NUMBER)
                )
            if itm == self.mip_model.variable_names["wc_s"]:
                self.data.wc_s_its = (
                    result.sort_values(["Scenario", "Period", "Work Center"])
                    .set_index(["Work Center", "Period", "Scenario"])
                    .round(self.DECIMAL_NUMBER)
                )
            if itm == self.mip_model.variable_names["wc_e"]:
                self.data.wc_e_its = (
                    result.sort_values(["Scenario", "Period", "Work Center"])
                    .set_index(["Work Center", "Period", "Scenario"])
                    .round(self.DECIMAL_NUMBER)
                )
            if itm == self.mip_model.variable_names["wc_dev_s"]:
                self.data.wc_dev_s_its = (
                    result.sort_values(["Scenario", "Period", "Work Center"])
                    .set_index(["Work Center", "Period", "Scenario"])
                    .round(self.DECIMAL_NUMBER)
                )
            if itm == self.mip_model.variable_names["wc_dev_e"]:
                self.data.wc_dev_e_its = (
                    result.sort_values(["Scenario", "Period", "Work Center"])
                    .set_index(["Work Center", "Period", "Scenario"])
                    .round(self.DECIMAL_NUMBER)
                )

        # # Precompute
        # Get propagation of variables shortage, excess and inventory investment
        all_periods = (
            pd.Series(self.data.periods).reset_index().rename(columns={0: "periods"})
        )
        dev_by_scenario = []
        inv_by_scenario = []

        if self.data.forecast_considered in ["All Sampled Scenarios"]:
            scen_list = self.data.slist
        else:
            scen_list = ["nominal"]
        print('Computing inventory_investment')
        self.data.inventory_investment = compute_inventory_investments(
            self.data.Target,
            self.data.x_jts,
            self.data.s_jts,
            self.data.s_e_jts,
            self.data.s_s_jts,
            scen_list = scen_list
            )
        print('Obtained inventory_investment')

        for sce in scen_list:
            (
                excess_prop_demand,
                excess_demand,
                shortage_prop_demand,
                shortage_demand,
                investment_prop_demand,
                investment_demand,
            ) = (defaultdict(float) for i in range(6))
            
            print('collecting propagated_demand')
            for j in self.data.items:
                for t in self.data.periods:
                    # Excess and Shortage deviation
                    if t != self.data.periods[-1]:
                        next_period = self.data.periods[self.data.period_numbers[t] + 1]
                        excess_value = self.data.s_e_jts.loc[j,t,sce].values[0]
                        shortage_value = self.data.s_s_jts.loc[j,t,sce].values[0]
                        investment_value = self.data.inventory_investment[(j, t, sce)]
                        self.mip_model.get_multilayer_prop_demand(
                            j,
                            next_period,
                            excess_value,
                            excess_prop_demand,
                            excess_demand,
                            "deviation",
                        )
                        self.mip_model.get_multilayer_prop_demand(
                            j,
                            next_period,
                            shortage_value,
                            shortage_prop_demand,
                            shortage_demand,
                            "deviation",
                        )
                        # Inventory investments
                        self.mip_model.get_multilayer_prop_demand(
                            j,
                            t,
                            investment_value,
                            investment_prop_demand,
                            investment_demand,
                            "deviation",
                        )
            print(f"  done collecting PD: total CALLS={self.mip_model.calls_to_multilayer_prop_demand}")
            excess_prop_demand = self.mip_model._format_prop_demand(
                all_periods, excess_prop_demand
            )
            shortage_prop_demand = self.mip_model._format_prop_demand(
                all_periods, shortage_prop_demand
            )
            investment_prop_demand = self.mip_model._format_prop_demand(
                all_periods, investment_prop_demand
            )
            deviation_sum = shortage_prop_demand - excess_prop_demand
            dev_by_scenario.append(pd.concat([deviation_sum], keys=[sce]))
            inv_by_scenario.append(pd.concat([investment_prop_demand], keys=[sce]))

        deviation_propagation = pd.concat(dev_by_scenario)
        deviation_propagation.index = deviation_propagation.index.set_names(
            ["Scenario", "Item", "Period"]
        ).reorder_levels(["Item", "Period", "Scenario"])
        inventory_inv_propagation = pd.concat(inv_by_scenario)
        inventory_inv_propagation.index = inventory_inv_propagation.index.set_names(
            ["Scenario", "Item", "Period"]
        ).reorder_levels(["Item", "Period", "Scenario"])

        deviation_in_period = pd.Series(dtype="float64")
        print('collecting period deviation')
        for j in self.data.s_e_jts.index.get_level_values(0).unique():
            for scen in scen_list:
                excess_aux = self.data.s_e_jts[
                    (self.data.s_e_jts.index.get_level_values(0) == j)
                    & (self.data.s_e_jts.index.get_level_values(2) == scen)
                ]
                shortage_aux = self.data.s_s_jts[
                    (self.data.s_s_jts.index.get_level_values(0) == j)
                    & (self.data.s_e_jts.index.get_level_values(2) == scen)
                ]
                deviation_aux = shortage_aux - excess_aux
                deviation_aux = deviation_aux.shift(1).fillna(0.0)
                deviation_in_period = pd.concat(
                    [deviation_in_period, deviation_aux["Solution"]]
                )

        deviation_in_period.index = pd.MultiIndex.from_tuples(
            deviation_in_period.index, names=["Item", "Period", "Scenario"]
        )

        # add demand as a new column
        if self.data.forecast_considered in ["All Sampled Scenarios"]:
            # get the simulated demand as a dataframe
            demand = self.data.demand.round(self.DECIMAL_NUMBER).reset_index()
        else:
            # in nominal mode, use nominal demand input and as a scenario column
            demand = self.data.nom.round(self.DECIMAL_NUMBER).reset_index()
            demand["Scenario"] = "nominal"
            demand.columns = ["Product", "Period", "Demand", "Scenario"]
            demand = demand[["Product", "Period", "Scenario", "Demand"]]
        demand.columns = ["Item", "Period", "Scenario", "Demand"]

        prop_demand = pd.Series(dtype="float64")
        print('collecting prop_demand')
        for scen in scen_list:
            demand_tmp = (
                demand[demand["Scenario"] == scen]
                .drop("Scenario", axis=1)
                .rename(
                    columns={
                        "Item": "Product",
                    }
                )
                .set_index(["Product", "Period"])
                .Demand
            )

            prop_ml_demand, ml_demand = self.mip_model.get_prop_demand(demand_tmp)

            prop_demand_tmp = pd.concat(
                [prop_ml_demand], keys=[scen], names=["Scenario"]
            )
            prop_demand = pd.concat([prop_demand, prop_demand_tmp])
        index_prop_demand = prop_demand.index
        index_prop_demand = pd.MultiIndex.from_tuples(
            index_prop_demand, names=["Scenario", "Item", "Period"]
        )
        prop_demand.index = index_prop_demand

        # Populates propagated demand entity (to be used in Scenario Evaluation)
        self.data.propagated_demand = prop_demand
        self.data.demand_report = demand
        self.data.deviation_propagation = deviation_propagation
        self.data.inventory_inv_propagation = inventory_inv_propagation
        self.data.deviation_in_period = deviation_in_period
        self.data.prop_demand = prop_demand

        if self.data.evaluate_solutions:
            self.data.evaluations = self.evaluations.set_index(["Scenario", "Simulation"])
            self.data.esf = self.esf.set_index(["Scenario"])
            self.data.evaluations_tableau = self.evaluations_tableau
            self.data.esf_tableau = self.esf_tableau

        print("Output tables done!")

    def export_excel(
        self,
        app=None,
        timestamp=True,
        app_name="MyApp",
        include_inputs=False,
        color={"options": "orange", "input": "green", "output": "blue"},
    ):
        self.app = app
        # data should be attached to the app
        app = self.app if self.app else self
        columns_list = ["Item", "Period", "Scenario", "Solution"]
        filename = "optimization_output"
        if app_name:
            filename = f"{app_name}_{filename}"
        if timestamp:
            now = time.strftime("%Y%m%d%H%M%S")
            filename = f"{filename}_{now}"
        filename = f"{filename}.xlsx"
        # Write output to excel
        print(f"Writing output to file {filename}...")
        writer = pd.ExcelWriter(filename, engine="xlsxwriter")
        for itm in self.data.output:
            result = pd.DataFrame(columns=columns_list)
            for col in columns_list:
                result[col] = self.data.output[itm][col]
            result = result.sort_values(["Scenario", "Period", "Item"])
            result.to_excel(writer, sheet_name=itm, index=False)
        writer.save()


def convert(seconds):
    if seconds < 60:
        return "%02ds" % (seconds)
    elif seconds < 3600:
        minutes, sec = divmod(seconds, 60)
        return "%02dm %02ds" % (minutes, sec)
    hour, minutes = divmod(minutes, 60)
    return "%dh %02dm %02ds " % (hour, minutes, sec)


if __name__ == "__main__":
    filename = "EES_SCMS_InputTemplate_v0.2.xlsx"
    data = SCMSdata(filename)
    data.load_simulation_input()
    data.load_optimization_input()
    m = SCMSModel(data)
    m.solve()
    m.export_excel()



#### Error message: SCMSdata is not defined in line 816 ####