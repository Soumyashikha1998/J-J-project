import logging
import math
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.core.base.constraint import Constraint
from pyomo.environ import value

APP_NAME = "Polaris"
logger = logging.getLogger(APP_NAME)
logger.setLevel(logging.INFO)
logger.propagate = False


class MIPModel(object):
    def __init__(self, data):
        """initialize self object with data object"""
        self.d = data
        self.variable_names = {
            "x": "Actual Production",
            "x_hat": "Actual Detailed Production",
            "y": "Products Qualified Capacity Investment",
            "y_s": "Planned Products Capacity Investment",
            "y_a": "Cumulative Capital Investment",
            "y_d": "Cumulative Depreciated Investments",
            "y_f": "Depreciation Instances",
            "u": "Planned Qualification Project",
            "v": "Qualified Qualification Project",
            "s": "Inventory Level",
            "s_l": "Inventory Lower Bound Violation",
            "s_u": "Inventory Upper Bound Violation",
            "s_e": "Excess Deviation",
            "s_s": "Shortage Deviation",
            "s_bin": "Shortage Deviation Indicator",
            "z": "Total Capacity",
            "z_hat": "Detailed Capacity",
            "d": "Items Demand",
            "ad": "Allocation Deviation",
            "wc_s": "Utilization Shortage",
            "wc_e": "Utilization Excess",
            "wc_dev_s": "Utilization Deviation Shortage",
            "wc_dev_e": "Utilization Deviation Excess",
        }
        self.calls_to_multilayer_prop_demand = 0
        
    # AHTY-350 get inventory bounds
    def get_multilayer_prop_demand(self,j,t,demand,prop_ml_demand,ml_demand,demand_or_target="demand",shift_periods_parent=0,level=0):
        self.calls_to_multilayer_prop_demand +=1
        if level == 0:
            if demand_or_target == "target" or demand_or_target == "deviation":
                shift_periods_parent = 0
            else:
                shift_periods_parent = -self.d.Component_Lead_Time.get(j, 0)
            if demand_or_target != "deviation":
                prop_ml_demand[(j, level, t, shift_periods_parent)] = demand
                ml_demand[(j, level, t, 0)] = demand
            level = 1
        components = self.component_demand[j]
        Scrap_Rate = self.d.Scrap_Rate
        for index, row in components.iterrows():
            k = row.Component
            Qty = row.Qty_Per
            shift_periods = -self.d.Component_Lead_Time[k] + shift_periods_parent
            period_number = self.d.period_numbers[t] + shift_periods
            if period_number >= 0:
                t_pre = self.d.periods[period_number]
            else:
                t_pre = 0
            sourcing = self.d.Sourcing.get((k, t_pre), 1.0)

            Qds = Qty * demand * sourcing / (1 - Scrap_Rate[(k, j)])
            prop_ml_demand[(k, level, t, shift_periods)] = prop_ml_demand.get((k, level, t, shift_periods), 0) + Qds
            ml_demand[(k, level, t, shift_periods_parent)] = ml_demand.get((k, level, t, shift_periods_parent), 0) + Qds
            self.get_multilayer_prop_demand(k,t,Qds,
                                            prop_ml_demand, ml_demand, demand_or_target, shift_periods, level + 1)
        return prop_ml_demand, ml_demand

    def get_average_demand(self, j, t, next_t, demand):
        period_numbers = pd.Series(
            index=self.d.periods, data=range(len(self.d.periods))
        )

        t_number = period_numbers[t]
        while t_number + next_t >= len(self.d.periods):
            if next_t == 1:
                break
            else:
                next_t -= 1

        sum_demand = 0
        # If not last period, we look for following demands
        if t != self.d.periods[-1]:
            start = period_numbers[t] + 1
            end = period_numbers[t] + 1 + next_t
        # if last period
        else:
            start = period_numbers[t]
            end = period_numbers[t] + next_t

        for t_aux in range(start, end):
            sum_demand += demand.get((j, self.d.periods[t_aux]), 0)

        if next_t == 0:
            return 0
        else:
            return sum_demand / next_t

    def _format_prop_demand(self, all_periods, ml_dict_demand):
        ml_dict_demand = pd.Series(ml_dict_demand)
        ml_dict_demand.index.names = ["Item", "level", "Period", "shifted_periods"]
        ml_dict_demand = ml_dict_demand.reset_index()
        ml_dict_demand.rename(columns={0: "Prop_demand"}, inplace=True)
        ml_dict_demand["Final Period"] = (
            ml_dict_demand["Period"].apply(lambda x: self.d.period_numbers[x])
            + ml_dict_demand["shifted_periods"]
        )
        ml_dict_demand = ml_dict_demand[ml_dict_demand["Final Period"] >= 0]
        ml_dict_demand["Final Period"] = ml_dict_demand["Final Period"].apply(
            lambda x: self.d.periods[x]
        )
        ml_dict_demand = ml_dict_demand[["Item", "Final Period", "Prop_demand"]]
        ml_dict_demand = ml_dict_demand.pivot_table(
            values="Prop_demand", columns="Item", index="Final Period", aggfunc="sum"
        )

        ml_dict_demand = ml_dict_demand.merge(
            all_periods.drop(columns=["index"]),
            how="right",
            left_on="Final Period",
            right_on="periods",
        )
        ml_dict_demand = ml_dict_demand.fillna(method="ffill")
        ml_dict_demand = ml_dict_demand.melt(
            id_vars=["periods"], var_name="Item", value_name="Prop_demand"
        ).rename(columns={"periods": "Period"})
        ml_dict_demand = ml_dict_demand.set_index(["Item", "Period"])["Prop_demand"]
        return ml_dict_demand

    def get_prop_demand(self, demand):
        prop_ml_demand = {}
        ml_demand = {}
        
        all_periods = pd.Series(self.d.periods).reset_index()
        col_name = all_periods.columns[1]
        all_periods = all_periods.rename(columns={col_name: "periods"})

        # precomputing components
        logger.info('Precomputing component_demand')
        self.component_demand = {j: self.d.Qty_Per[self.d.Qty_Per.index.get_level_values(1)==j].reset_index()
                        for j in set(self.d.products)}
        for j in self.d.items:
            self.component_demand[j] = self.d.Qty_Per[self.d.Qty_Per.index.get_level_values(1)==j].reset_index()
        logger.info(f'Precomputed {len(self.component_demand)} component_demand')
        for j in self.d.products:
            for t in self.d.periods:
                demand_value = demand.loc[j,t]
                self.get_multilayer_prop_demand(j, t, demand_value, prop_ml_demand, ml_demand)
            LT_FG = self.d.Component_Lead_Time.get(j, 0)
            for tt in range(1, LT_FG + 1):
                self.get_multilayer_prop_demand(j,self.d.periods[-tt],
                    demand_value,prop_ml_demand,ml_demand,"deviation")
        logger.info(f'Past double get_multilayer_prop_demand loop, {self.calls_to_multilayer_prop_demand} calls')
        prop_ml_demand = self._format_prop_demand(all_periods, prop_ml_demand)
        ml_demand = self._format_prop_demand(all_periods, ml_demand)
        return prop_ml_demand, ml_demand

    # AHTY-350 get inventory bounds
    def get_inventory_bounds(self, demand):
        # gets propagated demand objects from function
        logger.info("Getting propagated demand")
        prop_ml_demand, ml_demand = self.get_prop_demand(demand)
        logger.info("Got propagated demand")

        period_days = 360 / self.d.PERIODS_PER_YEAR

        minbound = {}
        maxbound = {}
        target = {}
        av_demand = {}

        for j in self.d.items:
            LT_j = self.d.Production_LT_Days[j]
            SS_j = self.d.SS_Days[j]
            MaxInv_j = self.d.Max_Inv_DOS[j]

            t_hat = math.ceil((LT_j + SS_j) / period_days)
            t_dot = math.ceil(MaxInv_j / period_days)

            for t in self.d.periods:
                AvDemand_hat = self.get_average_demand(j, t, t_hat, ml_demand)
                AvDemand_dot = self.get_average_demand(j, t, t_dot, ml_demand)

                minbound[(j, t)] = LT_j * AvDemand_hat / period_days
                target[(j, t)] = (LT_j + SS_j) * AvDemand_hat / period_days
                maxbound[(j, t)] = MaxInv_j * AvDemand_dot / period_days
                av_demand[(j, t)] = AvDemand_hat

                # AHTY-678: Adjust Max Inventory Bound in case Target is greater
                maxbound[(j, t)] = max(maxbound[(j, t)], target[(j, t)])

                if prop_ml_demand.get((j, t), 0) == 0:
                    minbound[(j, t)] = 0
                    target[(j, t)] = 0
                    maxbound[(j, t)] = 0

        self.d.prop_ml_demand = prop_ml_demand
        self.d.ml_demand = ml_demand
        self.d.minbound = pd.Series(minbound)
        self.d.target = pd.Series(target)
        self.d.maxbound = pd.Series(maxbound)
        self.d.avdemand = pd.Series(av_demand)

        demand = (self.d.prop_ml_demand, self.d.ml_demand, self.d.minbound, self.d.target, self.d.maxbound, self.d.avdemand)

    def get_last_demand(self, demand):
        last_demand = demand.reset_index().groupby(["Product"]).last()[demand.name]
        return last_demand

    def initial_inventory_level(self):
        self.d.adjust_initial_inventory = 0
        self.d.S0_original = self.d.S0.copy()

    def get_target_req(self):
        self.d.target.index.names = ["Item", "Period"]
        target_diff = self.d.target.groupby(level=0).diff()
        target_t0 = self.d.target[
            self.d.target.index.get_level_values(1) == self.d.periods[0]
        ]
        target_diff.fillna(target_t0, inplace=True)
        first_target = self.d.target[self.d.target > 0].reset_index()
        first_target["num_period"] = [
            self.d.period_numbers[x] for x in first_target["Period"]
        ]
        first_target = first_target.set_index(["Item", "Period"])
        first_target = first_target["num_period"]
        first_target = first_target.groupby(level=0).idxmin().apply(lambda x: x[1])

        for i in first_target.index:
            value = first_target[i]
            target_diff[(i, value)] -= self.d.S0[i]

        target_prop_demand = defaultdict(float)
        target_demand = defaultdict(float)
        all_periods = pd.Series(self.d.periods).reset_index()
        col_name = all_periods.columns[1]
        all_periods = all_periods.rename(columns={col_name: "periods"})
        logger.info('Computing target_req')
        for j, t in product(self.d.items, self.d.periods):
            demand_value = target_diff.loc[(j, t)]
            self.get_multilayer_prop_demand(
                j, t, demand_value, target_prop_demand, target_demand, "target"
            )
        logger.info('    done computing target_req')
        target_prop_demand = self._format_prop_demand(all_periods, target_prop_demand)
        target_diff.index.names = ["Item", "Period"]
        return target_prop_demand, target_diff

    def build(self, demand):
        """
        function to build SCMS capacity planning mathematical optimization model
        input:
         - self: MIPModel object
         - demand: dictionary where key is (product, time period), value is demand
        output:
         - Pyomo model where
         - Parameter Dⱼₜ: demand of product j in period t
         - Parameter Fᵢ: Investment value per capacity chunk of work center i
         - Parameter H:  Holding cost proportion of a product per period
         - Parameter EPⱼ: Earliest production of a product j
         - Paremeter Availabilityᵢₜ: Avaiability of work center i in period t
         - Paremeter Sourcingᵢₜ: Minimum sourcing of component k in period t
         - Parameter SRⱼₖ: Scrap Rate of component k to product j
         - Parameter AB: Anual Budget
         - Parameter Cⱼ: Inventory carrying cost of item j
         - Parameter ICₜ: Investment coefficient on period t
         - Parameter cₗₜ: Component lead time
         - Parameter M: Large enough number
         - Parameter EFWₚ: End period of frozen window of project p
         - Parameter CFP: Committed fixed period
         - Set CFi: Committed fixed projects
         - Set CFl: Committed flexible projects
         - Variables xⱼₜ: actual production of product j in period t
         - Variables xᵢⱼₜ: actual production of product j in period t
         - Variables uᵢⱼₜ: planned cross qualification of work center i for product j in period t
         - Variables vᵢⱼₜ: qualified cross qualification of work center i for product j in period t
         - Variables yᵢₜ: qualified capacity investment of project p of type n in period t
         - Variables y_sᵢₜ: planned capacity investment of project p of type n in period t
         - Variables sⱼₜ: inventory level of product j in the end of period t
         - Variables zᵢₜ: capacity of work center i in period t
         - Variables zᵢⱼₜ: capacity of work center i in period t
         - Variables dⱼₜ: demand fullmillment of item j in period t
         - Variables y_aᵢₜ: cumulative sum of qualified capacity investment of project p until period t
         - Variables y_dᵢₜ: cumulative sum of depreciated capacity investment of project p until period t
         - Variables y_fᵢₜ: instance of depreciation of project p in period t
         - Variables sˢⱼₜ: shortage of inventory from target level.
         - Variables sᴱⱼₜ: excess of inventory from target level.
         - Variables sᴸⱼₜ: violation of s from minimum level (lower bound).
         - Variables sᵁⱼₜ: violation of s from maximum level (upper bound).
         - Variables sᴮᴵᴺⱼₜ: shortage from target indicator
         - Variables wcˢᵢₜ: shortage of utilization from work center minimum utilization.
         - Variables wcᴱᵢₜ: excess of utilization from work center maximum utilization.
         - Variables ad_ᵢⱼₜ: deviation from min allocation specified for item j in workcenter i in period t
         - Objective function:      max Σⱼₜ Cⱼ·dⱼₜ - Σᵢₜ y_fᵢₜ·Fᵢ/(Depreciation_periods) -
                                    Σⱼₜ H·Cⱼ·Sᴱⱼₜ - Σⱼₜ Cⱼ·rⱼₜ - Σᵢⱼₜ QCostᵢⱼ·vᵢⱼₜ -
                                    Σⱼₜ αₑ·(Sᴱⱼₜ + 10 sᵁⱼₜ) - Σⱼₜ αₛ·(Sˢⱼₜ + 10 sᴸⱼₜ)
         - Inventory balance:       sⱼₜ = sⱼₜ₋₁ + xⱼₜ - dⱼₜ             ∀ j, t>1
         - Max inventory Bound:     sⱼₜ ≤ MaxBoundⱼₜ                   ∀ j, t
         - Min inventory Bound:     MinBoundⱼₜ ≤  sⱼₜ                  ∀ j, t
         - Target inventory Bound:  sⱼₜ =  Targetⱼₜ - Sˢⱼₜ + Sᴱⱼₜ       ∀ j, t
         - Demand fullmillment:     dⱼₜ₋꜀ₗₜ_ₖ ≤ Dⱼₜ                      ∀ j ∈ Jp, t
         - BOM:                     dₖₜ₋cₗₜ = Σⱼ₍ₖ₎ αⱼₖ·xⱼₜ/(1 - SRⱼₖ)    ∀ j ∈ Jc, t, k ∉ Group
         - Group BOM:               Σ₉₍ₖ₎dₖₜ₋cₗₜ = Σⱼ₍ₖ₎ αⱼₖ·xⱼₜ/(1 - SRⱼₖ) ∀ j ∈ Jc, t, k ∈ Group
         - Min Sourcing             Σ₉₍ₖ₎dₖₜ · Sourcingₖₜ ≤ dₖₜ          ∀ t, k ∈ Group
         - Init Project:            Σₜ yₚₜ ≤ 1                         ∀ p ∈ Pᴵ
         - Subs Project:            Σₜ yₚₜ ≤ γₚ - 1                    ∀ p ∈ Pˢ
         - Project investment:      yᵨₜ ≤ γₚ·Σₜₜ yᵨₜₜ                    ∀ p,q: p=q, p∈ Pᴵ,q∈ Pˢ, t>1
         - Investment Lead Time:    y_sᵨₜₜ = yᵨₜ                         ∀ p ∈ P, t, tt∈ T: tt=t-LTᵨ
         - Capacity used:           x̂ᵢⱼₜ/Yieldᵢⱼₜ = ẑᵢⱼₜ·PRᵢⱼₜ          ∀ i, j, t∈ T: TVUᵢⱼ≤t≤TVFᵢⱼ
         - Capacity not used        x̂ᵢⱼₜ = 0                          ∀ i, j, t∈ T: TVUᵢⱼ>t ∨ t>TVFᵢⱼ
         - Capacity constraints:    Σⱼ₍ᵢ₎ ẑᵢⱼₜ ≤ zᵢₜ · Availabilityᵢₜ   ∀ i, t
         - Capacity constraints:    Σⱼ₍ᵢ₎ ẑᵢⱼₜ + wcˢᵢₜ ≥ zᵢₜ · Availabilityᵢₜ · Min_Utilᵢₜ ∀ i, t
         - Capacity constraints:    Σⱼ₍ᵢ₎ ẑᵢⱼₜ - wcᴱᵢₜ ≤ zᵢₜ · Availabilityᵢₜ · Max_Utilᵢₜ ∀ i, t
         - Capacity investment:     zᵢₜ = zᵢₜ₋₁ + Σₚ ϕₚ·yₚₜ + CIᵢₜ        ∀ i, t>1
         - Cross Qualification:     x̂ᵢⱼₜ ≤ M·Σ(ₜₜ_ₜ) vᵢⱼₜₜ                ∀ (i,j) ∈ Q, t>1
         - CQ Selection:            Σₜ vᵢⱼₜ ≤ 1                        ∀ (i,j) ∈ Q
         - CQ Lead Time:            uᵢⱼₜₜ = vᵢⱼₜ                        ∀ (i,j) ∈ Q, (tt, t) ∈ T: tt = t-CLTᵢⱼ
         - Alternative Work Center: Σ₍ᵢ_Iⱼₛ₎ x̂ᵢⱼₜ = Σ₍ᵢ_Iⱼₛ₊₁₎ x̂ᵢⱼₜ/Yieldᵢⱼₜ  ∀s < LSⱼ
         - Alt. Work Center output: Σ₍ᵢ_Iⱼₛ₎ x̂ᵢⱼₜ = xⱼₜ                 ∀s = LSⱼ
         - Annual budget:           Σᵢ Fᵢₜ·yᵢₜ + Σⱼ Cⱼ·pⱼₜ ≤ AB       ∀ t
         - Cumulative investment:   y_aᵢₜ = Σᵢᵧ yᵢᵧ                   ∀ 𝛄 ϵ [1..t], ∀ i
         - Cumulative depreciated:  y_dᵢₜ = Σᵢᵧ yᵢᵧ                   ∀ 𝛄 ϵ [1..t-N], ∀ i
         - Depreciation instances:  y_fᵢₜ = y_aᵢₜ - y_dᵢₜ                  ∀ j, t
         - Shortage dev. control:   sˢⱼₜ ≤ sᴮᴵᴺⱼₜ·(targetⱼₜ - 1)        ∀ j, t
         - Excess dev. control:     sᴱⱼₜ ≤ (1 -sᴮᴵᴺⱼₜ)·2·s_UBⱼₜ         ∀ j, t
         - Minimum Allocation :     x̂ᵢⱼₜ ≥ Σ₍ᵢᵢ_Iⱼₛ₎x̂ᵢᵢⱼₜ*MAᵢⱼ-adᵢⱼₜ    ∀ i, j, t, s: s ∈ Steps and s(i)=s(ii)
         - Inv. bounds violation:   sᴸⱼₜ ≥ s_LBⱼₜ - sⱼₜ                 ∀ j, t
         - Inv. bounds violation:   sᵁⱼₜ ≥ sⱼₜ - s_UBⱼₜ                 ∀ j, t
         - Earliest production:     xⱼₜ = 0                           ∀ j, t: t<EPⱼ
         - Variable bounding:       xⱼₜ, yᵢₜ, sⱼₜ, zᵢₜ,
                                    dⱼₜ, pⱼₜ, qᵢₜ. rᵢₜ, tᵢₜ             >= 0
         - Committed Fixed:         yₚₜ = 1                           ∀ p ϵ CFi, t ϵ CFPₚ
         - Committed Fixed 2:       yₚₜ = 0                           ∀ p ϵ CFi, ∀ t <> CFPₚ
         - Flex in Frozen Window:   yₚₜ = 0                           ∀ p ϵ CFl, ∀ t <= EFW
         - Flex out Frozen Window:  Σ(t > EFW) yₚₜ = 1                ∀ p ϵ CFl
        """
        # initialize
        d = self.d
        model = pyo.ConcreteModel()
        logger.info("Obtaining inventory bounds")
        self.get_inventory_bounds(demand)
        logger.info("Obtained inventory bounds")
        self.initial_inventory_level()
        logger.info("Obtained initial inventory")
        last_demand = self.get_last_demand(demand)
        logger.info('Obtained last demand')
        model.demand_param = pyo.Param(
            d.products, d.periods, mutable=True, initialize=0
        )

        model.minbound_param = pyo.Param(d.items, d.periods, mutable=True, initialize=0)

        model.maxbound_param = pyo.Param(d.items, d.periods, mutable=True, initialize=0)

        model.target_param = pyo.Param(d.items, d.periods, mutable=True, initialize=0)

        model.last_demand_param = pyo.Param(d.products, mutable=True, initialize=0)

        model.S0_param = pyo.Param(d.items, mutable=True, initialize=0)

        #! TODO: it should be in the input info 
        self.d.use_discrete_po = 'No'

        for j in d.products:
            for t in d.periods:
                CLT = d.Component_Lead_Time.get(j, 0)
                model.last_demand_param[j].value = round(
                    last_demand.get(j, 0), d.DECIMAL_NUMBER
                )
                # get value of demand
                if d.period_numbers[t] < CLT:
                    # t_pre will be last periods because of calling negative index of d.periods
                    model.demand_param[j, t] = model.last_demand_param[j].value
                else:
                    model.demand_param[j, t] = round(
                        demand.get((j, t), 0), d.DECIMAL_NUMBER
                    )

        for j in d.items:
            model.S0_param[j].value = d.S0.get(j, 0)

        model.ML_param = pyo.Param(d.items, d.periods, mutable=True, initialize=0)
        for j in d.items:
            for t in d.periods:
                model.ML_param[j, t].value = d.ml_demand.get((j, t), 0)

        # xⱼₜ
        model.x = pyo.Var(d.items, d.periods, initialize=0, within=pyo.NonNegativeReals)

        # xᵢⱼₜ
        model.x_hat = pyo.Var(d.ijt_set, initialize=0, within=pyo.NonNegativeReals)

        # yᵢₜ
        model.y = pyo.Var(d.projects, d.periods, initialize=0, within=pyo.NonNegativeIntegers)

        # yˢᵢₜ
        model.y_s = pyo.Var(
            d.projects, d.periods, initialize=0.0, within=pyo.NonNegativeIntegers
        )

        # sⱼₜ
        model.s = pyo.Var(d.items, d.periods, initialize=0, within=pyo.NonNegativeReals)

        # sᴸⱼₜ
        model.s_l = pyo.Var(d.items, d.periods, initialize=0, within=pyo.NonNegativeReals)

        # sᵁⱼₜ
        model.s_u = pyo.Var(d.items, d.periods, initialize=0, within=pyo.NonNegativeReals)

        # zᵢₜ
        model.z = pyo.Var(d.workcenters, d.periods, initialize=0, within=pyo.NonNegativeReals)

        # zᵢⱼₜ
        model.z_hat = pyo.Var(d.ijt_set, initialize=0, within=pyo.NonNegativeReals)

        # dⱼₜ
        model.d = pyo.Var(d.items, d.periods, initialize=0, within=pyo.NonNegativeReals)

        # yᵃᵢₜ
        model.y_a = pyo.Var(d.projects, d.periods, initialize=0, within=pyo.NonNegativeIntegers)

        # yᵈᵢₜ
        model.y_d = pyo.Var(d.projects, d.periods, initialize=0, within=pyo.NonNegativeIntegers)

        # yᶠᵢₜ
        model.y_f = pyo.Var(d.projects, d.periods, initialize=0, within=pyo.NonNegativeIntegers)

        # uᵢⱼₜ
        model.u = pyo.Var(d.ijt_set_cq, within=pyo.Binary, initialize=0)

        # vᵢⱼₜ
        model.v = pyo.Var(d.ijt_set_cq, within=pyo.Binary, initialize=0)

        # Sᴱⱼₜ
        model.s_e = pyo.Var(d.items, d.periods, initialize=0, within=pyo.NonNegativeReals)

        # Sˢⱼₜ
        model.s_s = pyo.Var(d.items, d.periods, initialize=0, within=pyo.NonNegativeReals)

        # sᵇⱼₜ
        model.s_bin = pyo.Var(d.items, d.periods, initialize=0, within=pyo.Binary)

        # adᵢⱼₜ
        model.ad = pyo.Var(d.ijt_set, initialize=0, within=pyo.NonNegativeReals)

        model.wc_s = pyo.Var(d.workcenters, d.periods, initialize=0, within=pyo.NonNegativeIntegers)

        model.wc_e = pyo.Var(d.workcenters, d.periods, initialize=0, within=pyo.NonNegativeIntegers)

        model.wc_dev_s = pyo.Var(d.workcenters, d.periods, initialize=0, within=pyo.NonNegativeReals)

        model.wc_dev_e = pyo.Var(d.workcenters, d.periods, initialize=0, within=pyo.NonNegativeReals)

        # objective is to maximize total demand fulfilled
        # minus total investment (capital or inventory) minus total holding cost
        # while penalizing deviation from target inventory.

        #! TODO: this info should be parameterized
        d.WEIGHT_SALES = 1
        d.WEIGHT_CAPITAL_INVESTMENT_COST = 1
        d.WEIGHT_INVENTORY_INVESTMENT_COST = 1
        d.WEIGHT_CROSS_QUALIFICATION_COST = 1
        d.WEIGHT_HOLDING_COST = 1
        d.WEIGHT_TOTAL_DEVIATION = 1
        
        #! TODO: This info comes from the input:

        d.DEPRECIATION_YEARS = 5
        d.PERIODS_PER_YEAR = 4
        d.WEEKS_PER_YEAR = 52
        d.OPTIMIZATION_OBJECTIVE = "Standard"
        d.HOLDING_COST = 9.0
        d.PERIODS_PER_YEAR = 4
        d.ANNUAL_BUDGET = 100

        d.optimization_mode = "Optimized"
        d.allow_exceeding_capacity = False
        d.consider_workcenter_avail = True
        d.consider_wc_utilization = False
        d.time_limit = 200
        d.optimality_gap = 5.0  # (%)
        d.excess_deviation_penalization = 10
        d.shortage_deviation_penalization = 10
        d.Profit = 60.0  # (%)
        d.ANNUAL_DISCOUNT_RATE = 12.0  # (%)
        d.WEIGHT_SALES = 1
        d.WEIGHT_CAPITAL_INVESTMENT_COST = 1
        d.WEIGHT_INVENTORY_INVESTMENT_COST = 1
        d.WEIGHT_CROSS_QUALIFICATION_COST = 1
        d.WEIGHT_HOLDING_COST = 1
        d.WEIGHT_TOTAL_DEVIATION = 1
        d.WEEKS_PER_PERIOD = 13.0
        d.force_demand_fulfillment = False
        d.balance_wc_groups = False


        logger.info("Creating objective function")
        def _objfunc(model):
            model.SALES = (
                d.WEEKS_PER_PERIOD
                * d.WEIGHT_SALES
                * sum(
                    d.Price.get(j, 0) * model.d[j, t] * d.DISCOUNT_FACTOR[t]
                    for j in d.products
                    for t in d.periods
                )
            )

            model.CAPITAL_INVESTMENT = d.WEIGHT_CAPITAL_INVESTMENT_COST * sum(
                d.F.get(p, 0)
                / (d.DEPRECIATION_YEARS * d.PERIODS_PER_YEAR)
                * model.y_f[p, t]
                * d.DISCOUNT_FACTOR[t]
                for p in d.projects
                for t in d.periods
            )

            model.CROSS_QUALIFICATION_COST = d.WEIGHT_CROSS_QUALIFICATION_COST * sum(
                model.v[i, j, t] * d.DISCOUNT_FACTOR[t] * d.QCost[i, j]
                for (i, j, t) in d.ijt_set_cq
            )

            model.HOLDING_COST = (
                d.WEEKS_PER_PERIOD
                * d.WEIGHT_HOLDING_COST
                * sum(
                    d.HOLDING_COST
                    / 100
                    / d.PERIODS_PER_YEAR
                    * d.Standard_Cost.get(j, 0)
                    * model.s_e[j, t]
                    * d.DISCOUNT_FACTOR[t]
                    for j in d.items
                    for t in d.periods
                )
            )

            model.TOTAL_DEVIATION = d.WEIGHT_TOTAL_DEVIATION * sum(
                (
                    d.excess_deviation_penalization
                    * d.item_deviation_penalization[j]
                    * model.s_e[j, t]
                    + d.shortage_deviation_penalization
                    * d.item_deviation_penalization[j]
                    * model.s_s[j, t]
                    + 10
                    * d.excess_deviation_penalization
                    * d.item_deviation_penalization[j]
                    * model.s_u[j, t]
                    + 10
                    * d.shortage_deviation_penalization
                    * d.item_deviation_penalization[j]
                    * model.s_l[j, t]
                )
                * d.DISCOUNT_FACTOR[t]
                for j in d.items
                for t in d.periods
            )

            model.WC_CAPACITY_DEV = 10000000000 * sum(
                (model.wc_s[i, t] + model.wc_e[i, t])
                / (d.Capacity[i] / d.Max_ProductionRate[i])
                for i in d.workcenters
                for t in d.periods
            )

            model.ALLOCATION_DEVIATION = sum(
                d.allocation_deviation_penalization[j] * model.ad[i, j, t]
                for (i, j, t) in d.ijt_set
            )

            model.PRIORITY_COST = sum(
                d.wc_priority_cost.get((i, j, t), 0) * model.x_hat[i, j, t]
                for (i, j, t) in d.ijt_set
            )

            model.FLEXIBLE_COMMITTED_INVESTMENT = sum(
                d.flexible_inv_cost[p, t] * model.y[p, t]
                for p in d.committed_flexible
                for t in d.periods
            )

            model.GROUP_UTILIZATION_DEVIATION = 0.01 * sum(
                model.wc_dev_s[i, t] + model.wc_dev_e[i, t]
                for i in d.workcenters
                for t in d.periods
            )

            obj = (
                model.SALES
                - model.CAPITAL_INVESTMENT
                - model.CROSS_QUALIFICATION_COST
                - model.HOLDING_COST
                - model.TOTAL_DEVIATION
                - model.ALLOCATION_DEVIATION
                - model.WC_CAPACITY_DEV
                - model.PRIORITY_COST
                - model.FLEXIBLE_COMMITTED_INVESTMENT
                - model.GROUP_UTILIZATION_DEVIATION
            )

            return obj

        # objective function of maximizing attended demand
        model.obj_attended_demand = pyo.Objective(
            rule=(
                lambda model: sum(
                    (model.d[j, t] * (0.95 ** d.period_numbers[t]))
                    for j in d.products
                    for t in d.periods
                )
            ),
            sense=pyo.maximize,
        )
        model.obj_attended_demand.deactivate()

        # objective function
        model.obj = pyo.Objective(
            rule=_objfunc,
            sense=pyo.maximize,
        )

        logger.info("Creating model constraints")

        # # constraints
        # 1. Inventory Constraints
        def _inventory_balance(m, j, t):
            CLT = d.Component_Lead_Time.get(j, 0)
            t_prima = d.periods[d.period_numbers[t] - CLT]            
            t_pre = d.periods[d.period_numbers[t] - 1]                                    
            if d.period_numbers[t] < CLT: 
                if d.period_numbers[t] == 0:
                    return m.s[j, t] == m.S0_param[j].value - m.d[
                        j, t
                    ] + d.Discrete_PO.get((j, t), 0)
                else:
                    return m.s[j, t] == m.s[j, t_pre] - m.d[j, t
                    ] + d.Discrete_PO.get((j, t), 0)
            else:   
                if d.period_numbers[t] == 0:
                    return m.s[j, t] == m.S0_param[j].value + m.x[j, t_prima] - m.d[
                        j, t
                    ] + d.Discrete_PO.get((j, t), 0)
                else:
                    return m.s[j, t] == m.s[j, t_pre] + m.x[j, t_prima]- m.d[
                        j, t
                    ] + d.Discrete_PO.get((j, t), 0)

        model.inventory_balance = pyo.Constraint(
            d.items, d.periods, rule=_inventory_balance
        )
        logger.info(
            f"{len(model.inventory_balance)} Inventory Balance constraints built!"
        )

        def _production_last_periods(m,j,t):
            CLT = d.Component_Lead_Time.get(j, 0)
            t_prima = d.periods[d.period_numbers[t] - CLT]            
            t_limit = d.periods[d.period_numbers.tail(1).iloc[0] - CLT]
            if j in d.items:
                if t_limit < t_prima:
                    return m.x[j,t_prima] == m.x[j, t_limit]
                else:
                    return pyo.Constraint.Skip           

        model.production_last_periods = pyo.Constraint(
            d.items, d.periods, rule=_production_last_periods
        )
        logger.info(
            f"{len(model.production_last_periods)} Items Production when Last_Period - CLT < t!"
        )

        def _inventory_shortage_dev(m, j, t):
            m.target_param[j, t].value = d.target[j, t]
            return m.s_s[j,t] >= m.target_param[j, t].value - m.s[j,t]

        model.inventory_shortage = pyo.Constraint(
            d.items, d.periods, rule=_inventory_shortage_dev
        )
        logger.info(
            f"{len(model.inventory_shortage)} Inventory Shortage Deviation constraints built!"
        )

        def _inventory_excess_dev(m, j, t):
            return m.s_e[j,t] >= m.s[j,t] - m.target_param[j, t].value

        model.inventory_excess = pyo.Constraint(
            d.items, d.periods, rule=_inventory_excess_dev
        )
        logger.info(
            f"{len(model.inventory_excess)} Inventory Excess Deviation constraints built!"
        )

        # 2. Capacity Constraints
        def _capacity_investment(m, i, t):
            if d.period_numbers[t] == 0:
                # initial capacity (z_pre in original unit ea/wk or hr/wk)
                z_pre = d.Capacity.get(i, 0)
            else:
                # get t - 1 (previous t) STRING
                t_pre = d.periods[d.period_numbers[t] - 1]
                # z_pre in original unit ea/wk or hr/wk
                z_pre = m.z[i, t_pre] * d.Max_ProductionRate[i]
            return (
                m.z[i, t]  # z always in hr/wk
                == (
                    z_pre
                    + sum(
                        d.PHI.get(p, 0) * m.y[p, t]
                        for p in d.projects
                        if (i, p) in d.ip_set
                    )
                )  # z_pre and projects are in original unit
                / d.Max_ProductionRate[i]
            )

        model.capacity_investment = pyo.Constraint(
            d.workcenters, d.periods, rule=_capacity_investment
        )
        logger.info(
            f"{len(model.capacity_investment)} Capacity Investment constraints built!"
        )
        # 3. Demand Fulfillment
        def _demand_product(m, j, t):
            # force demand when the option is active
            CLT = d.Component_Lead_Time.get(j, 0)
            t_pre = d.periods[d.period_numbers[t] - CLT]
            if not d.force_demand_fulfillment:
                return m.d[j, t_pre] <= m.demand_param[j, t]
            else:
                return m.d[j, t_pre] == m.demand_param[j, t]

        model.demand_product = pyo.Constraint(
            d.products, d.periods, rule=_demand_product
        )
        logger.info(f"{len(model.demand_product)} Demand Product constraints built!")

        # 4. BOM

        def _demand_component(m, k, t):
            # The delivey is in general d[k,t] = c*x[j,t]
            # So, the demand in the period is triggered by the consumption. 
            CLT = d.Component_Lead_Time.get(k, 0)
            if CLT < len(d.periods):
                return m.d[k, t] == sum(
                    d.Valid_Qty_Per.get((k, j, t), 0)
                    * m.x[j, t]
                    / (1 - d.Scrap_Rate.get((k, j), 0))
                    for j in d.valid_j_set.get(k, {})
                )
            else:
                return pyo.Constraint.Skip

        model.demand_component = pyo.Constraint(
            d.single_items, d.periods, rule=_demand_component
        )

        logger.info(
            f"{len(model.demand_component)} Demand Component constraints built!"
        )

        # # 5. Project Constraints
        # initial or primary investment in project p
        def _initial_investment(m, p):
            return sum(m.y[p, t] for t in d.periods) <= 1

        model.initial_investment = pyo.Constraint(
            d.projects_init, rule=_initial_investment
        )
        logger.info(
            f"{len(model.initial_investment)} Initial Investment constraints built!"
        )
        # subs investment in project p
        def _subs_investment(m, p):
            return sum(m.y[p, t] for t in d.periods) <= max(0, d.Max_Available[p] - 1)

        model._subs_investment_rule = pyo.Constraint(
            d.projects_subs, rule=_subs_investment
        )
        logger.info(
            f"{len(model._subs_investment_rule)} Subsequent Investment constraints built!"
        )
        # investment sequence: subs investment (q) only if initial is done (p)
        def _investment_sequence(m, p, q, t):
            if (p, q) in d.pq_set:
                return m.y[q, t] <= d.Max_Available[q] * sum(
                    m.y[p, tao]
                    for tao in d.periods
                    if d.period_numbers[tao] <= d.period_numbers[t]
                )
            else:
                return pyo.Constraint.Skip

        model._investment_sequence_rule = pyo.Constraint(
            d.projects_init, d.projects_subs, d.periods, rule=_investment_sequence
        )
        logger.info(
            f"{len(model._investment_sequence_rule)} Investment Sequence constraints built!"
        )
        # Investment lead time
        def _investment_leadtime(m, p, t):
            # period number, not string
            t_plan = d.period_numbers[t] - d.LT.get(p, 0)
            if t_plan < 0:  # it should be zero at least, to begin the plan
                # string
                return m.y[p, t] == 0
            else:  # if it is bigger than zero, from t to T, y[t] could be {0,1}
                t_plan = d.periods[t_plan]
                return m.y_s[p, t_plan] == m.y[p, t]

        model._investment_leadtime_rule = pyo.Constraint(
            d.projects, d.periods, rule=_investment_leadtime
        )
        logger.info(
            f"{len(model._investment_leadtime_rule)} Investment Leadtime constraints built!"
        )
        # 6. Capacity Limit Constraints
        def _capacity_limit(m, i, t):
            ijt_set = set((ii, j, tt) for ii, j, tt in d.ijt_set if i == ii and t == tt)
            used_capacity = sum(m.z_hat[i, j, t] for (i, j, t) in ijt_set)
            # used capacity can be empty because of phasing out in later periods
            if type(used_capacity) != int:
                return used_capacity <= m.z[i, t] * d.Availability.get((i, t), 1.0)
            else:
                return pyo.Constraint.Skip

        if not d.allow_exceeding_capacity:
            model.capacity_product = pyo.Constraint(
                d.workcenters, d.periods, rule=_capacity_limit
            )
            logger.info(
                f"{len(model.capacity_product)} Capacity Limit constraints built!"
            )

        def _min_capacity(m, i, t):
            return (
                sum(m.z_hat[i, j, t] for j in d.items if (i, j, t) in d.ijt_set)
                + m.wc_s[i, t]
            ) >= m.z[i, t] * d.Availability.get((i, t), 1.0) * d.Min_Utilization.get(
                (i, t), 0
            )

        if d.consider_wc_utilization:
            model.min_capacity = pyo.Constraint(
                d.workcenters, d.periods, rule=_min_capacity
            )
            logger.info(f"{len(model.min_capacity)} Min Capacity constraints built!")

        def _max_capacity(m, i, t):
            return (
                sum(m.z_hat[i, j, t] for j in d.items if (i, j, t) in d.ijt_set)
                - m.wc_e[i, t]
            ) <= m.z[i, t] * d.Availability.get((i, t), 1.0) * d.Max_Utilization.get(
                (i, t), 1
            )

        if d.consider_wc_utilization:
            model.max_capacity = pyo.Constraint(
                d.workcenters, d.periods, rule=_max_capacity
            )
            logger.info(f"{len(model.max_capacity)} Max Capacity constraints built!")


        # Available Capacity
        def _available_capacity(m, i, j, t):
            if d.Yield.get((i, j, t), -1) != -1:
                return (
                    m.x_hat[i, j, t] / d.Yield[i, j, t]
                    == m.z_hat[i, j, t] * d.PRate[i, j]
                )
            else:
                return pyo.Constraint.Skip

        model.available_capacity = pyo.Constraint(d.ijt_set, rule=_available_capacity)
        logger.info(
            f"{len(model.available_capacity)} Available Capacity constraints built!"
        )
        # 6.1 Cross Qualification Constraints
        def _cross_qualification(m, i, j, t):
            M = (
                d.Capacity[i] + d.Max_Capacity_Increase[i] * d.max_projects
            ) * d.Max_ProductionRate_BigM[i]

            # periods used in RHS should be only valid periods from qualifications
            t_valid = {t for (i1, j1, t) in d.ijt_set_cq if i1 == i and j1 == j}
            t_rhs = {d.periods[tau] for tau in range(d.period_numbers[t] + 1)}
            t_rhs = t_rhs.intersection(t_valid)

            # when no valid periods, skip the constraint
            if bool(t_rhs):
                return m.x_hat[i, j, t] <= M * sum(
                    m.v[i, j, tau] for tau in t_rhs
                )
            else:
                return pyo.Constraint.Skip

        model.cross_qualification = pyo.Constraint(
            d.ijt_set_cq, rule=_cross_qualification
        )
        logger.info(f"{len(model.cross_qualification)} CrossQ constraints built!")

        def _qualification_selection(m, i, j):
            # periods used in RHS should be only valid periods from qualifications
            t_valid = {t for (i1, j1, t) in d.ijt_set_cq if i1 == i and j1 == j}
            t_rhs = t_valid.intersection(set(d.periods))

            # when no valid periods, skip the constraint
            if bool(t_rhs):
                if (i, j) in d.ij_set and d.Validation_Status[i, j] == "Capable":
                    return sum(m.v[i, j, t] for t in t_rhs) <= 1
                else:
                    return sum(m.v[i, j, t] for t in t_rhs) <= 0
            else:
                return pyo.Constraint.Skip

        model.qualification_selection = pyo.Constraint(
            d.ij_set_cq, rule=_qualification_selection
        )
        logger.info(
            f"{len(model.qualification_selection)} Qualification Selection constraints built!"
        )
        def _qualification_lead_time(m, i, j, t):
            t_plan = d.period_numbers[t] - d.QLT.get((i, j), 0)
            if t_plan < 0:
                return m.v[i, j, t] == 0
            else:
                # when the t_plan is not valid period for u, skip the constraint
                if (i, j, d.periods[t_plan]) in d.ijt_set_cq:
                    return m.u[i, j, d.periods[t_plan]] == m.v[i, j, t]
                else:
                    return pyo.Constraint.Skip

        model.qualification_lead_time = pyo.Constraint(
            d.ijt_set_cq, rule=_qualification_lead_time
        )
        logger.info(
            f"{len(model.qualification_lead_time)} Qualification Leadtime constraints built!"
        )

        # 7. Alternative Work Centers Constraints
        def _alternative_workcenter(m, j, t, s):
            # Left Hand Side
            LHS = sum(m.x_hat[i, j, t] for i in d.jts_dict.get((j, t, s), []))
            # constraints applied only when item j has at least one production step
            if j in d.StepsOf:
                # constraints applied only when step s is one of the production steps of item j
                if s in d.StepsOf[j]:
                    # last step
                    if s == d.Last_Step[j]:
                        return LHS == m.x[j, t]
                    # other step
                    else:
                        RHS = sum(
                            m.x_hat[i, j, t] / d.Yield[i, j, t]
                            for i in d.jts_dict.get((j, t, s + 1), [])
                        )
                        # When LHS = 0 and RHS = 0 skip
                        if type(LHS) == int and type(RHS) == int:
                            return pyo.Constraint.Skip
                        else:
                            return LHS == RHS
                else:
                    return pyo.Constraint.Skip
            else:
                return pyo.Constraint.Skip

        model.alternative_workcenter = pyo.Constraint(
            d.items, d.periods, d.steps, rule=_alternative_workcenter
        )
        logger.info(
            f"{len(model.alternative_workcenter)} Alternative WC constraints built!"
        )
        # 8. Annual Budget
        def _annual_budget(m, t_year):
            periods_in_year = d.year_from_period[d.year_from_period == t_year].index
            return d.ANNUAL_BUDGET * 1e6 >= (
                sum(
                    sum(d.F.get(p, 0) * model.y_s[p, t] for p in d.projects)
                    + sum(
                        d.QCost[i, j] * m.v[i, j, t]
                        for (i, j, t) in d.ijt_set_cq
                        if d.Validation_Status[i, j] == "Capable"
                    )
                    for t in periods_in_year
                )
            )

        model.annual_budget = pyo.Constraint(d.years, rule=_annual_budget)
        logger.info(f"{len(model.annual_budget)} Annual Budget constraints built!")
        # 9. Min Allocation
        def _min_allocation(m, i, j, t):
            s = d.Step[i, j]
            total_production = sum(
                m.x_hat[ii, j, t] for ii in d.jts_dict.get((j, t, s), [])
            )
            return m.x_hat[i, j, t] >= (
                total_production * d.Min_Allocation.get((i, j), 0) - m.ad[i, j, t]
            )

        model.min_allocation = pyo.Constraint(d.ijt_set, rule=_min_allocation)
        logger.info(f"{len(model.min_allocation)} Min Allocation constraints built!")

        # 11. Depreciation effect for projects
        def _cumulative_investment(m, p, t):
            return m.y_a[p, t] == sum(
                m.y[p, tau]
                for tau in d.periods
                if d.period_numbers[tau] <= d.period_numbers[t]
            )

        model.cumulative_investment = pyo.Constraint(
            d.projects, d.periods, rule=_cumulative_investment
        )
        logger.info(
            f"{len(model.cumulative_investment)} Cumulative Investment constraints built!"
        )

        def _cumulative_depreciated_project(m, p, t):
            return m.y_d[p, t] == sum(
                m.y[p, tau]
                for tau in d.periods
                if d.period_numbers[tau]
                <= (d.period_numbers[t] - d.DEPRECIATION_YEARS * d.PERIODS_PER_YEAR)
            )

        model.cumulative_depreciated_project = pyo.Constraint(
            d.projects, d.periods, rule=_cumulative_depreciated_project
        )
        logger.info(
            f"{len(model.cumulative_depreciated_project)} Cumulative Depreciated Project constraints built!"
        )

        def _depreciation_instances(m, p, t):
            return m.y_f[p, t] == m.y_a[p, t] - m.y_d[p, t]

        model.depreciation_instances = pyo.Constraint(
            d.projects, d.periods, rule=_depreciation_instances
        )
        logger.info(
            f"{len(model.depreciation_instances)} Depreciation Instances constraints built!"
        )
        # 13. Inventory Bounds Violations
        def _min_inventory_violation(m, j, t):
            m.minbound_param[j, t].value = d.minbound[j, t]
            if m.minbound_param[(j, t)].value > 0:
                return m.s_l[j, t] >= m.minbound_param[j, t].value - m.s[j, t]
            else:
                return pyo.Constraint.Skip

        model.min_inventory_investment = pyo.Constraint(
            d.items, d.periods, rule=_min_inventory_violation
        )
        logger.info(
            f"{len(model.min_inventory_investment)} Min Inventory Violation constraints built!"
        )

        def _max_inventory_violation(m, j, t):
            m.maxbound_param[j, t].value = d.maxbound[j, t]
            if m.maxbound_param[(j, t)].value > 0:
                return m.s_u[j, t] >= m.s[j, t] - m.maxbound_param[j, t].value
            else:
                return pyo.Constraint.Skip

        model.max_inventory_investment = pyo.Constraint(
            d.items, d.periods, rule=_max_inventory_violation
        )
        logger.info(
            f"{len(model.max_inventory_investment)} Max Inventory Violation constraints built!"
        )

        # 13. Committed Increase
        def _Committed_Fixed(m, p):
            t = d.Committed_Period.get(p)
            return m.y[p, t] == 1

        model.committed_fixed = pyo.Constraint(d.committed_fixed, rule=_Committed_Fixed)
        logger.info(
            f"{len(model.committed_fixed)} Committed Fixed projects 1 constraints built!"
        )
        def _Committed_Fixed_2(m, p, t):
            t_committed = d.Committed_Period.get(p)
            if t == t_committed:
                return pyo.Constraint.Skip
            else:
                return m.y[p, t] == 0

        model.Committed_Fixed_2 = pyo.Constraint(
            d.committed_fixed, d.periods, rule=_Committed_Fixed_2
        )
        logger.info(
            f"{len(model.Committed_Fixed_2)} Committed Fixed projects 2 constraints built!"
        )

        def _flexible_in_frozen_window(m, p, t):
            return m.y[p, t] == 0

        model.flexible_in_frozen_window = pyo.Constraint(
            d.flexible_projects_out_window,
            d.frozen_window_periods,
            rule=_flexible_in_frozen_window,
        )
        logger.info(
            f"{len(model.flexible_in_frozen_window)} Flexible projects in Frozen Window constraints built!"
        )

        def _flexible_out_frozen_window(m, p):
            return sum(m.y[p, t] for t in d.unfrozen_window_periods) == 1

        model.flexible_out_frozen_window = pyo.Constraint(
            d.flexible_projects_out_window, rule=_flexible_out_frozen_window
        )
        logger.info(
            f"{len(model.flexible_out_frozen_window)} Flexible projects out of Frozen Window constraints built!"
        )

        # Variable Initializations
        for key in model.u:
            model.u[key].value = 0

        for key in model.d:
            model.d[key].value = 0

        # return pyomo model
        return model



    def valid_period(self, t, k, j):
        """return true/false if period t is a valid period for (component k, product j)"""
        if (k, j) in self.d.Qty_Per:
            # 1. get t_from and t_until
            t_from = self.d.Valid_From_Bom.get((k, j), self.d.periods[0])
            t_until = self.d.Valid_Until_Bom.get((k, j), self.d.periods[-1])

            # 2. convert string t to number n for comparing
            n = self.d.period_numbers[t]
            n_from = self.d.period_numbers.get(t_from, 0)
            n_until = self.d.period_numbers.get(t_until, len(self.d.periods))
            return n_from <= n <= n_until
        elif (k, j) in self.d.Group_Qty_Per:
            # If we have a group, we can check an item from the group
            k = self.d.group_components[k][0]
            # 1. get t_from and t_until
            t_from = self.d.Valid_From_Bom.get((k, j), self.d.periods[0])
            t_until = self.d.Valid_Until_Bom.get((k, j), self.d.periods[-1])

            # 2. convert string t to number n for comparing
            n = self.d.period_numbers[t]
            n_from = self.d.period_numbers.get(t_from, 0)
            n_until = self.d.period_numbers.get(t_until, len(self.d.periods))
            return n_from <= n <= n_until
        else:
            return False


# if __name__ == "__main__":
#     d = SCMSdata()
#     filename = "test/data/SimuHarmonic.xlsx"
#     d.read_simulation_input(filename)
#     gfilename = "MIT Files/input_data.xlsx"
#     d.read_optimization_input(gfilename)
#     m = MIPModel(d)
#     results = m.solve(method="")
