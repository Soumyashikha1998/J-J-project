# Copyright: Johnson & Johnson Digital & Data Science (JJDDS)
#
# This file contains trade secrets of JJDDS. No part may be reproduced or transmitted in any
# form by any means or for any purpose without the express written permission of JJDDS.
#
# Purpose:  Polaris Application Source

import logging
import math

import numpy as np
import pandas as pd


def get_projects_df(self):

    frame = {
        "Project": self.Project,
        "Reference Work Center": self.Reference_Work_Center,
        "Project Type": self.Project_Type,
        "Project Status": self.Project_Status,
        "Committed Period": self.Committed_Period,
        "Capacity Increase": self.Capacity_Increase,
        "Capacity Unit": self.Capacity_Unit_PR,
        "Initial Cost": self.Initial_Cost,
        "Initial Lead Time": self.Initial_Lead_Time,
        "Maximum Available": self.Maximum_Available,
        "Subsequent Cost": self.Subsequent_Cost,
        "Subsequent Lead Time": self.Subsequent_Lead_Time,
    }

    df = pd.DataFrame(frame)
    df = df[df["Project Status"] != "Inactive"]
    df = df[df["Reference Work Center"].isin(self.workcenters)]
    self.projects = self.projects[(self.projects.isin(df.index))]

    pdf_committed = df.copy(deep = True)
    pdf_committed = pdf_committed[
        (pdf_committed["Project Status"] == "Committed Fixed") |
        (pdf_committed["Project Status"] == "Committed Flexible")
        ]
    df = df[
        ~df.isin(pdf_committed.index)
    ].dropna()

    return df, pdf_committed


def get_validation_table_df(self):

    frame = {
        "Net Production Rate": self.Net_Production_Rate,
        "Status": self.Validation_Status,
        "Step": self.Step,
        "Valid From": self.Valid_From_Qual,
        "Valid Until": self.Valid_Until_Qual,
    }

    df = pd.DataFrame(frame)

    return df


def get_validation_table_series(vtdf):

    df1 = pd.DataFrame()
    # in precompute, but it can be edited in validation and data input
    df1 = vtdf
    # copying the filtered

    df_WC = df1.copy(deep=True)

    ijset_df = pd.DataFrame()
    ijset_df = df_WC

    PRate = ijset_df["Net Production Rate"].to_dict()

    Valid_From = ijset_df["Valid From"].to_dict()

    Valid_Until = ijset_df["Valid Until"].to_dict()

    ijs_set = ijset_df[["Step"]]

    ijs_set = set(
        ijs_set.reset_index()
        .set_index(["Work_Center", "Item", "Step"])
        .index
    )

    return ijs_set, PRate, Valid_From, Valid_Until


def get_production_steps(vtdf):
    """
    return two series by items
    e.g. Last_Step["Lens-23"] = 3, StepsOf["Lens-23"] = [1, 2, 3]
    """
    Last_Step = (
        vtdf.reset_index()[["Item", "Step"]]
        .drop_duplicates()
        .groupby(by=["Item"])["Step"]
        .max()
    ).to_dict()

    StepsOf = (
        vtdf.reset_index()[["Item", "Step"]]
        .drop_duplicates()
        .groupby(by=["Item"])["Step"]
        .apply(list)
    ).to_dict()

    return Last_Step, StepsOf


def get_committed_projects(self, pdf_committed):

    self.committed_fixed = pdf_committed[pdf_committed["Project Status"] == "Committed Fixed"].index
    self.committed_flexible = pdf_committed[pdf_committed["Project Status"] == "Committed Flexible"].index

    df_committed = pdf_committed.copy(deep=True)
    df_committed = df_committed[
        (~df_committed.isin(self.committed_fixed)) & 
        (~df_committed.isin(self.committed_flexible))
    ]

    columns = [
        "Reference Work Center",
        "Committed Period",
        "Capacity Increase",
        "Capacity Unit",
    ]

    df_committed = df_committed[columns]
    self.Committed_Period = df_committed["Committed Period"]
    self.Committed_Increase = df_committed.rename(columns = {
        "Reference Work Center":"Work_Center",
        "Committed Period":"Period",
        "Capacity Increase":"Committed Increase",
    }).set_index(["Work_Center","Period"])["Committed Increase"]

def get_it_set(df):
    """Return set of (project, period) tuples for committed projects"""
    columns_pt = ["Reference Work Center", "Committed Period"]
    df_committed = df[df["Project Status"] == "Committed"][columns_pt]
    return df_committed.set_index(columns_pt).index


def get_projects_subsets(self, pdf, pdf_committed):
    """
    Return pre-computed sets for application

    Input:
    application: use self.projects for set of projects, use self.projects_df for dataframe of table

    Output:
    projects: set of projects, including committed, base, initial, and subsequent, e.g., ["WC1", "Mold1_init", "Mold1_subs", "Mold2_init", "Mold2_subs", ...]
    LT: lead time of all projects (base, initial and subsequent).
    Chunk_Cost: cost of investment chunk of all projects (base, initial and subsequent)
    projects_i: series of list of projects indexed by work center, e.g., project_i["WC1"] = ["WC1", "Mold1_init", "Mold1_subs", "Mold2_init", "Mold2_subs"]
    projects_init: set of initial projects. e.g., {"Mold1_init", "Mold2_init"}
    projects_subs: set of subsequent projects. e.g., {"Mold1_subs", "Mold2_subs"}
    pq_set: list of tuples of related initial and subsequent projects, e.g., [("Mold1_init", "Mold1_subs"), (("Mold2_init", "Mold2_subs"), ...]
    alternative to pq_set: {"Mold1_init": "Mold_subs"}
    base_project: e.g., base_project["Mold1_subs"] = "Mold1"
    """
    self.projects = self.projects[
        (~self.projects.isin(self.workcenters)) & 
        (~self.projects.isin(self.committed_fixed)) & 
        (~self.projects.isin(self.committed_flexible))
    ]

    # # 0. define parameters for this function
    init = "_Initial"
    subs = "_Subsequent"
    default_max_available = 100
    default_project_type = "N/A"

    # # 1. initital and subsequent projects set
    self.projects_init = self.projects.copy() + init
    self.projects_subs = self.projects.copy() + subs

    # # 2. expand projects set
    # original projects set be like {"Mold1", "Mold2"}
    # extended projects set be like {"WC1", "Mold1_init", "Mold1_subs", "Mold2_init", "Mold2_subs"}
    self.projects_original = self.projects.copy()
    self.projects = self.workcenters.append(
        self.projects_init).append(
            self.projects_subs).append(
                self.committed_fixed
            ).append(
                self.committed_flexible)

    potential_projects_series = pd.Series("Potential", index = self.projects_init.append(self.projects_subs).append(self.workcenters))
    committed_projects_series = pd.Series("Committed", index = self.committed_fixed.append(self.committed_flexible))
    # TODO: Can this be a dict?
    self.project_types = pd.concat([
                    potential_projects_series.reset_index(),
                    committed_projects_series.reset_index()]).rename(
                        columns = {"index": "Project", 0: "Type"})

    # # 3.1 Lead Time and Chunk Cost by each project
    # Lead Time by initial, subsequent projects and committed
    LT1 = pdf[["Project", "Initial Lead Time"]].set_index(["Project"])[
        "Initial Lead Time"
    ]
    LT1.index = LT1.index.copy() + init
    LT2 = pdf[["Project", "Subsequent Lead Time"]].set_index(["Project"])[
        "Subsequent Lead Time"
    ]
    LT2.index = LT2.index.copy() + subs

    LT3 = pdf_committed[["Project", "Initial Lead Time"]].set_index(["Project"])[
        "Initial Lead Time"
    ]

    # Chunk_Cost by initial, subsequent projects and committed
    CC1 = pdf[["Project", "Initial Cost"]].set_index(["Project"])["Initial Cost"]
    CC1.index = CC1.index.copy() + init
    CC2 = pdf[["Project", "Subsequent Cost"]].set_index(["Project"])["Subsequent Cost"]
    CC2.index = CC2.index.copy() + subs
    CC3 = pdf_committed[["Project", "Initial Cost"]].set_index(["Project"])["Initial Cost"]
    # concat lead time of base, initial, and subsequent
    self.LT = pd.concat([self.Investment_Lead_Time, LT1, LT2, LT3]).to_dict()
    self.F = pd.concat([self.Increment_Cost, CC1, CC2, CC3]).to_dict()


    # # 4. Capacity Chunk (PHI), Max Available, and Project Type
    # Chunk Cost (PHI) by initial and subsequent projects
    PHI = pdf.copy(deep = True)
    cPHI = pdf_committed.copy(deep = True)
    PHI = unit_changer(self, PHI)
    cPHI = unit_changer(self, cPHI)
    
    PHI1 = PHI[["Project", "Capacity Increase"]].set_index(["Project"])[
        "Capacity Increase"
    ]
    PHI2 = PHI1.copy()
    PHI3 = cPHI[["Project", "Capacity Increase"]].set_index(["Project"])[
        "Capacity Increase"
    ]
    PHI1.index = PHI1.index.copy() + init
    PHI2.index = PHI2.index.copy() + subs
    # Max Available by initial and subsequent projects
    MR1 = pdf[["Project", "Maximum Available"]].set_index(["Project"])[
        "Maximum Available"
    ]
    MR1.fillna(default_max_available, inplace=True)
    MR2 = MR1.copy()
  
    MR3 = pdf_committed[["Project", "Maximum Available"]]
    MR3 = MR3.drop(columns = ["Project"])
    MR3["Maximum Available"] = 1
    MR3 = MR3['Maximum Available'].copy()

    MR1.index = MR1.index.copy() + init
    MR2.index = MR2.index.copy() + subs
    # Project Type by initial and subsequent projects
    PT1 = pdf[["Project", "Project Type"]].set_index(["Project"])["Project Type"]
    PT1.fillna(default_project_type, inplace=True)
    PT2 = PT1.copy()
    PT1.index = PT1.index.copy() + init
    PT2.index = PT2.index.copy() + subs
    PT3 = pdf_committed[["Project", "Project Type"]].set_index(["Project"])["Project Type"]
    # concat lead time of base, initial, and subsequent
    self.PHI = pd.concat([self.Capacity_Increment, PHI1, PHI2, PHI3]).to_dict()
    self.Max_Available = pd.concat([self.Maximum_Available, MR1, MR2, MR3]).to_dict()
    self.Project_Type = pd.concat([
        pd.Series(index=self.workcenters, data=["Base"] * len(self.workcenters)),
        PT1,
        PT2,
        PT3,
    ])

    # # 5. Project relationships
    # for all (p, q) in the set, p is the initial of q, and q is the subsequent of p
    self.pq_set = set(pd.Index([(p + init, p + subs) for p in self.projects_original]))

    # # 6. Projects by workcenter
    # for all (i, p) in the set, i is the work center of project p, p increase capacity for i
    df = pdf[["Reference Work Center", "Project"]]
    df["Initial"] = df["Project"].copy() + init
    df["Subsequent"] = df["Project"].copy() + subs
    df["Base"] = df["Reference Work Center"]
    df.rename(columns={"Project": "Original Project"}, inplace=True)

    df_committed = pdf_committed[["Reference Work Center","Project"]]
    df_committed["Base"] = df_committed["Reference Work Center"]

    df_ip = df.melt(
        id_vars=["Reference Work Center"],
        value_vars=["Initial", "Subsequent", "Base"],
        value_name="Project",
    )

    df_ip_committed = df_committed.melt(
        id_vars=["Reference Work Center"],
        value_vars="Project",
        value_name="col",
    ).rename(columns={'col': 'Project'})
    df_ip_committed["variable"] = "Base"
    df_ip = pd.concat(
        [
            df_ip,
            df_ip_committed,
            pd.DataFrame(
                {"Reference Work Center": self.workcenters, "Project": self.workcenters}
            ),
        ]
    )
    self.ip_set = set(df_ip.set_index(
        ["Reference Work Center", "Project"]
    ).index)

    # return aplication object
    return self


def get_component_lead_time(lead_time_days, periods_per_year=4, days_per_year=360):
    period_length = days_per_year / periods_per_year
    results = (lead_time_days / period_length).apply(lambda x : math.ceil(x))
    return results

def unit_changer(self, df):

    df1 = df.drop(columns = ['Project'])
    
    df1 = df1.reset_index().merge(self.Max_ProductionRate,
    how = 'inner',
    left_on = 'Reference Work Center',
    right_index = True)

    df1["Capacity Increase"] = np.where(
        df1["Capacity Unit"] == "ea/wk",
        df1["Capacity Increase"],
        df1["Capacity Increase"] * df1["Net Production Rate"]
    )
    df1["Capacity Unit"] = "ea/wk"
    # df = df.set_index("index")
    return df1

def get_max_productionrate(self, validation_table_df):
    frame = {
        "Capacity Unit": self.Capacity_Unit_WC,
    }
    df = pd.DataFrame(frame)

    df = df.rename_axis(index = "Work_Center")
    Max_ProductionRate = validation_table_df.groupby(level = "Work_Center")["Net Production Rate"].max()
    # TODO: Max_BigM doesn't seem to be used
    Max_ProductionRate_BigM = Max_ProductionRate.copy(deep=True)

    for i in df.index:
        if df.loc[i]["Capacity Unit"] in ["hour/we", "hr/wk", "hr/week"]:
            Max_ProductionRate[i] = 1
        else:
            pass

    return Max_ProductionRate, Max_ProductionRate_BigM

# TODO: Skipped Sets to decide if they need to be sets or keep as pd.Index
def rescale_sets(self):
    # Re-scale the sets to make sure not messed up after insight populate
    component_header_name = (
        self.Qty_Per.index.names[0]
        if self.Qty_Per.index.names[0]
        else "level_0"
    )
    self.products = self.products[self.products.isin(self.Qty_Per.index.get_level_values(1))]
    self.components = pd.Index(
        self.Qty_Per.reset_index()[component_header_name].drop_duplicates()
    )
    self.items = self.products.union(self.components)

    self.steps = pd.Index(self.Step.drop_duplicates())

    # Maks sure no None value in sets
    self.products = self.products[~self.products.isin(["None", "Null"])]
    self.components = self.components[~self.components.isin(["None", "Null"])]
    self.items = self.items[~self.items.isin(["None", "Null"])]
    self.periods = self.periods.sort_values()
    self.periods = self.periods[
        ~self.periods.isin(
            ["None", "Null"] + [f"Unnamed: {n}" for n in range(100)]
        )
    ]
    self.workcenters = self.workcenters[
        ~self.workcenters.isin(["None", "Null"])
    ]

    # Make sure the index set
    if type(self.ij_set) != pd.core.indexes.multi.MultiIndex:
        self.ij_set = self.ij_set.index
    if type(self.kj_set) != pd.core.indexes.multi.MultiIndex:
        self.kj_set = self.Qty_Per.index
    self.ij_set = self.ij_set[
        self.ij_set.get_level_values(0).isin(self.workcenters)
        ]

    # Modify kj set with dual sourcing
    dual_sourcing_df = self.Dual_Sourcing.reset_index()
    grouped_items = list(dual_sourcing_df["Component"].unique())
    single_items = self.components[~self.components.isin(grouped_items)]

    Qty_Per = self.Qty_Per.reset_index()

    Single_Qty_Per = Qty_Per[~Qty_Per['Component'].isin(grouped_items)]
    self.kj_set = Single_Qty_Per.set_index(['Component', 'Product']).index

    group_info = Qty_Per[Qty_Per['Component'].isin(grouped_items)]

    group_info = group_info.merge(
    dual_sourcing_df[['Component', 'Group']].drop_duplicates(),
    left_on = 'Component',
    right_on = 'Component'
    )

    CLT = self.Component_Lead_Time.reset_index()
    CLT.columns = ['Item', 'CLT']

    Scrap_Rate = self.Scrap_Rate.reset_index()

    group_info = group_info.merge(
        CLT,
        left_on = 'Component',
        right_on = 'Item'
    ).drop(['Item'], axis = 1)

    group_info = group_info.merge(
        Scrap_Rate,
        left_on = ['Component', 'Product'],
        right_on = ['Component', 'Product']
    )

    Group = pd.Index(group_info["Group"].unique())
    grouped_kj_set = pd.Index(group_info[['Group', 'Product']].drop_duplicates()) 

    group_components = {}
    group_components_by_item = {}
    for g in Group:
        components = group_info[group_info['Group'] == g]['Component'].unique()
        group_components[g] = list(components)
        for k in components:
            group_components_by_item[k] = list(components)
    
    Group_Qty_Per = group_info[['Product', 'Group', 'Qty_Per']].set_index(
        ["Group", "Product"])["Qty_Per"]
    Group_CLT = group_info[['Group', 'CLT']].set_index(
        ["Group"])["CLT"].drop_duplicates()
    Group_Scrap_Rate = group_info[['Product', 'Group', 'Scrap_Rate']].set_index(
        ["Group", "Product"])["Scrap_Rate"]

    self.single_items = single_items # Items that are not in a group
    self.grouped_items = grouped_items # Items that are in a group
    self.Group = Group # Set of item groups
    self.grouped_kj_set = grouped_kj_set # This will help with the necessary products for the group
    self.group_components = group_components # This dict stores the components for each group
    self.group_components_by_item = group_components_by_item #This dict stores components of the group by item
    self.Group_Qty_Per = Group_Qty_Per[~Group_Qty_Per.index.duplicated(keep='first')] # This holds the Qty Per for each group
    self.Group_CLT = Group_CLT[~Group_CLT.index.duplicated(keep='first')] #This has group CLT
    self.Group_Scrap_Rate = Group_Scrap_Rate[~Group_Scrap_Rate.index.duplicated(keep='first')] #This has group Scrap Rate
    self.Sourcing = dual_sourcing_df.reset_index().drop(["Group"], axis = 1).set_index(["Component", "Period"])["Allocation"]

    # Create period numbers to be countable in model, e.g., t - 1
    self.TOTAL_PERIODS = len(self.periods)
    self.period_numbers = pd.Series(
        index=self.periods, data=range(self.TOTAL_PERIODS)
    )

    #! TODO: balance_wc_groups should be an input. True or False
    self.balance_wc_groups = True 

    if self.balance_wc_groups:
        # Create a dataframe with WC, WC groups and Balance field
        wcg = pd.concat([self.Work_Center_Group, self.Balance_WC_Group], axis=1).reset_index()
        wcg =  wcg.rename(columns={'Work Center': 'Work_Center'})

        # Check if wc group is to be balanced according to input
        wcg['Balance_WC_Group'] = np.where(wcg['Balance_WC_Group'] == "Yes", 1, 0)
        wcg['Balance'] = wcg.groupby('Work_Center_Group')['Balance_WC_Group'].transform('sum')
        
        # Count workcenters on each workcenter group
        wcg['count'] = wcg.groupby('Work_Center_Group')['Work_Center'].transform('count')

        # Leave only wc groups that will be balanced and have more than one wc
        wcg = wcg[(wcg['Balance']>0)&(wcg['count'] > 1)].drop(columns=['Balance_WC_Group', 'Balance'])
        self.group_size = wcg[['Work_Center', 'count']].set_index('Work_Center')['count']
        
        # Get workcenters by group
        workcenters_by_group = {}
        for wc in wcg['Work_Center'].unique():
            wc_group = self.Work_Center_Group[wc]
            workcenters_by_group[wc] = list(wcg[wcg['Work_Center_Group'] == wc_group]['Work_Center'].unique())

        # Get initial capacity mean by wc group if no initial capacity
        capacity_wcg = wcg.merge(self.Capacity, left_on = 'Work_Center', right_index=True)
        capacity_wcg = capacity_wcg.merge(self.Max_ProductionRate, left_on = 'Work_Center', right_index=True)
        capacity_wcg['Capacity'] = capacity_wcg['Capacity']/capacity_wcg['Net Production Rate']
        capacity_wcg['Capacity'] = np.where(capacity_wcg['Capacity'] == 0.0, np.nan, capacity_wcg['Capacity'])
        capacity_wcg['Capacity'] = capacity_wcg['Capacity'].fillna(
            capacity_wcg.groupby('Work_Center_Group')['Capacity'].transform('mean'))
        capacity_wcg['Capacity'] = capacity_wcg['Capacity']*capacity_wcg['Net Production Rate']
        capacity_wcg = capacity_wcg.set_index('Work_Center')
        
        self.adjusted_Capacity = capacity_wcg['Capacity'].to_dict()
        self.workcenters_by_group = workcenters_by_group
        self.workcenters_in_groups = self.workcenters[self.workcenters.isin(workcenters_by_group.keys())]

def frozen_window(self):
    committed_flexible = self.Committed_Period.copy().reset_index()
    committed_flexible["Period Number"] = committed_flexible["Committed Period"].apply(
        lambda x : self.period_numbers[x]
    )
    
    committed_flexible = committed_flexible.rename_axis(index='index')

    committed_flexible = committed_flexible[
        committed_flexible["Project"].isin(self.committed_flexible)
    ]

    #! TODO: it should be in the input data - "No Frozen Window" or specific period to say yes

    self.frozen_window_end_period = "2021 Q1"
    
    if self.frozen_window_end_period == "No Frozen Window":
        self.flexible_projects_inside_window = pd.Index([], name="Flexible Inside Window")
        self.frozen_window_periods = pd.Index([], name = 'periods')
    else:
        self.frozen_window_end = self.period_numbers[self.frozen_window_end_period]
        self.flexible_projects_inside_window = pd.Index(committed_flexible[
            committed_flexible["Period Number"] <= self.frozen_window_end
            ], name = "Flexible Inside Window")
        frozen_window_periods = self.period_numbers.copy()
        frozen_window_periods.name = "Period Number"
  
        self.frozen_window_periods = pd.Index(frozen_window_periods
            [frozen_window_periods <= self.frozen_window_end].index
        )

    self.committed_fixed = self.committed_fixed.append(
        self.flexible_projects_inside_window
    )

    self.flexible_projects_out_window = pd.Index(committed_flexible[
        ~committed_flexible.isin(self.flexible_projects_inside_window)
    ], name = "Flexible Outside Window")

    self.unfrozen_window_periods = self.periods[
        ~self.periods.isin(self.frozen_window_periods)
    ]


def flexible_investment_cost(self):
    """
    What is flexible investment cost?
     - This function will calculate the cost of moving a committed project to be executed sooner or later
     - It is necessary for the MIP model to move the executions later unless needed, and it also helps avoid symmetry
    
    What is discount factor?
     - The calculation used discount factor as it is a shared parameter related to cost calculations

    What is expected?
     - This function will return a series indexed by projects and periods, with values of costs
     - The cost in the committed period of the project is 0
     - The sooner of the project execution, the more cost
    """
    fic = self.pdf_committed[
        self.pdf_committed.index.isin(self.committed_flexible)
        ][["Initial Cost","Committed Period"]]
    fic["Initial Cost"] = fic["Initial Cost"].apply(lambda x : 1000 if x == 0 else x)
    discount_factor = pd.Series(self.DISCOUNT_FACTOR)
    flexible_inv_cost = []
    for project in self.committed_flexible:
        flexible_inv_cost.append(pd.concat([_get_cost(
            discount_factor,
            fic.loc[project,"Initial Cost"],
            fic.loc[project, "Committed Period"]
        )],keys=[project]))
    try:
        self.flexible_inv_cost = pd.concat(flexible_inv_cost)
    except: 
        self.flexible_inv_cost = pd.Series(dtype='float64')
        
def _get_cost(discount_factor, base_cost, period):
    base_factor = discount_factor[period]
    cost = base_cost*discount_factor/base_factor - base_cost
    return cost

def get_max_capacity_increase(workcenters, projects, ip_set, PHI):
    max_increase_dict = {}
    for i in workcenters:
        max_increase_dict[i] = max([PHI[p] for p in projects if (i,p) in ip_set])
    
    max_increase = pd.Series(max_increase_dict)
    return max_increase

def get_bom_cons_precomputes(self):
    """return Valid_Qty_Per and valid_j_set to be used in BOM constraint in mip_model.py"""
    # 0. Get needed data
    periods = list(self.periods)
    # create bom df from these series: Qty_Per, UOM, Scrap_Rate, Valid_From_Bom, Valid_Until_Bom.
    bom_df = pd.DataFrame({
        "Qty Per": self.Qty_Per,
        "UOM": self.UOM,
        "Scrap Rate": self.Scrap_Rate,
        "Valid From": self.Valid_From_Bom,
        "Valid Until": self.Valid_Until_Bom
    })
    bom_df.index.names = ["Component", "Product"]
    bom_df = bom_df.reset_index()
    bom_df = bom_df.dropna()
    bom_df = bom_df[(bom_df['Product'].isin(self.items)) & (bom_df['Component'].isin(self.items))]

    # 1. valid bom indexed by j, k, and t
    bom_extended = pd.concat([bom_df] * len(periods)).reset_index(drop=True)
    bom_extended = bom_extended.sort_values(
        by=["Component", "Product"], ascending=[True, False]
    )
    bom_extended["Periods"] = periods * len(bom_df)
    bom_extended = bom_extended.reset_index(drop=True)

    bom_extended["Qty Per"] = np.where(
        (bom_extended["Periods"] < bom_extended["Valid From"])
        | (bom_extended["Periods"] > bom_extended["Valid Until"]),
        0,
        bom_extended["Qty Per"],
    )

    qty = bom_extended[["Component", "Product", "Periods", "Qty Per"]].copy()
    Valid_Qty_Per = (qty.set_index(["Component", "Product", "Periods"])["Qty Per"]).to_dict()

    # 2. valid set of all j associated to k
    valid_j_set = bom_df.groupby("Component")["Product"].apply(list).to_dict()
    return Valid_Qty_Per, valid_j_set


def precompute(self):
    """execute all precomputations"""
    # # 0. Remove inactived Work Centers
    self.workcenters = self.workcenters[
        self.WC_Status != "Inactive"
    ]

    # # 1. Reconstruct dataframes from series
    # project dataframe
    pdf, pdf_committed = get_projects_df(self)

    self.pdf = pdf
    self.pdf_committed = pdf_committed
    # validation table dataframe
    vtdf = get_validation_table_df(self)
    self.vtdf = vtdf
    # Get Max_ProductionRate
    Max_ProductionRate, Max_ProductionRate_BigM = get_max_productionrate(self, self.vtdf)
    self.Max_ProductionRate = Max_ProductionRate
    self.Max_ProductionRate_BigM = Max_ProductionRate_BigM

    # # 2. Precompute series by dataframes
    # series related to project dataframe

    self.Committed_Projects = get_committed_projects(self, pdf_committed)
    # self.it_set = get_it_set(pdf)
    get_projects_subsets(self, pdf, pdf_committed)

    # compute biggest project for each wc
    self.Max_Capacity_Increase = get_max_capacity_increase(
        self.workcenters,
        self.projects,
        self.ip_set,
        self.PHI
    )

    # series related to validation table dataframe
    (
        self.ijs_set,
        self.PRate,
        self.Valid_From_Qual,
        self.Valid_Until_Qual,
    ) = get_validation_table_series(vtdf)

    self.Last_Step, self.StepsOf = get_production_steps(vtdf)

    # # 3. Other precompute entities
    self.Availability[self.Availability < 1.0] = 1.0

    # AHTY-349 get component lead time
    self.Component_Lead_Time = get_component_lead_time(
        self.Production_LT_Days, periods_per_year=self.PERIODS_PER_YEAR
    )

    rescale_sets(self)

    ## Frozen Window
    frozen_window(self)

    self.DISCOUNT_RATE_PER_PERIOD = (
        (1 + self.ANNUAL_DISCOUNT_RATE / 100) ** (1 / self.PERIODS_PER_YEAR)
    ) - 1

    self.DISCOUNT_FACTOR = {
        t: (1 / (1 + self.DISCOUNT_RATE_PER_PERIOD) ** self.period_numbers[t])
        for t in self.periods
    }

    ## Flexible Investment Cost
    flexible_investment_cost(self)

    self.WEEKS_PER_YEAR = 52

    self.WEEKS_PER_PERIOD = self.WEEKS_PER_YEAR / self.PERIODS_PER_YEAR
    self.S0 = (self.S0 / self.WEEKS_PER_PERIOD).apply(lambda s: s // 1)
    self.year_from_period = self.periods.to_series().apply(lambda x : x[:4])
    self.years = self.year_from_period.unique()

    self.ijt_set = []
    for (i,j) in self.ij_set:
        start = self.period_numbers[self.Valid_From_Qual[i, j]]
        stop = self.period_numbers[self.Valid_Until_Qual[i, j]]
        for t in self.periods:
            n = self.period_numbers[t]
            if start <= n <= stop:
                self.ijt_set.append((i, j, t))

    ij_dict_tmp = self.Validation_Status.reset_index()[["Work_Center","Item"]]
    ij_dict = {}
    for i in self.workcenters:
        ij_dict[i] = []
        for j in ij_dict_tmp[ij_dict_tmp["Work_Center"] == i]["Item"]:
            ij_dict[i].append(j)
    self.ij_dict = ij_dict
    self.ij_set_cq = self.Validation_Status[self.Validation_Status == "Capable"].index
    self.ijt_set_cq = []

    # Generate jts_dict
    df_ijs = pd.DataFrame(self.ijs_set, columns=['i', 'j', 's'])
    self.js_dict = df_ijs.groupby(['j','s']).agg(lambda x: set(x.to_list()))['i'].to_dict()
    df_ijt = pd.DataFrame(self.ijt_set, columns=['i', 'j', 't'])
    merged = pd.merge(df_ijt, df_ijs, on=['i','j'])#.drop_duplicates()
    self.jts_dict = merged.groupby(['j','t','s']).agg(lambda x: x.to_list())['i'].to_dict()
    
    for (i, j) in self.ij_set_cq:
        start = self.period_numbers[self.Valid_From_Qual[i, j]]
        stop = self.period_numbers[self.Valid_Until_Qual[i, j]]
        for t in self.periods:
            n = self.period_numbers[t]
            if start <= n <= stop:
                self.ijt_set_cq.append((i, j, t))


    # # get priority costs
    wc_priority_cost = {}
    for t in self.periods:
        for j in self.items:
            if j in self.StepsOf:
                for s in self.StepsOf[j]:
                    parallel_wcs = [(i,self.Priority[(i, j)]) for i in self.jts_dict.get((j,t,s), [])]
                    priorities_number = len(set(aux[1] for aux in parallel_wcs))
                    priority = np.argsort(list(set([x[1] for x in parallel_wcs])))
                    costs = np.linspace(0, 1, priorities_number)*1e-2
                    for (idx,pty) in enumerate(set(aux[1] for aux in parallel_wcs)):
                        for i in set(aux[0] for aux in parallel_wcs if aux[1] == pty):
                            wc_priority_cost[(i, j, t)] = costs[priority[idx]]
    self.wc_priority_cost = pd.Series(wc_priority_cost)

    # Default Penalizations
    self.allocation_deviation_penalization = round(self.Standard_Cost*(1+self.Profit/100)*0.2, 1)

    # get bom constraints precomputes
    self.Valid_Qty_Per, self.valid_j_set = get_bom_cons_precomputes(self)

    return None
