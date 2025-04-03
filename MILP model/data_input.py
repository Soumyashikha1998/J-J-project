
'''
Data should be complete, because there is no validation step

'''
#%%

import pandas as pd


class SCMData:
    def __init__(self, filename=None, use_discrete_po = "No"):
        # default decimal number when rounding input
        self.DECIMAL_NUMBER = 2
        self.forecast_considered = "Only Nominal"
        self.slist = pd.Index(["nominal"])

        # Reading file

        self.xl = pd.read_excel(filename, sheet_name=None, engine="openpyxl")

        # Reading options for simulation
        print(f'Options is Loading from Excel..\n')
        self.options = self.xl["Options"].copy()
        self.options.drop("Unnamed: 0", axis=1, inplace=True)
        print(f'Options is successfully Loaded from Excel..\n{self.options}\n')
        self.options.columns = self.options.iloc[1]
        self.options = self.options[2:]
        self.options.set_index("Options", inplace=True)

    def load_data(self):

        # Reading options for simulation
        self.max_simulations = int(
            self.options.loc["Maximum Number of Simulations", "Value"]
        )
        self.SELECTED_UNCERTAINTY_COVERAGE = str(
            self.options.loc["Uncertainty Coverage (%)", "Value"]
        )
        self.SCENARIO_SELECTION_MODE = str(
            self.options.loc["Scenario Selection Mode", "Value"]
        )

        # Upload Dataframes for data preview in the appplication
        # Each column in Dataframes has to be renamed as declared in application.py

        #Reading Data From Excel

        # Upload Dataframes for data preview in the appplication
        # Each column in Dataframes has to be renamed as declared in application.py

        self.instructions_index = pd.Index(
            [x for x in range(self.xl["Simulation Instructions"].shape[0])]
        )
        instructions_df = pd.DataFrame(
            data=self.xl["Simulation Instructions"],
            index=self.instructions_index,
        )
        instructions_df = instructions_df[
            instructions_df.columns.drop(instructions_df.filter(regex="Unnamed:"))
        ]
        instructions_df.columns = instructions_df.columns.str.replace(" ", "_")

        self.relations_index = pd.Index([x for x in range(self.xl["Relations"].shape[0])])

        relation_df = pd.DataFrame(
            data=self.xl["Relations"],
            index=self.relations_index,
        )
        relation_df = relation_df[
            relation_df.columns.drop(relation_df.filter(regex="Unnamed:"))
        ]
        relation_df.columns = relation_df.columns.str.replace(" ", "_")

        items_df = self.xl["Items"].set_index("Item")
        items_df = items_df[items_df.columns.drop(items_df.filter(regex="Unnamed:"))]
        items_df.columns = items_df.columns.str.replace(" ", "_")

        workcenter_df = self.xl["Work Centers"].drop_duplicates().set_index(["Work Center"])
        workcenter_df = workcenter_df[
            workcenter_df.columns.drop(workcenter_df.filter(regex="Unnamed:"))
        ]
        workcenter_df.columns = workcenter_df.columns.str.replace(" ", "_")

        wc_params_df = self.xl["Work Center Parameters"].drop_duplicates().copy()
        wc_params_df = wc_params_df[wc_params_df.columns.drop(wc_params_df.filter(regex="Unnamed:"))]

        avail_df = wc_params_df[wc_params_df['Parameter'] == 'Availability'].drop(
                columns = ["Parameter"]
            )
        min_utilization_df = wc_params_df[wc_params_df['Parameter'] == 'Min Utilization'].drop(
                columns = ["Parameter"]
            )
        max_utilization_df = wc_params_df[wc_params_df['Parameter'] == 'Max Utilization'].drop(
                columns = ["Parameter"]
            )

        avail_df = (
            pd.melt(avail_df, ["Work Center"])
            .rename(columns={"variable": "Period", "value": "Availability"})
            .set_index(["Work Center", "Period"])
        )
        min_utilization_df = (
            pd.melt(min_utilization_df, ["Work Center"])
            .rename(columns={"variable": "Period", "value": "Min_Utilization"})
            .set_index(["Work Center", "Period"])
        )
        max_utilization_df = (
            pd.melt(max_utilization_df, ["Work Center"])
            .rename(columns={"variable": "Period", "value": "Max_Utilization"})
            .set_index(["Work Center", "Period"])
        )

        projects_df = self.xl["Projects"].set_index(["Project"])
        projects_df = projects_df[
            projects_df.columns.drop(projects_df.filter(regex="Unnamed:"))
        ]
        projects_df.columns = projects_df.columns.str.replace(" ", "_")
        projects_df.index = projects_df.index.fillna("UNKNOWN_PROJECT")

        bom_df = self.xl["BOM"].set_index(["Component", "Product"])
        bom_df = bom_df[bom_df.columns.drop(bom_df.filter(regex="Unnamed:"))]
        bom_df.columns = bom_df.columns.str.replace(" ", "_")

        demand = self.xl["Forecast"].copy()
        demand = demand[demand.columns.drop(demand.filter(regex="Unnamed:"))]
        demand_df = (
            pd.melt(demand, ["Product"])
            .rename(columns={"variable": "Period", "value": "Demand"})
            .set_index(["Product", "Period"])
        )
        demand_df.columns = demand_df.columns.str.replace(" ", "_")

        # Discrete POs
        # This part gives the possibility of having or not the sheet. If it is not present, here we create an empty sheet.
        # It will be useful to keep constant the rest of the steps.
        # app.use_discrete_po = "Yes"
        
        #! TODO: this info is part of the input
        self.use_discrete_po = "No"
        
        if self.use_discrete_po == "No": # in this case the data that is available in the template is skipped.
            column_names = list(
                [col for col in demand.columns if "Unnamed" not in col]
            )
            po = pd.DataFrame(columns=column_names)
            po.rename(columns={"Product": "Item"}, inplace=True)
        else:
            # if Yes: 1. take the existing DPO, 2. otherwise, update the entity app.discrete_po_not_present
            if "Discrete PO" in self.xl: 
                po = self.xl["Discrete PO"].copy()
                po = po[po.columns.drop(po.filter(regex="Unnamed:"))]
            else: # we create an empty sheet - it should give
                column_names = list(
                    [col for col in demand.columns if "Unnamed" not in col]
                )
                po = pd.DataFrame(columns=column_names)
                po.rename(columns={"Product": "Item"}, inplace=True)

        discrete_po_df = (
            pd.melt(po, ["Item"])
            .rename(columns={"variable": "Period", "value": "Discrete_PO"})
            .set_index(["Item", "Period"])
        )
        discrete_po_df.columns = discrete_po_df.columns.str.replace(" ", "_")

        cap_target = self.xl["Capacity Target"].copy()
        cap_target = cap_target[cap_target.columns.drop(cap_target.filter(regex="Unnamed:"))]
        cap_target_df = (
            pd.melt(cap_target, ["Family"])
            .rename(columns={"variable": "Year", "value": "Target"})
        )
        cap_target_df["Year"] = cap_target_df["Year"].astype('str')
        cap_target_df = cap_target_df.set_index(["Family", "Year"])

        dual_sourcing = self.xl["Dual Sourcing"].copy()
        dual_sourcing = dual_sourcing[dual_sourcing.columns.drop(dual_sourcing.filter(regex="Unnamed:"))]
        dual_sourcing = (
            pd.melt(dual_sourcing, ["Group", "Component"])
            .rename(columns={"variable": "Period", "value": "Allocation"})
        ).set_index(["Group", "Component", "Period"])

        self.qualifications_df = self.xl["Qualifications"].copy()
        self.qualifications_df = self.qualifications_df[
            self.qualifications_df.columns.drop(
                self.qualifications_df.filter(regex="Unnamed:")
            )
        ]
        self.qualifications_df.columns = self.qualifications_df.columns.str.replace(
            " ", "_"
        )

        yield_df = self.xl["Yield"].copy()
        yield_df = yield_df[yield_df.columns.drop(yield_df.filter(regex="Unnamed:"))]
        yield_validation = (
            pd.melt(yield_df, ["Item", "Work Center"])
            .rename(columns={"variable": "Period", "value": "Yield"})
            .set_index(["Work Center", "Item", "Period"])
        )

        self.qualifications_df.columns = self.qualifications_df.columns.str.replace(
            " ", "_"
        )
            # Indices initialization
        component_label = "C"
        finish_good_label = "FG"
        self.workcenters = pd.Index(
            self.xl["Work Centers"]["Work Center"].drop_duplicates()
        )
        self.components = pd.Index(
            items_df[items_df["Type"] == component_label].index.drop_duplicates()
        )
        self.products = pd.Index(
            items_df[items_df["Type"] == finish_good_label].index.drop_duplicates()
        )
        self.projects = pd.Index(self.xl["Projects"]["Project"].drop_duplicates())
        self.items = self.products.union(self.components)
        self.periods = pd.Index(list(demand.columns[1:]))
        self.families = pd.Index(items_df["Family"].unique())
        self.cap_target_years = pd.Index(cap_target_df.index.get_level_values(1).drop_duplicates())
        self.platforms = pd.Index(items_df["Platform"].unique())
        self.commodities = pd.Index(items_df["Commodity"].unique())
        self.suppliers = pd.Index(workcenter_df["Supplier_Name"].unique())
        self.wc_groups = pd.Index(workcenter_df["Work_Center_Group"].unique())
        self.item_groups = pd.Index(dual_sourcing.index.get_level_values("Group").unique())
        self.periods_window = pd.Index(["No Frozen Window"]+list(demand.columns[1:]))
        self.year_from_period = self.periods.to_series().apply(lambda x : x[:4])
        self.years = pd.Index(self.year_from_period.unique())

        # Optimization model parameters
        self.HOLDING_COST = float(self.options.loc["Annual Holding Cost (%)", "Value"])
        self.PERIODS_PER_YEAR = int(self.options.loc["Periods Per Year", "Value"])
        self.DEPRECIATION_YEARS = int(self.options.loc["Depreciation (years)", "Value"])
        self.ANNUAL_BUDGET = float(self.options.loc["Annual Budget ($M)", "Value"])
        self.forecast_considered = str(self.options.loc["Forecast Considered", "Value"])
        self.ESF_ALPHA = float(self.options.loc["Risk Analysis Threshold (%)", "Value"])
        self.default_QLT = int(self.options.loc["Default Qualification Lead Time (periods)", "Value"])
        self.default_Cost = int(self.options.loc["Default Qualification Cost", "Value"])
        self.ANNUAL_DISCOUNT_RATE = float(self.options.loc["Annual Discount Rate (%)", "Value"])

        self.Profit = float(self.options.loc["Profit (%)", "Value"])

        # Qualifications Table Haokun

        # Replacing Available Status by 1, Capable by 2

        self.Item_VT = self.qualifications_df["Item"]
        self.Work_Center_VT = self.qualifications_df["Work_Center"]

        self.Net_Production_Rate = self.qualifications_df.set_index(["Work_Center", "Item"])["Net_Production_Rate"]
        self.Step = self.qualifications_df.set_index(["Work_Center", "Item"])["Step"]
        self.Min_Allocation = self.qualifications_df.set_index(["Work_Center", "Item"])["Min_Allocation_%"]
        self.Priority = self.qualifications_df.set_index(["Work_Center", "Item"])["Priority"]

        self.Validation_Status = self.qualifications_df.set_index(["Work_Center", "Item"])["Status"]

        self.Valid_From_Qual = self.qualifications_df.set_index(["Work_Center", "Item"])["Valid_From"]
        self.Valid_Until_Qual = self.qualifications_df.set_index(["Work_Center", "Item"])["Valid_Until"]
        self.QLT = self.qualifications_df.set_index(["Work_Center", "Item"])["Qualification_Lead_Time_(periods)"]
        self.QCost = self.qualifications_df.set_index(["Work_Center", "Item"])["Qualification_Cost"]

        self.qualifications_df["Valid"] = 1  # - Haokun
        # todo: move ij_set to precompute
        self.ij_set = self.qualifications_df.set_index(["Work_Center", "Item"])["Valid"]
        # self.ij_set = self.ij_set.replace("Available", int(float(1)))
        # self.ij_set = self.ij_set.replace("Capable", int(float(0)))

        #  Items
        self.Item = items_df.index.to_series()
        self.Description = items_df["Description"]
        self.Platform = items_df["Platform"]
        self.Family = items_df["Family"]
        self.Commodity = items_df["Commodity"]
        self.Type = items_df["Type"]
        self.Status = items_df["Status"]
        self.Category = items_df["Category"]
        self.Standard_Cost = items_df["Standard_Cost"]
        self.S0 = items_df["Initial_Inventory"]
        self.Max_Inv_DOS = items_df["Maximum_Inventory_DOS"]
        self.EP = items_df["Earliest_Production"]
        self.Production_LT_Days = items_df["Production_LT_(days)"]
        self.SS_Days = items_df["SS_(days)"]
        self.Price = items_df["Price"]

        # Workcenters
        self.Workcenter = workcenter_df.index.to_series()
        self.Supplier_Name = workcenter_df["Supplier_Name"]
        self.Work_Center_Type = workcenter_df["Work_Center_Type"]
        self.Work_Center_Group = pd.Series(workcenter_df["Work_Center_Group"], dtype=str)
        self.Asset_Owner = workcenter_df["Asset_Owner"]
        self.WC_Status = workcenter_df["Status"]
        self.Capacity = workcenter_df["Capacity"]
        self.Capacity_Unit_WC = workcenter_df["Capacity_Unit"]
        self.Investment_Lead_Time = workcenter_df["Investment_Lead_Time_(periods)"]
        self.Capacity_Increment = workcenter_df["Capacity_Increment"]
        self.Increment_Cost = workcenter_df["Increment_Cost"]
        self.Balance_WC_Group = workcenter_df["Balance_WC_Group"]

        # Projects
        self.Project = projects_df.index.to_series()
        self.Reference_Work_Center = projects_df["Reference_Work_Center"]
        self.Project_Type = projects_df["Project_Type"]
        self.Project_Status = projects_df["Project_Status"].fillna("")
        self.Committed_Period = projects_df["Committed_Period"].fillna("")
        self.Capacity_Unit_PR = projects_df["Capacity_Unit"]
        self.Capacity_Increase = projects_df["Capacity_Increase"]
        self.Initial_Cost = projects_df["Initial_Cost"]
        self.Initial_Lead_Time = projects_df["Initial_Lead_Time_(periods)"]
        self.Maximum_Available = projects_df["Maximum_Available"]
        self.Subsequent_Cost = projects_df["Subsequent_Cost"]
        self.Subsequent_Lead_Time = projects_df["Subsequent_Lead_Time_(periods)"]

        # BOM
        self.kj_set = bom_df["Qty_Per"]
        self.Component_Bom = bom_df.reset_index().Component
        self.Product_Bom = bom_df.reset_index().Product
        self.Qty_Per = bom_df["Qty_Per"]
        self.Scrap_Rate = bom_df["Scrap_Rate"]
        self.UOM = bom_df["UOM"]
        self.Valid_From_Bom = bom_df["Valid_From"]
        self.Valid_Until_Bom = bom_df["Valid_Until"]

        # Demand
        self.nom = demand_df["Demand"]

        # Discrete POs
        self.Discrete_PO = discrete_po_df["Discrete_PO"]

        # Capacity Target
        self.Cap_Target = cap_target_df["Target"]

        # Sim Instructions
        self.Item_Sim = instructions_df["Item"]
        self.Uncertainty_Type = instructions_df["Uncertainty_Type"]
        self.Distribution = instructions_df["Distribution"]
        self.Min = instructions_df["Min"]
        self.Max = instructions_df["Max"]

        # Relations
        self.relation_from = relation_df["Item_From"]
        self.relation_to = relation_df["Item_To"]
        self.relation = relation_df["Relation"]

        # Validation
        self.Yield = yield_validation["Yield"].rename("Yield")

        # Work Center Parameters
        self.Availability = avail_df["Availability"]
        self.Min_Utilization = min_utilization_df["Min_Utilization"]
        self.Max_Utilization = max_utilization_df["Max_Utilization"]
        # Dual Sourcing Table
        self.Dual_Sourcing = dual_sourcing["Allocation"]

if __name__ == "__main__":
    pass

# %%
