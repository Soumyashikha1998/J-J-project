# Copyright: Johnson & Johnson Digital & Data Science (JJDDS)
#
# This file contains trade secrets of JJDDS. No part may be reproduced or transmitted in any
# form by any means or for any purpose without the express written permission of JJDDS.
#
# Purpose:  Polaris Application Source

import os
from datetime import datetime

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils.dataframe import dataframe_to_rows


def compute_capacity_utilization(
    workcenters, periods, Capacity, Availability, Min_Utilization, Max_Utilization
):
    if not workcenters.empty:
        capacity_calc = pd.Series(
            {(i, t): Capacity[i] for i in workcenters for t in periods}
        )
        availability_series = pd.Series(
            {
                (i, t): Availability.get((i, t), 1.0)
                for i in workcenters
                for t in periods
            }
        )
        min_utilization_series = pd.Series(
            {
                (i, t): Min_Utilization.get((i, t), 0.0)
                for i in workcenters
                for t in periods
            }
        )
        max_utilization_series = pd.Series(
            {
                (i, t): Max_Utilization.get((i, t), 1.0)
                for i in workcenters
                for t in periods
            }
        )

        available_capacity_calc = capacity_calc.multiply(availability_series)
        min_capacity_calc = available_capacity_calc.multiply(min_utilization_series)
        max_capacity_calc = available_capacity_calc.multiply(max_utilization_series)

    else:
        capacity_calc = pd.Series({("No_WC", "No_Period"): 0.0})
        available_capacity_calc = capacity_calc
        min_capacity_calc = capacity_calc
        max_capacity_calc = capacity_calc

    return capacity_calc, available_capacity_calc, min_capacity_calc, max_capacity_calc


# def update_indexes(app):
#     app.families = pd.Index(app.Family.unique())
#     app.platforms = pd.Index(app.Platform.unique())
#     app.commodities = pd.Index(app.Commodity.unique())
#     app.suppliers = pd.Index(app.Supplier_Name.unique())
#     app.wc_groups = pd.Index(app.Work_Center_Group.unique())
#     app.item_groups = pd.Index(
#         app.Dual_Sourcing.index.get_level_values("Group").unique()
#     )


# def read_query(tablename, condition, connection, delete_prefix=True):
#     query_tablename = "polaris_" + tablename

#     query = f"""
#     SELECT * from "{query_tablename}"
#     WHERE {condition}
#     """

#     df = pd.read_sql_query(query, connection)

#     if tablename == "projects_output":
#         tablename = "project_table"
#     elif tablename == "production_detail_output":
#         tablename = "production_detail_table"

#     if delete_prefix:
#         df.columns = df.columns.str.replace(f"{tablename}_", "")

#     return df


# def add_scenario_names(table, df):
#     table = table.merge(df, left_on="scenario_id", right_on="id").drop(["id"], axis=1)
#     table.insert(loc=1, column="Scenario Name", value=table["name"])
#     table = table.drop(["name"], axis=1)
#     table = table.drop(["scenario_id"], axis=1)

#     return table


# def update_column_names(uom_list=None, vol_list=None):
#     if uom_list:
#         df_item_summary_table = uom_list[0]
#         df_work_center_summary_table = uom_list[1]
#         df_work_center_summary_hr_table = uom_list[2]
#         df_projects_output = uom_list[3]
#         df_production_detail_output = uom_list[4]
#         df_cost_summary_table = uom_list[5]

#         item_summary_table = df_item_summary_table.rename(
#             columns={
#                 "items": "Item",
#                 "Description": "Description",
#                 "Platform": "Platform",
#                 "Family": "Family",
#                 "Commodity": "Commodity",
#                 "Item_Group": "Item Group",
#                 "Item_Type": "Item Type",
#                 "periods": "Period",
#                 "slist": "Scenario",
#                 "Forecast": "Forecast (ea/wk)",
#                 "Propagated_Demand": "Propagated Demand (ea/wk)",
#                 "Demand_Fulfillment": "Demand Fulfillment (ea/wk)",
#                 "Demand_Fulfillment_LT": "Demand Fulfillment LT (ea/wk)",
#                 "Demand_Requirement": "Demand Requirement (ea/wk)",
#                 "Demand_Requirement_Attendance": "Demand Requirement Attendance",
#                 "Production_Requirement": "Production Requirement (ea/wk)",
#                 "Production": "Production (ea/wk)",
#                 "Discrete_PO": "Discrete PO (ea/wk)",
#                 "New_Discrete_PO": "New Discrete PO (ea/wk)",
#                 "Production_Requirement_Attendance": "Production Requirement Attendance",
#                 "Total_Equivalent_Capacity": "Total Equivalent Capacity (ea/wk)",
#                 "Reference_Utilization": "Reference Utilization",
#                 "Min_Sourcing": "Min Sourcing",
#                 "Sourcing": "Sourcing",
#                 "Missed_Demand": "Missed Demand (ea/wk)",
#                 "Inventory_Investment": "Inventory Investment (ea)",
#                 "Inventory": "Inventory (ea)",
#                 "Min_Inventory": "Min Inventory (ea)",
#                 "Target_Inventory": "Target Inventory (ea)",
#                 "Max_Inventory": "Max Inventory (ea)",
#                 "Inventory_DOS": "Inventory DOS",
#                 "Min_Inventory_DOS": "Min Inventory DOS",
#                 "Target_Inventory_DOS": "Target Inventory DOS",
#                 "Max_Inventory_DOS": "Max Inventory DOS",
#                 "Revenue": "Revenue ($)",
#                 "Inventory_Investment_Cost": "Inventory Investment Cost ($)",
#                 "Holding_Cost": "Holding Cost ($)",
#                 "Missed_Sales": "Missed Sales ($)",
#             }
#         )

#         work_center_summary_table = df_work_center_summary_table.rename(
#             columns={
#                 "workcenters": "Work Center",
#                 "WC_Type": "WC Type",
#                 "periods": "Period",
#                 "WC_Group": "WC Group",
#                 "Supplier": "Supplier",
#                 "Asset_Owner": "Asset Owner",
#                 "slist": "Scenario",
#                 "Initial_Capacity": "Initial Capacity",
#                 "Capacity_Increase": "Capacity Increase",
#                 "Potential_Increase": "Potential Increase",
#                 "Committed_Increase": "Committed Increase",
#                 "Expansion_Cost": "Expansion Cost",
#                 "Capacity_Unit_WC": "Capacity Unit",
#                 "Available_Capacity": "Available Capacity",
#                 "Used_Capacity": "Used Capacity",
#                 "Installed_Capacity_Target": "Installed Capacity Target",
#                 "Min_Utilization": "Min Utilization",
#                 "Max_Utilization": "Max Utilization",
#                 "Utilization": "Utilization",
#                 "Availability": "Availability",
#             }
#         )

#         work_center_summary_hr_table = df_work_center_summary_hr_table.rename(
#             columns={
#                 "workcenters": "Work Center",
#                 "WC_Type_hr": "WC Type",
#                 "periods": "Period",
#                 "WC_Group_hr": "WC Group",
#                 "Supplier_hr": "Supplier",
#                 "Asset_Owner_hr": "Asset Owner",
#                 "slist": "Scenario",
#                 "Initial_Capacity_hr": "Initial Capacity",
#                 "Capacity_Increase_hr": "Capacity Increase",
#                 "Potential_Increase_hr": "Potential Increase",
#                 "Committed_Increase_hr": "Committed Increase",
#                 "Expansion_Cost_hr": "Expansion Cost",
#                 "Capacity_hr": "Capacity",
#                 "Capacity_Unit_WC_hr": "Capacity Unit",
#                 "Available_Capacity_hr": "Available Capacity",
#                 "Used_Capacity_hr": "Used Capacity",
#                 "Installed_Capacity_Target_hr": "Installed Capacity Target",
#                 "Min_Utilization_hr": "Min Utilization",
#                 "Max_Utilization_hr": "Max Utilization",
#                 "Utilization_hr": "Utilization",
#                 "Availability_hr": "Availability",
#             }
#         )

#         projects_table = df_projects_output.rename(
#             columns={
#                 "Work_Center": "Work Center",
#                 "projects": "Project",
#                 "slist": "Scenario",
#                 "Status": "Status",
#                 "Execution": "Execution",
#                 "Planned_Period": "Planned Period",
#                 "Lead_Time": "Lead Time (periods)",
#                 "periods": "Production Qualification Period",
#                 "Capacity_Increase_ea": "Capacity Increase (ea/wk)",
#                 "Capacity_Increase_hr": "Capacity Increase (hr/wk)",
#                 "Initial_Capacity_ea": "Initial Capacity (ea/wk)",
#                 "Initial_Capacity_hr": "Initial Capacity (hr/wk)",
#                 "Cost": "Cost ($)",
#                 "Project_Type": "Project Type",
#                 "etypes": "Execution Type",
#             }
#         )

#         production_detail_table = df_production_detail_output.rename(
#             columns={
#                 "workcenters": "Work Center",
#                 "WC_Type": "WC Type",
#                 "WC_Group": "WC Group",
#                 "Supplier": "Supplier",
#                 "Asset_Owner": "Asset Owner",
#                 "slist": "Scenario",
#                 "periods": "Period",
#                 "items": "Item",
#                 "Description": "Description",
#                 "Platform": "Platform",
#                 "Family": "Family",
#                 "Commodity": "Commodity",
#                 "Item_Group": "Item Group",
#                 "Item_Type": "Item Type",
#                 "Production_ea": "Production (ea/wk)",
#                 "Production_hr": "Production (hr/wk)",
#                 "Equivalent_Capacity_Consumed": "Equivalent Capacity Consumed (ea/wk)",
#                 "Net_Production_Rate": "Net Production Rate (ea/hr)",
#                 "Yield": "Yield",
#                 "Step": "Step",
#                 "Min_Allocation": "Min Allocation",
#                 "Allocation": "Allocation",
#                 "Min_Sourcing": "Min Sourcing",
#                 "Sourcing": "Sourcing",
#             }
#         )

#         cost_summary_table = df_cost_summary_table.rename(
#             columns={
#                 "slist": "Scenario",
#                 "periods": "Period",
#                 "Sales": "Sales (k$)",
#                 "Capital_Investment_Cost": "Capital Investment Cost (k$)",
#                 "Inventory_Investment_Cost": "Inventory Investment Cost (k$)",
#                 "Holding_Cost": "Holding Cost (k$)",
#             }
#         )

#         return (
#             item_summary_table,
#             work_center_summary_table,
#             work_center_summary_hr_table,
#             projects_table,
#             production_detail_table,
#             cost_summary_table,
#         )

#     if vol_list:
#         df_vol_platform = vol_list[0]
#         df_vol_family = vol_list[1]
#         df_vol_commodity = vol_list[2]
#         df_vol_FG = vol_list[3]
#         df_vol_supplier = vol_list[4]
#         df_vol_WC_group = vol_list[5]
#         df_vol_WC = vol_list[6]
#         df_vol_comp = vol_list[7]

#         report_volume_platform = df_vol_platform.rename(
#             columns={
#                 "slist": "Scenario",
#                 "Period": "periods",
#                 "Volume_platform": "Volume (ea)",
#                 "Spend_platform": "Spend",
#                 "Sales_platform": "Sales",
#                 "Total_Spend_platform": "% Total Spend",
#                 "Total_Sales_platform": "% Total Sales",
#             }
#         )

#         report_volume_family = df_vol_family.rename(
#             columns={
#                 "slist": "Scenario",
#                 "periods": "Period",
#                 "Volume_family": "Volume (ea)",
#                 "Spend_family": "Spend",
#                 "Sales_family": "Sales",
#                 "Total_Spend_family": "% Total Spend",
#                 "Total_Sales_family": "% Total Sales",
#             }
#         )

#         report_volume_commodity = df_vol_commodity.rename(
#             columns={
#                 "slist": "Scenario",
#                 "periods": "Period",
#                 "Volume_commodity": "Volume (ea)",
#                 "Spend_commodity": "Spend",
#                 "Sales_commodity": "Sales",
#                 "Total_Spend_commodity": "% Total Spend",
#                 "Total_Sales_commodity": "% Total Sales",
#             }
#         )

#         report_volume_FG = df_vol_FG.rename(
#             columns={
#                 "slist": "Scenario",
#                 "periods": "Period",
#                 "Finished_Good": "Item",
#                 "Volume_FG": "Volume (ea)",
#                 "Spend_FG": "Spend",
#                 "Sales_FG": "Sales",
#                 "Total_Spend_FG": "% Total Spend",
#                 "Total_Sales_FG": "% Total Sales",
#             }
#         )

#         report_volume_supplier = df_vol_supplier.rename(
#             columns={
#                 "slist": "Scenario",
#                 "periods": "Period",
#                 "Volume_supplier": "Volume (ea)",
#                 "Spend_supplier": "Spend",
#                 "Sales_supplier": "Sales",
#                 "Total_Spend_supplier": "% Total Spend",
#                 "Total_Sales_supplier": "% Total Sales",
#             }
#         )

#         report_volume_WC_group = df_vol_WC_group.rename(
#             columns={
#                 "slist": "Scenario",
#                 "periods": "Period",
#                 "Work_Center_Group": "WC Group",
#                 "Volume_WC_Group": "Volume (ea)",
#                 "Spend_WC_Group": "Spend",
#                 "Sales_WC_Group": "Sales",
#                 "Total_Spend_WC_Group": "% Total Spend",
#                 "Total_Sales_WC_Group": "% Total Sales",
#             }
#         )

#         report_volume_WC = df_vol_WC.rename(
#             columns={
#                 "slist": "Scenario",
#                 "periods": "Period",
#                 "Work_Center": "Work Center",
#                 "Volume_WC": "Volume (ea)",
#                 "Spend_WC": "Spend",
#                 "Sales_WC": "Sales",
#                 "Total_Spend_WC": "% Total Spend",
#                 "Total_Sales_WC": "% Total Sales",
#             }
#         )

#         report_volume_comp = df_vol_comp.rename(
#             columns={
#                 "slist": "Scenario",
#                 "periods": "Period",
#                 "Component": "Item",
#                 "Volume_comp": "Volume (ea)",
#                 "Spend_comp": "Spend",
#                 "Sales_comp": "Sales",
#                 "Total_Spend_comp": "% Total Spend",
#                 "Total_Sales_comp": "% Total Sales",
#             }
#         )

#         return (
#             report_volume_platform,
#             report_volume_family,
#             report_volume_commodity,
#             report_volume_FG,
#             report_volume_supplier,
#             report_volume_WC_group,
#             report_volume_WC,
#             report_volume_comp,
#         )
