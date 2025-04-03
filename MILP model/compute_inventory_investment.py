import numpy as np
import pandas as pd


def compute_inventory_investments(targets, x_jts, s_jts, s_e_jts, s_s_jts, scen_list=None):
    r_list = []

    if scen_list is None:        # Get parameters and variables for each scenario

        target = pd.DataFrame(
            {"Value": [targets[idx].value for idx in targets]},
            index=pd.MultiIndex.from_tuples(targets, names=["Item", "Period"]),
        )
        target_pre = target.groupby(level="Item").shift(1, fill_value=0)
        diff_target = target - target_pre

        # For Variables

        s_e = pd.DataFrame(
            {"Value": [s_e_jts[idx].value for idx in s_e_jts]},
            index=pd.MultiIndex.from_tuples(s_e_jts, names=["Item", "Period"]),
        )
        s_e_pre = s_e.groupby(level="Item").shift(1, fill_value=0)

        s_s = pd.DataFrame(
            {"Value": [s_s_jts[idx].value for idx in s_s_jts]},
            index=pd.MultiIndex.from_tuples(s_s_jts, names=["Item", "Period"]),
        )
        s_s_pre = s_s.groupby(level="Item").shift(1, fill_value=0)

        s = pd.DataFrame(
            {"Value": [s_jts[idx].value for idx in s_jts]},
            index=pd.MultiIndex.from_tuples(s_jts, names=["Item", "Period"]),
        )
        s_pre = s.groupby(level="Item").shift(1, fill_value=0)

        x = pd.DataFrame(
            {"Value": [x_jts[idx].value for idx in x_jts]},
            index=pd.MultiIndex.from_tuples(x_jts, names=["Item", "Period"]),
        )
        r_scen = calculation(s, s_pre, diff_target, s_s_pre,s_e_pre,x)
        # Append to results list
        r_list.append(r_scen['R'])
        rdf = pd.concat(r_list)
        rdf.name = 'Inventory Investment'

    else:
        for scen in scen_list:
            # Get parameters and variables for each scenario
            target = targets[targets.index.get_level_values('Scenario') == scen].to_frame().rename(columns={0: 'Solution'})
            target_pre = target.groupby(level='Item').shift(1, fill_value=0)
            diff_target = target - target_pre
            s_e = s_e_jts[s_e_jts.index.get_level_values('Scenario') == scen]
            s_e_pre = s_e.groupby(level='Item').shift(1, fill_value=0)
            s_s = s_s_jts[s_s_jts.index.get_level_values('Scenario') == scen]
            s_s_pre = s_s.groupby(level='Item').shift(1, fill_value=0)
            s = s_jts[s_jts.index.get_level_values('Scenario') == scen]
            s_pre = s.groupby(level='Item').shift(1, fill_value=0)
            x = x_jts[x_jts.index.get_level_values('Scenario') == scen]
            r_scen = calculation(s, s_pre, diff_target, s_s_pre,s_e_pre,x)
            # Append to results list
            r_list.append(r_scen['R'])
            rdf = pd.concat(r_list)
            rdf.name = 'Inventory Investment'

    return rdf

def calculation(s, s_pre, diff_target, s_s_pre,s_e_pre,x):
    # Rename columns after each object
    s.columns = pd.Index(['S'])
    s_pre.columns = pd.Index(['S_pre'])
    diff_target.columns = pd.Index(['Diff_target'])
    s_s_pre.columns = pd.Index(['S_s_pre'])
    s_e_pre.columns = pd.Index(['S_e_pre'])
    x.columns = pd.Index(['X'])

    # Concatenate objects to compute inventory investment
    df = pd.concat([s, s_pre, diff_target, s_s_pre, s_e_pre, x], axis=1)

    # Compute extra inventory
    df['extra_inv'] = np.maximum(
        0,
        df['S'] - df['S_pre'] - df['Diff_target'] - df['S_s_pre'] + df['S_e_pre']
    )

    # Find inventory investment using production variable
    df['R'] = round(np.minimum(df['X'], df['extra_inv']), 1)

    return df
