# Copyright: Johnson & Johnson Digital & Data Science (JJDDS)
#
# This file contains trade secrets of JJDDS. No part may be reproduced or transmitted in any
# form by any means or for any purpose without the express written permission of JJDDS.
#
# Purpose:  Polaris Application Source
import pandas as pd
import logging
APP_NAME = "Polaris"
logger = logging.getLogger(APP_NAME)
logger.setLevel(logging.INFO)


def get_options(solver, time=300, gap=0.01):
    if solver in ["gurobi", "gurobi_direct"]:
        options = {"MIPGap": gap, "TimeLimit": time}
    elif solver in ["xpress", "xpress_direct"]:
        options = {"miprelstop": gap, "maxtime": time}
    elif solver in ["cbc"]:
        options = {"ratio": gap, "sec": time, "preprocess":"off"}
    elif solver in ["glpk"]:
        options = {"mipgap": gap, "glp_time": time}
    else:
        KeyError(
            f"Solver {solver} not found in data library, please try another solver or contact developer to add {solver} to the library"
        )
    return options

def append_bounds(model,MinB,TargB,MaxB,AvD,TargR,TargD,sce):
    TargetReq, TargetDiff = model.get_target_req()
    MinB.append(pd.concat([model.d.minbound],keys=[sce]))
    TargB.append(pd.concat([model.d.target],keys=[sce]))
    MaxB.append(pd.concat([model.d.maxbound],keys=[sce]))
    AvD.append(pd.concat([model.d.avdemand],keys=[sce]))
    TargR.append(pd.concat([TargetReq], keys=[sce]))
    TargD.append(pd.concat([TargetDiff], keys=[sce]))

    return MinB,TargB,MaxB,AvD,TargR,TargD

def save_bounds(app,MinB,TargB,MaxB,AvD,TargR,TargD):
    MinBound = pd.concat(MinB)
    MinBound.index = MinBound.index.set_names(["Scenario","Item","Period"]).reorder_levels(["Item","Period","Scenario"])
    app.data.MinBound = MinBound

    Target = pd.concat(TargB)
    Target.index = Target.index.set_names(["Scenario","Item","Period"]).reorder_levels(["Item","Period","Scenario"])
    app.data.Target = Target

    MaxBound = pd.concat(MaxB)
    MaxBound.index = MaxBound.index.set_names(["Scenario","Item","Period"]).reorder_levels(["Item","Period","Scenario"])
    app.data.MaxBound = MaxBound

    AvDemand = pd.concat(AvD)
    AvDemand.index = AvDemand.index.set_names(["Scenario","Item","Period"]).reorder_levels(["Item","Period","Scenario"])
    app.data.AvDemand = AvDemand

    TargetR = pd.concat(TargR)
    TargetR.index = TargetR.index.set_names(["Scenario","Item","Period"]).reorder_levels(["Item","Period","Scenario"])
    app.data.TargetReq = TargetR

    TargetD = pd.concat(TargD)
    TargetD.index = TargetD.index.set_names(["Scenario","Item","Period"]).reorder_levels(["Item","Period","Scenario"])
    app.data.TargetDiff = TargetD
    
    return app   

def log_obj_values(instance, logger):
    value_dict = {
        "Objective": instance.obj,
        "SALES": instance.SALES,
        "CAPITAL_INVESTMENT": instance.CAPITAL_INVESTMENT,
        "CROSS_QUALIFICATION_COST": instance.CROSS_QUALIFICATION_COST,
        "HOLDING_COST": instance.HOLDING_COST,
        "TOTAL_DEVIATION": instance.TOTAL_DEVIATION,
        "ALLOCATION_DEVIATION": instance.ALLOCATION_DEVIATION,
        "WC_CAPACITY_DEV": instance.WC_CAPACITY_DEV,
        "PRIORITY_COST": instance.PRIORITY_COST,
        "FLEXIBLE_COMMITTED_INVESTMENT": instance.FLEXIBLE_COMMITTED_INVESTMENT,
        "GROUP_UTILIZATION_DEVIATION": instance.GROUP_UTILIZATION_DEVIATION,
    }

    for (n, v) in value_dict.items():
        value = v() if type(v) not in [float, int] else 0
        logger.info(f"Model {n} is: {value}")
        
    return None
