#%%

import pandas as pd

from compute_ui_params import compute_penalizations
from data_helper import compute_capacity_utilization
from data_input import SCMData
from precompute import *
from scms_model_CM import SCMSModel
from simhelper import *
from simulation import *
from simulation_routine import *

# File to load - The template should be complete, because there is no instance for blank cells completition 
# in an automatic way. There are many elements that are marked in the files with a TODO that should be provided 
# by the user. All the information come from an UI in the normal app, so I recommend to put everything in an extra sheet
# in the input file.

# filename = 'Polaris_InputTemplate_v0.8.xlsx'
filename = "Polaris_InputTemplate_v0.8_samuel.xlsx"

# Loading the information and creating the elements that will be used in the routine
data = SCMData(filename)
data.load_data()

# Here some values that are parameterized, are calculated.

# Computes default values for deviation penalizations before the UI (user editable)
(
    data.shortage_deviation_penalization,
    data.excess_deviation_penalization,
    data.item_deviation_penalization,
) = compute_penalizations(data.Standard_Cost, skip=data.items.empty)


# Calculations for workcenter capacity 
(
    data.capacity_calc,
    data.available_capacity_calc,
    data.min_capacity_calc,
    data.max_capacity_calc,
) = compute_capacity_utilization(
    data.workcenters,
    data.periods,
    data.Capacity,
    data.Availability,
    data.Min_Utilization,
    data.Max_Utilization,
)
'''
After loading the information, we can go to the simulations. Simulations are made with Montecarlo, following the theory
that is in the document that Braulio shared. There is two mode for optimization: Optimized and Run as is. 
Optimized mode gives the possibility to invest in inventory and devices to expand the capacity. Run as is, will only invest 
in inventory. The Optimized mode is the default one, and the one that we are interested in.
You can optimize also a single scenario or a group of scenarios. The declaration for this is in data.forecast_considered.
This data is in the Option tab in the input file. 
If the value is "Only Nominal" it will run a single optimization, using the forecast that is in the input.
If the value is "All Sampled Scenarios", it will take the nominal scenario and the scenarios that were picked after the
simulation. 
Here I connect the simulation directly with the All Sampled Scenarios. The simulation follows the comments in the document.
After the simulation you will have a list of scecnarios (slist), demand (the demand of the original nominal and selected scenarios)
simulations (the simulations that are represntative of the number of clusters that were used for making the size reduction)
and the simulation probability, where you can look for the probability of the specific scenarios that represent the clusters.
'''

# # Precomputation
# #! TODO: this element is part of the input
# data.forecast_considered = "All Sampled Scenarios"
# data.write_file = True

# if data.forecast_considered == "All Sampled Scenarios":
#     sim_df, slist, demand, simulations, simulation_probability = simulate_main(data)

'''
The precomputation is the computation of the sets and other values that will be used in optimization
'''
precompute(data)
print("Precomputation finished")
#%%
# Optimization
'''
Evaluate_Solutions is a tool that we are evaluating for risk. It runs the optimization for the selected scenarios
and then the expanding capacity actions are fixed in all of them. The selected scenarios are called isntances. 
For each instances a cycle of optimization is made for the set of reduced simulations, and the expanding capacity policy 
for the instance is fixed. So the feedback of these optimization is to have new production policies, with different demands.
We obtain different KPIs, using expected Shortfall. This could be useful to make decisions between different 
drivers to conduct decisions. 
'''
data.evaluate_solutions = False #True

'''
This is the routine that coordinates the optimization steps
'''
m = SCMSModel(data)
#%%
'''
In this step you will solved the selected scenarios, and for each of them you will have outputs. 
I did not put here the output routine because it has a lot of thing that are useful for the real app.
I think that you can obtain the data and analize without our "formal" output.
If you need I can help you sharing haow the content are processed.
'''
m.solve()
#%%



#### Error message: pyomo.common.errors.ApplicationError: No executable found for solver 'cbc' ####