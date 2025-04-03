from pyomo.environ import *
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd

# Define the problem data

def datainput(data):

    periods = [1, 2, 3, 4]

# Read Machine Data
    # df_machine = pd.read_excel("/Users/soumyashikha/Desktop/parameters_work_centers.xlsx")
    df_machine = data['work_centers']
    machines = df_machine["Work Centers"].unique().tolist()
    C = dict(zip(df_machine["Work Centers"], df_machine["Capacity"]))
    Cost_m = dict(zip(df_machine["Work Centers"], df_machine["Usage Cost"]))
    Fix_InvCost_m = dict(zip(df_machine["Work Centers"], df_machine["Fixed Inv Cost"]))
    Var_InvCost_m = dict(zip(df_machine["Work Centers"], df_machine["Variable Inv Cost"]))

# Read Product Data    
    # df_product = pd.read_excel("/Users/soumyashikha/Desktop/parameters_products.xlsx")
    df_product = data['products']
    products = df_product["Products"].unique().tolist()
    c = dict(zip(df_product["Products"], df_product["Unit Cost"]))
    r = dict(zip(df_product["Products"], df_product["Revenue"]))
    h = dict(zip(df_product["Products"], df_product["Holding Cost"]))
    s0 = dict(zip(df_product["Products"], df_product["Initial Inventory"]))

# Read Product Demand Data
    # df_demand = pd.read_excel("/Users/soumyashikha/Desktop/parameters_products_demand.xlsx") 
    df_products_demand = data['products_demand']
    d_nom = {}
    for index, row in df_products_demand.iterrows():
        product = row['Products']
        for i in range(1, 5):  # For t1 to t4
            d_nom[(product, i)] = row[f't{i}']

    Min_value = data['Min_value']
    Max_value = data['Max_value']
# d_nom = {('P1', 1): 80, ('P1', 2): 100, ('P1', 3): 90, ('P1', 4): 80}
    d_min = {(p, t): Min_value/100 * d_nom[(p, t)] for (p, t) in d_nom}
    d_max = {(p, t): Max_value/100 * d_nom[(p, t)] for (p, t) in d_nom}

    M = 100000
    print(periods)
    print(products)
    print(c)
    print(r)
    print(h)
    print(s0)
    print(machines)
    print(C)
    print(Cost_m)
    print(Fix_InvCost_m)
    print(Var_InvCost_m)
    print(d_nom)
    print(d_min)
    print(d_max)
    print(M)
    return periods, products, c,r,h,s0, machines, C, Cost_m, Fix_InvCost_m, Var_InvCost_m, d_nom, d_min, d_max, M

# Step 0: Initialization - Solve MILP with fixed nominal demand
def solve_initial_feasible_milp(data):
    periods, products,  c,r,h,s0, machines, C, Cost_m, Fix_InvCost_m, Var_InvCost_m, d_nom, d_min, d_max, M=datainput(data)
    model = pyo.ConcreteModel()
    model.products = pyo.Set(initialize = products)
    model.periods = pyo.Set(initialize = periods)
    model.machines = pyo.Set(initialize = machines)

    # Decision variables
    model.x = pyo.Var(model.products, model.periods, bounds=(0, 1000), within=pyo.NonNegativeReals) ##Production
    model.s = pyo.Var(model.products, model.periods, bounds=(0, 1000), within=pyo.NonNegativeReals) ##Inventory
    model.y = pyo.Var(model.machines, model.periods, bounds=(0, 1000), within=pyo.NonNegativeIntegers)  ##Machine investment count
    model.y_bin = pyo.Var(model.machines, model.periods, bounds=(0,1), within=pyo.Binary) #Machine investment indicator
    model.y_l = pyo.Var(model.machines, model.periods, bounds=(0,1000), within=pyo.NonNegativeIntegers) #Cumulative machine availability

    # Objective function: Maximize profit
    def objective_rule(model):
        return (
            sum((c[p])* model.x[p, t] for p in products for t in periods) +     #Production cost
            sum(h[p] * model.s[p, t] for p in model.products for t in model.periods) +  #Inventory cost
            sum(Cost_m[m] * model.y_l[m,t] for t in periods for m in machines) +  #Operating cost
            sum(Fix_InvCost_m[m] * model.y_bin[m,t] for t in periods for m in machines) + #Fixed investment cost
            sum(Var_InvCost_m[m] * model.y[m,t] for t in periods for m in machines) #Variable investemnt cost
        )
    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)


    # Constraints
    def demand_satisfaction_rule(model, p, t):
        if t == 1:
            return model.x[p,t] + s0[p] >= d_max[p,t]
        else:
            return model.x[p,t] + model.s[p, t - 1] >= d_max[p,t]
    model.demand_satisfaction = pyo.Constraint(model.products, model.periods, rule = demand_satisfaction_rule)


    def inventory_balance_rule(model, p, t):
        if t == 1:
            return s0[p] + model.x[p,t] == d_max[(p,t)] + model.s[p, t]
        else:
            return model.s[p, t - 1] + model.x[p, t] == d_max[(p, t)] + model.s[p, t]
    model.inventory_balance = pyo.Constraint(model.products, model.periods, rule=inventory_balance_rule)


    def cumulative_machine_rule(model,m,t):
        if t == 1:
            return model.y_l[m, t] == model.y[m, t]  # Only new investments in first period
        else:
            return model.y_l[m, t] == model.y_l[m, t - 1] + model.y[m, t]  # Accumulate investments
    model.cumulative_machine = pyo.Constraint(model.machines, model.periods, rule=cumulative_machine_rule)


    def capacity_rule(model, t):
        return sum(model.x[p, t] for p in model.products) <= sum(C[m]*model.y_l[m,t] for m in machines)
    model.capacity = pyo.Constraint(model.periods, rule=capacity_rule)


    def production_indicator_rule(model, p, t):
        return model.x[p, t] <= sum(C[m] * model.y_l[m, t] for m in machines)
    model.production_indicator = pyo.Constraint(model.products, model.periods, rule=production_indicator_rule)


    def investment_activation_rule(model, m, t):
        return model.y[m, t] <= M * model.y_bin[m, t]  # If `y_bin=1`, `y` can be positive; otherwise `y=0`
    model.investment_activation = pyo.Constraint(model.machines, model.periods, rule=investment_activation_rule)


    def investment_demand_rule(model, t):
        return sum(C[m]*model.y[m, t] for m in machines) >= sum(model.x[p, t] for p in products)
    model.investment_demand = pyo.Constraint(model.periods, rule=investment_demand_rule)


##Verify this inventory condition for first and last time period
    for p in model.products:
        model.s[p, 1].fix(s0[p])
        model.s[p, 4].fix(s0[p])


    # Solve the MILP
    solver = SolverFactory('gurobi')
    results = solver.solve(model)

    # Extract the solution for y
    y_bin_solution = {(m, t): pyo.value(model.y_bin[m, t]) for m in machines for t in periods}
    y_solution = {(m, t): pyo.value(model.y[m, t]) for m in machines for t in periods}
    y_l_solution = {(m, t): pyo.value(model.y_l[m, t]) for m in machines for t in periods}

    return y_bin_solution, y_solution, y_l_solution

# Step 1: Multiparametric LP Problem
def solve_multiparametric_lp(data,y_bin_solution,y_solution,y_l_solution):
    periods, products,  c,r,h,s0, machines, C, Cost_m, Fix_InvCost_m, Var_InvCost_m, d_nom, d_min, d_max, M=datainput(data)
    model = pyo.ConcreteModel()
    model.products = pyo.Set(initialize = products)
    model.periods = pyo.Set(initialize = periods)
    model.machines = pyo.Set(initialize = machines)

    # Decision variables
    model.x = pyo.Var(model.products, model.periods, bounds=(0, 1000), within=pyo.NonNegativeReals)
    model.s = pyo.Var(model.products, model.periods, bounds=(0, 1000), within=pyo.NonNegativeReals)
    model.d = pyo.Var(model.products, model.periods, within=pyo.NonNegativeReals, 
                       bounds=lambda model, p, t: (d_min[(p, t)], d_max[(p, t)]))
    model.y = pyo.Var(model.machines, model.periods, bounds=(0, 1000), within=pyo.NonNegativeIntegers)  ##Machine investment count
    model.y_bin = pyo.Var(model.machines, model.periods, bounds=(0,1), within=pyo.Binary) #Machine investment indicator
    model.y_l = pyo.Var(model.machines, model.periods, bounds=(0,1000), within=pyo.NonNegativeIntegers) #Cumulative machine availability

    # Objective function: Maximize profit
    def objective_rule(model):
        return (
            sum((c[p])* model.x[p, t] for p in products for t in periods) +     #Production cost
            sum(h[p] * model.s[p, t] for p in model.products for t in model.periods) +  #Inventory cost
            sum(Cost_m[m] * model.y_l[m,t] for t in periods for m in machines) +  #Operating cost
            sum(Fix_InvCost_m[m] * y_bin_solution[m,t] for t in periods for m in machines) + #Fixed investment cost
            sum(Var_InvCost_m[m] * y_solution[m,t] for t in periods for m in machines) #Variable investemnt cost
        )
    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Constraints
    def demand_satisfaction_rule(model, p, t):
        if t == 1:
            return model.x[p,t] + s0[p] >= model.d[p,t]
        else:
            return model.x[p,t] + model.s[p, t - 1] >= model.d[p,t]
    model.demand_satisfaction = pyo.Constraint(model.products, model.periods, rule = demand_satisfaction_rule)

    def inventory_balance_rule(model, p, t):
        if t == 1:
            return s0[p] + model.x[p,t] == model.d[p,t] + model.s[p, t]
        else:
            return model.s[p, t - 1] + model.x[p, t] == model.d[p,t] + model.s[p, t]
    model.inventory_balance = pyo.Constraint(model.products, model.periods, rule=inventory_balance_rule)

    # def cumulative_machine_rule(model,m,t):
    #     if t == 1:
    #         return y_l_solution[m, t] == y_solution[m, t]  # Only new investments in first period
    #     else:
    #         return y_l_solution[m, t] == y_l_solution[m, t - 1] + y_solution[m, t]  # Accumulate investments
    # model.cumulative_machine = pyo.Constraint(model.machines, model.periods, rule=cumulative_machine_rule)

    def capacity_rule(model, t):
        return sum(model.x[p, t] for p in model.products) <= sum(C[m]*model.y_l[m,t] for m in machines)
    model.capacity = pyo.Constraint(model.periods, rule=capacity_rule)

    def production_indicator_rule(model, p, t):
        return model.x[p, t] <= sum(C[m] * y_l_solution[m, t] for m in machines)
    model.production_indicator = pyo.Constraint(model.products, model.periods, rule=production_indicator_rule)


    # def investment_activation_rule(model, m, t):
    #     return y_solution[m, t] <= M * y_bin_solution[m, t] 
    # model.investment_activation = pyo.Constraint(model.machines, model.periods, rule=investment_activation_rule)


    def investment_demand_rule(model, t):
        return sum(C[m]*y_solution[m, t] for m in machines) >= sum(model.x[p, t] for p in products)
    model.investment_demand = pyo.Constraint(model.periods, rule=investment_demand_rule)


    for p in model.products:
        model.s[p, 1].fix(s0[p])
        model.s[p, 4].fix(s0[p])


    solver = SolverFactory('gurobi')

    # # Function to get active constraints
    def get_active_constraints(model, tolerance=1e-5):
        active = []
        for c in model.component_objects(pyo.Constraint, active=True):
            for index in c:
                lhs = pyo.value(c[index].body)
                rhs = pyo.value(c[index].upper) if c[index].has_ub() else pyo.value(c[index].lower)
                if abs(lhs - rhs) <= tolerance:
                    active.append(c.name)
        return frozenset(active)


    # Generate demand values over the parametric range
    theta_values = {t: np.linspace(d_min['P1', t], d_max['P1', t], num=3) for t in periods}
    critical_regions = {}
    infeasible_regions = []
    stored_data = []

    # y_keys = sorted(y_solution.keys())
    # for key in y_keys:
    #     df[f"Investment({key[0]},{key[1]})"] = [y_solution[key]]

    for theta1 in theta_values[1]:
        for theta2 in theta_values[2]:
            for theta3 in theta_values[3]:
                for theta4 in theta_values[4]:
                    # Set parameter values dynamically

                    model.d['P1', 1].fix(theta1)
                    model.d['P1', 2].fix(theta2)
                    model.d['P1', 3].fix(theta3)
                    model.d['P1', 4].fix(theta4)

                    results = solver.solve(model, tee=False)
                    if results.solver.termination_condition == pyo.TerminationCondition.infeasible:
                        infeasible_regions.append((theta1,theta2,theta3,theta4))
                        continue #Skipping further calculations for infeasible regions
                    # Extract solution values
                    x_solution = {(p, t): pyo.value(model.x[p, t]) for p in model.products for t in model.periods}
                    obj_value = pyo.value(model.obj)


                    # Get active constraints
                    active_set = get_active_constraints(model)

                    if active_set not in critical_regions:
                        critical_regions[active_set] = []
                    critical_regions[active_set].append((theta1, theta2, theta3, theta4, obj_value,
                                                         x_solution['P1', 1], x_solution['P1', 2], x_solution['P1', 3], x_solution['P1', 4], 
                                                         y_solution['WC1', 1], y_solution['WC1', 2], y_solution['WC1', 3], y_solution['WC1', 4],
                                                         y_solution['WC2', 1], y_solution['WC2', 2], y_solution['WC2', 3], y_solution['WC2', 4],
                                                         y_solution['WC3', 1], y_solution['WC3', 2], y_solution['WC3', 3], y_solution['WC3', 4]))          
    # print("Critical Regions:",critical_regions)
    row_index = 0  # Start index for each row                
    # print("\nCritical Regions:")
    for i, (active_set, values) in enumerate(critical_regions.items(), start=1):
        # print(f"Critical Region {i}: Active constraints = {active_set}")
        for val in values:
            # print(f"  θ1={val[0]:.2f}, θ2={val[1]:.2f}, θ3={val[2]:.2f}, θ4={val[3]:.2f} → "
            #     f"Obj={val[4]:.2f}\n"
            #     f"x(P1,1)={val[5]:.2f}, x(P1,2)={val[6]:.2f}, x(P1,3)={val[7]:.2f}, x(P1,4)={val[8]:.2f} "
            # #     # f"y(P1,1)={val[9]:.2f}, s(P1,2)={val[10]:.2f}, s(P1,3)={val[11]:.2f}, s(P1,4)={val[12]:.2f}\n"
            #     + "---"*40 + "\n")
            stored_data.append([row_index, f"CR {i}",
            f"{val[0]:.2f}", f"{val[1]:.2f}", f"{val[2]:.2f}", f"{val[3]:.2f}",  # Theta values
            f"{val[4]:.2f}",  # Objective function
            f"{val[5]:.2f}", f"{val[6]:.2f}", f"{val[7]:.2f}", f"{val[8]:.2f}",  # X values
            f"{val[9]:.2f}", f"{val[10]:.2f}", f"{val[11]:.2f}", f"{val[12]:.2f}",  # Y values
            f"{val[13]:.2f}", f"{val[14]:.2f}", f"{val[15]:.2f}", f"{val[16]:.2f}",  # Y values
            f"{val[17]:.2f}", f"{val[18]:.2f}", f"{val[19]:.2f}", f"{val[20]:.2f}"  # Y values
            ])
            row_index += 1  # Increment index per row

        columns = ["Index", "Critical Regions",
            "Theta 1", "Theta 2", "Theta 3", "Theta 4", 
            "Obj",
            "X (P1,1)", "X (P1,2)", "X (P1,3)", "X (P1,4)", 
            "Y (WC1,1)", "Y (WC1,2)", "Y (WC1,3)", "Y (WC1,4)",
            "Y (WC2,1)", "Y (WC2,2)", "Y (WC2,3)", "Y (WC2,4)",
            "Y (WC3,1)", "Y (WC3,2)", "Y (WC3,3)", "Y (WC3,4)"
            ]

        # Convert list to DataFrame
        df = pd.DataFrame(stored_data, columns=columns)
        # df.to_csv("results.csv")
        df["Scenario Name"] = ["Sc" + str(i+1) for i in range(len(df))]

    # Reorder columns to place "Scenario Name" after "Critical Regions"
        column_order = df.columns.tolist()  # Get current column order
    # Remove "Scenario Name" from the end if it was added there
        column_order.remove("Scenario Name")
    # Insert "Scenario Name" right after "Critical Regions"
        insert_position = column_order.index("Critical Regions") + 1  # Find position after "Critical Regions"
        column_order.insert(insert_position, "Scenario Name")  # Insert at the correct position

        df = df.loc[:, column_order]
    print("\nInfeasible Regions:")
    for region in infeasible_regions:
        print(f"  θ1={region[0]:.2f}, θ2={region[1]:.2f}, θ3={region[2]:.2f}, θ4={region[3]:.2f}")

    return critical_regions, infeasible_regions, df

def iterative_MILP(data, df, critical_regions, y_bin_solution,y_solution,y_l_solution):
    periods, products,  c,r,h,s0, machines, C, Cost_m, Fix_InvCost_m, Var_InvCost_m, d_nom, d_min, d_max, M=datainput(data)
    theta1_values = []
    theta2_values = []
    theta3_values = []
    theta4_values = []
    obj_values = []

    for i, (active_set, values) in enumerate(critical_regions.items(), start=1):
        for val in values:
            theta1_values.append(val[0])
            theta2_values.append(val[1])
            theta3_values.append(val[2])
            theta4_values.append(val[3])
            obj_values.append(val[4])
    model = pyo.ConcreteModel()
    model.products = pyo.Set(initialize = products)
    model.periods = pyo.Set(initialize = periods)
    model.machines = pyo.Set(initialize = machines)

    # Decision variables
    model.x = pyo.Var(model.products, model.periods, bounds=(0, 1000), within=pyo.NonNegativeReals)
    model.s = pyo.Var(model.products, model.periods, bounds=(0, 1000), within=pyo.NonNegativeReals)
    model.d = pyo.Var(model.products, model.periods, within=pyo.NonNegativeReals, 
                       bounds=lambda model, p, t: (d_min[(p, t)], d_max[(p, t)]))
    model.y = pyo.Var(model.machines, model.periods, bounds=(0, 1000), within=pyo.NonNegativeIntegers)  ##Machine investment count
    model.y_bin = pyo.Var(model.machines, model.periods, bounds=(0,1), within=pyo.Binary) #Machine investment indicator
    model.y_l = pyo.Var(model.machines, model.periods, bounds=(0,1000), within=pyo.NonNegativeIntegers) #Cumulative machine availability
    model.old_obj = pyo.Var(within=pyo.NonNegativeIntegers)

    # Objective function: Maximize profit
    def objective_rule(model):
        return (
            sum((c[p])* model.x[p, t] for p in products for t in periods) +     #Production cost
            sum(h[p] * model.s[p, t] for p in model.products for t in model.periods) +  #Inventory cost
            sum(Cost_m[m] * model.y_l[m,t] for t in periods for m in machines) +  #Operating cost
            sum(Fix_InvCost_m[m] * model.y_bin[m,t] for t in periods for m in machines) + #Fixed investment cost
            sum(Var_InvCost_m[m] * model.y[m,t] for t in periods for m in machines) #Variable investemnt cost
        )
    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Constraints
    def demand_satisfaction_rule(model, p, t):
        if t == 1:
            return model.x[p,t] + s0[p] >= model.d[p,t]
        else:
            return model.x[p,t] + model.s[p, t - 1] >= model.d[p,t]
    model.demand_satisfaction = pyo.Constraint(model.products, model.periods, rule = demand_satisfaction_rule)

    def inventory_balance_rule(model, p, t):
        if t == 1:
            return s0[p] + model.x[p,t] == model.d[p,t] + model.s[p, t]
        else:
            return model.s[p, t - 1] + model.x[p, t] == model.d[p,t] + model.s[p, t]
    model.inventory_balance = pyo.Constraint(model.products, model.periods, rule=inventory_balance_rule)

    def cumulative_machine_rule(model,m,t):
        if t == 1:
            return model.y_l[m, t] == model.y[m, t]  # Only new investments in first period
        else:
            return model.y_l[m, t] == model.y_l[m, t - 1] + model.y[m, t]  # Accumulate investments
    model.cumulative_machine = pyo.Constraint(model.machines, model.periods, rule=cumulative_machine_rule)

    def capacity_rule(model, t):
        return sum(model.x[p, t] for p in model.products) <= sum(C[m]*model.y_l[m,t] for m in machines)
    model.capacity = pyo.Constraint(model.periods, rule=capacity_rule)

    def production_indicator_rule(model, p, t):
        return model.x[p, t] <= sum(C[m] * model.y_l[m, t] for m in machines)
    model.production_indicator = pyo.Constraint(model.products, model.periods, rule=production_indicator_rule)

    def investment_activation_rule(model, m, t):
        return model.y[m, t] <= M * model.y_bin[m, t] 
    model.investment_activation = pyo.Constraint(model.machines, model.periods, rule=investment_activation_rule)

    def investment_demand_rule(model, t):
        return sum(C[m]*model.y[m, t] for m in machines) >= sum(model.x[p, t] for p in products)
    model.investment_demand = pyo.Constraint(model.periods, rule=investment_demand_rule)
 
    for p in model.products:
        model.s[p, 1].fix(s0[p])
        model.s[p, 4].fix(s0[p])

    def combined_integer_cut_rule(model):
        return (sum(model.y[m, t] - y_solution[m, t] for m in machines for t in periods) 
                + sum(model.y_l[m, t] - y_l_solution[m, t] for m in machines for t in periods)
                + sum(model.y_bin[m,t]*(1-y_bin_solution[m,t]) + y_bin_solution[m,t]*(1-model.y_bin[m,t]) for m in machines for t in periods) >= 1)
    model.combined_integer_cut = pyo.Constraint(rule = combined_integer_cut_rule)

    def main_cut_rule(model):
        return (sum((c[p])* model.x[p, t] for p in products for t in periods) +     #Production cost
            sum(h[p] * model.s[p, t] for p in model.products for t in model.periods) +  #Inventory cost
            sum(Cost_m[m] * model.y_l[m,t] for t in periods for m in machines) +  #Operating cost
            sum(Fix_InvCost_m[m] * model.y_bin[m,t] for t in periods for m in machines) + #Fixed investment cost
            sum(Var_InvCost_m[m] * model.y[m,t] for t in periods for m in machines) #Variable investemnt cost
            <= model.old_obj)
    model.main_cut = pyo.Constraint(rule = main_cut_rule)


    solver = SolverFactory('gurobi')
    feasible_demand_values = []  # List to store all feasible demand values
    feasible_x_solutions = {}  # Dictionary to store feasible y solutions
    feasible_y_solutions = {}
    feasible_y_bin_solutions = {}  # Dictionary to store feasible y_bin solutions
    feasible_obj_values = []  # List to store feasible objective values

    

    for i, (d1, d2, d3, d4, o) in enumerate(zip(theta1_values, theta2_values, theta3_values, theta4_values, obj_values)):
        model.d['P1', 1].fix(d1)
        model.d['P1', 2].fix(d2)
        model.d['P1', 3].fix(d3)
        model.d['P1', 4].fix(d4)
        model.old_obj.fix(o)

        results = solver.solve(model, tee=False)
        if results.solver.termination_condition == pyo.TerminationCondition.infeasible:
            print(f"Solution is infeasible for demand values theta = {d1, d2, d3, d4}")
            
            continue 
        feasible_demand_values.append((d1, d2, d3, d4))
        feasible_x_solutions[i] = {(p,t): pyo.value(model.x[p,t]) for p in model.products for t in model.periods}
        feasible_y_solutions[i] = {(m, t): pyo.value(model.y[m, t]) for m in model.machines for t in model.periods} 
        feasible_y_bin_solutions[i] = {(m, t): pyo.value(model.y_bin[m, t]) for m in model.machines for t in model.periods}
        feasible_obj_values.append(pyo.value(model.obj))


    # Step 1: Initialize the Updated columns with original values
    df["Updated Obj"] = df["Obj"]
    for m in machines:
        for t in periods:
            col_name = f"Y ({m},{t})"
            new_col_name = f"Y_up({m},{t})"

            if col_name in df.columns:
                df[new_col_name] = df[col_name]

    for p in products:
        for t in periods:
            col_name = f"X ({p},{t})"
            new_col_name = f"X_up({p},{t})"

            if col_name in df.columns:
                df[new_col_name] = df[col_name]   

    # Step 2: Convert Theta values to tuples for matching
    df["Theta Tuple"] = list(zip(df["Theta 1"], df["Theta 2"], df["Theta 3"], df["Theta 4"]))
    df["Theta Tuple"] = df["Theta Tuple"].apply(lambda x: tuple(map(float, x)))

    # Step 3: Create a dictionary mapping theta tuples to updated objective values
    updated_obj_dict = {tuple(theta): obj for theta, obj in zip(feasible_demand_values, feasible_obj_values)}
    # Create a mapping from theta tuples to feasible y solutions
    updated_y_dict = {tuple(theta): y_dict for theta, y_dict in zip(feasible_demand_values, feasible_y_solutions.values())}
    updated_x_dict = {tuple(theta): x_dict for theta, x_dict in zip(feasible_demand_values, feasible_x_solutions.values())}


    # Step 4: Update only feasible demand values using `apply()`
    df["Updated Obj"] = df.apply(lambda row: updated_obj_dict.get(row["Theta Tuple"], row["Obj"]), axis=1)
    # Update the Y_up columns using the mapping
    for m in machines:
        for t in periods:
            col_name = f"Y_up({m},{t})"  # Column name in DataFrame
            df[col_name] = df.apply(lambda row: updated_y_dict.get(row["Theta Tuple"], {}).get((m, t), row[col_name]), axis=1)

    for p in products:
        for t in periods:
            col_name = f"X_up({p},{t})"  # Column name in DataFrame
            df[col_name] = df.apply(lambda row: updated_x_dict.get(row["Theta Tuple"], {}).get((p, t), row[col_name]), axis=1)

    # Step 5: Remove the helper column
    df.drop(columns=["Theta Tuple"], inplace=True)
    # Drop original Y columns
    for m in machines:
        for t in periods:
            col_name = f"Y ({m},{t})"
            if col_name in df.columns:
                df.drop(columns=[col_name], inplace=True, errors='ignore')
            else:
                print(f"Warning: Column '{col_name}' not found in DataFrame.")

    # for p in products:
    #     for t in periods:
    #         col_name = f"X ({p},{t})"
    #         if col_name in df.columns:
    #             df.drop(columns=[col_name], inplace=True, errors='ignore')
    #         else:
    #             print(f"Warning: Column '{col_name}' not found in DataFrame.")

    # Step 1: Select only the Y_up columns
    y_up_columns = [col for col in df.columns if col.startswith("Y_up(")]

    # Step 2: Convert unique combinations into a categorical 'Plan' column
    df["Plan"] = df[y_up_columns].apply(lambda row: tuple(row), axis=1).astype("category").cat.codes

    # Step 3: Convert numerical labels into "Plan A", "Plan B", etc.
    df["Plan"] = df["Plan"].apply(lambda x: f"Plan {chr(65 + x)}")  # Convert 0 -> A, 1 -> B, ...

    # Ensure all theta values are treated as numeric
    df[["Theta 1", "Theta 2", "Theta 3", "Theta 4"]] = df[["Theta 1", "Theta 2", "Theta 3", "Theta 4"]].astype(float)

    # Sort the dataframe in a hierarchical way to match Excel sorting behavior
    df = df.sort_values(by=["Theta 4"], ascending=True, kind="stable")
    df = df.sort_values(by=["Theta 3"], ascending=True, kind="stable")
    df = df.sort_values(by=["Theta 2"], ascending=True, kind="stable")
    df = df.sort_values(by=["Theta 1"], ascending=True, kind="stable")
        # Reset index to maintain order
    df = df.reset_index(drop=True)

    df["Scenario Name"] = ["Sc" + str(i+1) for i in range(len(df))]

    # Reorder columns to place "Scenario Name" after "Critical Regions"
    column_order = df.columns.tolist()  # Get current column order
    # Remove "Scenario Name" from the end if it was added there
    column_order.remove("Scenario Name")
    # Insert "Scenario Name" right after "Critical Regions"
    insert_position = column_order.index("Critical Regions") + 1  # Find position after "Critical Regions"
    column_order.insert(insert_position, "Scenario Name")  # Insert at the correct position


    # Step 3: Reorder dataframe
    df = df[column_order]
    # save updated DataFrame
    df.to_csv("results_updated.csv", index=False)
    print("Updated Objective Function in the file")

    return feasible_demand_values, feasible_y_solutions, feasible_y_bin_solutions, feasible_obj_values, df
    

# # Main algorithm
# def main():
#     # Step 0: Solve initial feasible MILP

#     y_bin_insolution,y_insolution,y_l_insolution = solve_initial_feasible_milp()
#     print("Initial y_bin_solution:", y_bin_insolution)
#     print("Initial y_solution:", y_insolution)
#     print("Initial y_l_solution:", y_l_insolution)
#     print("-"*100)
#     print("Initial MILP is solved\n")
#     #Step 1: Solve multiparametric LP
#     critical_regions,infeasible_regions,df= solve_multiparametric_lp(y_bin_insolution,y_insolution,y_l_insolution)
#     print("Critical regions are created")
#     # Step 2: Solve MILP for each CR (each feasible demand scenarios)
#     demand_values, y_solution, y_bin_solution, obj = iterative_MILP(df, critical_regions,y_bin_insolution,y_insolution,y_l_insolution)
#     print("Demand values for which new better solution is found:", demand_values)
#     print("y_solution", y_solution)
#     print("y_bin_solution", y_bin_solution)
#     print("Objective Function", obj)

# if __name__ == "__main__":
#     main()

# y_bin_insolution,y_insolution,y_l_insolution = solve_initial_feasible_milp()
# print("Initial y_bin_solution:", y_bin_insolution)
# print("Initial y_solution:", y_insolution)
# print("Initial y_solution:", y_insolution)
# print("Initial y_l_solution:", y_l_insolution)    
# print("-"*100)
# print("Initial MILP is solved\n")
# #Step 1: Solve multiparametric LP
# critical_regions,infeasible_regions,df= solve_multiparametric_lp(y_bin_insolution,y_insolution,y_l_insolution)
# print("Critical regions are created")
# # Step 2: Solve MILP for each CR (each feasible demand scenarios)
# demand_values, y_solution, y_bin_solution, obj = iterative_MILP(df, critical_regions,y_bin_insolution,y_insolution,y_l_insolution)
# print("Demand values for which new better solution is found:", demand_values)
# print("y_solution", y_solution)
# print("y_bin_solution", y_bin_solution)
# print("Objective Function", obj)
