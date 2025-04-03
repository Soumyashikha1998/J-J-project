import pandas as pd
import time

#from application import logger


def scenario_sampling(simulations, simulation_probability, selection, selection_mode='Aggregate'):
    percentile = [float(i)/100 for i in str(selection).split(",")]

    simulations = simulations.reset_index()
    simulations = simulations.merge(simulation_probability.reset_index(), on = "Simulation")

    # Aggregate
    if selection_mode == "Aggregate":

        print(
            "Pick the {} percent quantile demand for each period and products using aggregated selection mode".format(
                [x*100 for x in percentile]
            )
        )
        agg_demand = simulations.groupby('Simulation')["Demand"].sum()
        agg_demand.name = 'AggDemand'
        simulations = simulations.merge(agg_demand.reset_index(), on = "Simulation")
        simulations = simulations.sort_values(['AggDemand'], ascending=True)
        aggregated_prob = (simulations.groupby(["Simulation"], sort=False)["Probability"].sum()).cumsum()
        aggregated_prob = aggregated_prob/aggregated_prob.max()
        aggregated_prob = aggregated_prob.round(5)

        picked_sim = []
        for value in percentile:
            prob_sim = aggregated_prob[aggregated_prob >= value].iloc[0]
            picked_sim.append(prob_sim)

        dlist = [i for i in aggregated_prob.index if aggregated_prob.loc[i] in picked_sim]
        demand = simulations[simulations["Simulation"].isin(dlist)].rename(columns={"Simulation":"Scenario"}).reset_index(drop=True)
        mapping = {}
        for i in dlist:
            for j in percentile:
                if aggregated_prob[i] == aggregated_prob[aggregated_prob >= j].iloc[0]:
                    mapping[i] = "{}%_Aggregate".format(int(j*100))
                    break
        demand["Scenario"] = demand["Scenario"].apply(lambda x: mapping[x])
        demand = demand.drop(columns = ['Probability', 'AggDemand'])
        slist = demand["Scenario"].unique().tolist()
    #Individual
    elif selection_mode == "Individual":
        print(
            "Pick the {} percent quantile demand for each period and products using individual selection mode".format(
                [x*100 for x in percentile]
            )
        )

        simulations = simulations.sort_values(['Demand'],ascending=True)
        simulations['cum_prob'] = simulations.groupby(['Product', 'Period'])['Probability'].cumsum()
        simulations = simulations.reset_index(drop=True)
        simulations['cum_prob'] = simulations['cum_prob'].round(5)
        
        demand = pd.DataFrame()
        for value in percentile:
            percentile_sim = simulations[simulations.cum_prob >= value]
            percentile_df = percentile_sim.groupby(['Product', 'Period'])["Demand"].agg(lambda x: x.iloc[0]).reset_index()
            percentile_df["Scenario"] = value
            demand = pd.concat([demand, percentile_df], ignore_index=True)

        demand["Scenario"] = demand["Scenario"].apply(lambda x: "{}%_{}".format(int(x*100), selection_mode))
        slist = demand["Scenario"].unique().tolist()
    #Unknown
    else:
        raise ValueError('Invalid selection mode')

    demand = (
        demand.sort_values(["Scenario", "Period", "Product"])
        .set_index(["Product", "Period", "Scenario"])
        .squeeze()
    )
    return demand, slist


def export_selected_scenarios(demand):
    filename = "selected_scenarios_output"
    now = time.strftime("%Y%m%d%H%M%S")
    filename = f"{filename}_{now}.xlsx"
    writer = pd.ExcelWriter(filename, engine="xlsxwriter")
    demand.reset_index().to_excel(
        writer, sheet_name="Selected Scenarios", index=False
    )
    writer.save()
    print("Selected scenarios data written to file successfully")