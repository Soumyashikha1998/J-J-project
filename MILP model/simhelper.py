import numpy as np
import pandas as pd
import random
from pyclustering.cluster.kmedoids import kmedoids


def get_random_number(distribution, umin, umax):
    if distribution.upper() == "UNIFORM":
        return np.random.uniform(umin, umax)
    elif distribution.upper() == "TRIANGULAR":
        return np.random.triangular(umin, (umin + umax) / 2, umax)
    else:
        raise ValueError(f"Unrecognized distribution {distribution}")


def bass_model(demand):
    cumm = np.insert(demand.cumsum(), 0, 0)[:-1]
    coefs = np.polyfit(cumm, demand, 2)
    roots = np.roots(coefs)
    m = max(roots)
    q = -coefs[0] * m
    p = coefs[2] / m

    return coefs, m, p, q


def bass_scale(demand, mf=1, qf=1):
    cdemand = demand.cumsum()
    zeros = len(cdemand[cdemand == 0])
    demand = demand[zeros:]

    coefs, m, p, q = bass_model(demand)
    c1 = np.insert(demand.cumsum(), 0, 0)[:-1]
    diff1 = np.polyval(coefs, c1)
    m *= mf
    q *= qf

    c2 = [0]  # Second cummulative curve
    diff2 = []  # Second diffusion curve

    for t in range(len(demand)):
        diff2.append((p * m) + (q - p) * c2[t] + (-q / m) * c2[t] ** 2)
        c2.append(diff2[t] + c2[t])

    d2 = diff2 / diff1 * demand

    return np.append(np.zeros(zeros), d2)


def shift_demand(demand, shift):
    if shift > 0:
        # Create an array of zeros with the size of the shift argument
        zeros = np.zeros(shift)
        # Append zeros and demand to get a single array
        a = np.append(zeros, demand)
        # Generate a new array with the same length of demand
        new_demand = a[0 : len(demand)]

    elif shift < 0:
        # Create an array of the last value of the demand array with the size of the shift argument
        last = np.full(shape=-shift, fill_value=demand[-1])
        # Append last values and demand to get a single array
        a = np.append(demand, last)
        # Generate a new array with the same length of demand
        new_demand = a[-(len(demand)) :]
    else:
        return demand

    return new_demand


def enough_simulations(d, threshold=0.01):
    """
    Function to determine if enough simulations have been done

    """
    products = set(d.Product)

    std_min = (
        d.groupby("Product")["Demand"]
        .expanding(2)
        .std()
        .dropna()
        .rolling(10)
        .min()
        .dropna()
    )
    std_max = (
        d.groupby("Product")["Demand"]
        .expanding(2)
        .std()
        .dropna()
        .rolling(10)
        .max()
        .dropna()
    )
    mean_min = (
        d.groupby("Product")["Demand"]
        .expanding(2)
        .mean()
        .dropna()
        .rolling(10)
        .min()
        .dropna()
    )
    mean_max = (
        d.groupby("Product")["Demand"]
        .expanding(2)
        .mean()
        .dropna()
        .rolling(10)
        .max()
        .dropna()
    )

    std_band = (std_max - std_min) / std_max
    std_condition = std_band < threshold
    mean_band = (mean_max - mean_min) / mean_max
    mean_condition = mean_band < threshold
    condition = std_condition & mean_condition
    products_meeting_condition = set(
        condition[condition == True].reset_index()["Product"]
    )

    return products_meeting_condition == products


def get_hierarchy(platform, family, product):
    frame = {"Platform": platform, "Family": family, "Product": product}

    hierarchy_df = pd.DataFrame(frame)

    return hierarchy_df


def get_instructions(item, uncertainty_Type, distribution, Min, Max):
    frame = {
        "Item": item,
        "Uncertainty_Type": uncertainty_Type,
        "Distribution": distribution,
        "Min": Min,
        "Max": Max,
    }

    instructions_df = pd.DataFrame(frame)

    return instructions_df


def zero_division(n, d):
    return n / d if d else 0


def sort_instructions(idf):
    order = pd.DataFrame(
        {
            "order": [0, 1, 2, 3],
            "Uncertainty_Type": [
                "Market Potential",
                "Adoption Rate",
                "Delay",
                "Launch",
            ],
        }
    )
    idf = (
        idf.merge(order, how="left", on="Uncertainty_Type")
        .sort_values(by=["order"])
        .drop(["order"], axis=1)
    )

    return idf
