import numpy as np

def generate_demand(forecast):

    sigma = 0.2 * forecast

    demand = np.random.normal(
        forecast,
        sigma
    )

    return np.maximum(demand, 0)