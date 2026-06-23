import numpy as np

def compute_service_level(stockout,demand):

    return 1 - np.sum(stockout)/max(
        np.sum(demand),1
    )


def compute_fill_rate(sales,demand):

    return np.sum(sales)/max(
        np.sum(demand),1
    )