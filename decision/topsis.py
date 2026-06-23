import numpy as np
import pandas as pd


def select_best_solution(result,
                         profit_weight=0.5,
                         carbon_weight=0.2,
                         satisfaction_weight=0.3):

    """
    Select best managerial solution from NSGA-II Pareto front
    using TOPSIS.

    Parameters
    ----------
    result : pymoo result object

    profit_weight : float

    carbon_weight : float

    satisfaction_weight : float
    """

    # ------------------------------------
    # Extract objectives from NSGA-II
    # ------------------------------------

    profit = -result.F[:, 0]
    carbon = result.F[:, 1]
    satisfaction = -result.F[:, 2]

    df = pd.DataFrame({

        "Profit": profit,
        "Carbon": carbon,
        "Satisfaction": satisfaction

    })

    # ------------------------------------
    # Decision matrix
    # ------------------------------------

    X = df.values

    # ------------------------------------
    # Normalize
    # ------------------------------------

    norm = np.sqrt(
        np.sum(X**2, axis=0)
    )

    X_norm = X / norm

    # ------------------------------------
    # Apply weights
    # ------------------------------------

    weights = np.array([

        profit_weight,
        carbon_weight,
        satisfaction_weight

    ])

    V = X_norm * weights

    # ------------------------------------
    # Ideal best
    # ------------------------------------

    ideal_best = np.array([

        np.max(V[:, 0]),
        np.min(V[:, 1]),
        np.max(V[:, 2])

    ])

    # ------------------------------------
    # Ideal worst
    # ------------------------------------

    ideal_worst = np.array([

        np.min(V[:, 0]),
        np.max(V[:, 1]),
        np.min(V[:, 2])

    ])

    # ------------------------------------
    # Distances
    # ------------------------------------

    D_plus = np.sqrt(
        np.sum(
            (V - ideal_best)**2,
            axis=1
        )
    )

    D_minus = np.sqrt(
        np.sum(
            (V - ideal_worst)**2,
            axis=1
        )
    )

    # ------------------------------------
    # TOPSIS score
    # ------------------------------------

    score = D_minus / (D_plus + D_minus)

    best_index = np.argmax(score)

    best_solution = {

        "index": best_index,

        "profit":
            profit[best_index],

        "carbon":
            carbon[best_index],

        "satisfaction":
            satisfaction[best_index],

        "decision_variables":
            result.X[best_index],

        "topsis_score":
            score[best_index]

    }



  

    return best_solution