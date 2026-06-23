import numpy as np
import pandas as pd

from pymoo.core.problem import Problem

class RetailProblem(Problem):

    def __init__(self):


        products = pd.read_csv(
        "data/products.csv"
        )

        n_products = len(products)

        super().__init__(

        n_var=n_products,

        n_obj=3,

        n_constr=1,

        xl=0,

        xu=100
        )

    def _evaluate(
        self,
        X,
        out,
        *args,
        **kwargs
    ):

        profit = []

        carbon = []

        satisfaction = []

        g = []

        for x in X:

            p = np.sum(x*20)

            c = np.sum(x*3)

            s = np.sum(x*5)

            profit.append(-p)

            carbon.append(c)

            satisfaction.append(-s)

            g.append(np.sum(x)-500)

        out["F"] = np.column_stack([
            profit,
            carbon,
            satisfaction
        ])

        out["G"] = np.array(g)