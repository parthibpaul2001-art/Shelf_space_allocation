import pandas as pd
from decision.topsis import select_best_solution
from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.optimize import minimize

from nsga2.retail_problem import RetailProblem

from utils.plotting import plot_pareto

problem = RetailProblem()

algorithm = NSGA2(

    pop_size=100
)

result = minimize(

    problem,

    algorithm,

    ("n_gen",200),

    seed=1,

    verbose=True
)
best = select_best_solution(
    result,
    profit_weight=0.5,
    carbon_weight=0.2,
    satisfaction_weight=0.3
)

print("\n")
print("=" * 50)
print("BEST MANAGERIAL SOLUTION")
print("=" * 50)

print(
    "Profit:",
    best["profit"]
)

print(
    "Carbon:",
    best["carbon"]
)

print(
    "Satisfaction:",
    best["satisfaction"]
)

print(
    "TOPSIS Score:",
    best["topsis_score"]
)

print(
    "Shelf Allocation:"
)

df = pd.read_csv(
"data/products.csv"
)

products = df["Product"].values

allocation = best["decision_variables"]

print("\nRecommended Shelf Allocation\n")

for p, a in zip(products, allocation):

    print(
        f"{p} : {round(a,2)} facings"
    )

plot_pareto(
    result.F
)