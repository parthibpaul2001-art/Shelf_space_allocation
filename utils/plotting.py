import matplotlib.pyplot as plt

def plot_pareto(F):

    plt.figure()

    plt.scatter(
        F[:,0],
        F[:,1]
    )

    plt.xlabel("Profit")

    plt.ylabel("Carbon")

    plt.title("Pareto Front")

    plt.show()