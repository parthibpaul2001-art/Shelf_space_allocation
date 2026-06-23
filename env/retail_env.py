# retail_env.py
import pandas as pd

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class RetailShelfEnv(gym.Env):

    """
    Sustainable Retail Shelf Space Allocation Environment

    Objectives:
        1. Maximize Profit
        2. Minimize Carbon Emissions
        3. Maximize Customer Satisfaction
        4. Minimize Waste
        5. Minimize Stockouts
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):

        super(RetailShelfEnv, self).__init__()

        # --------------------------------------------------
        # SYSTEM PARAMETERS
        # --------------------------------------------------
        self.products = pd.read_csv(
            "data/products.csv"
        )

        

        self.demand_mean = self.products["DemandMean"].values
        self.demand_std = self.products["DemandStd"].values

        self.gamma = self.products["ShelfAttractiveness"].values
        self.min_display = self.products["MinDisplay"].values
        self.max_display = self.products["MaxDisplay"].values

        self.num_products = len(self.products)
        self.current_step = 0
        self.max_episode_steps = 100

        self.selling_price = self.products["Price"].values

        self.purchase_cost = self.products["Cost"].values

        self.holding_cost = self.products["HoldingCost"].values

        self.carbon_factor = self.products["CarbonFactor"].values

        self.customer_coeff = self.products["SatisfactionCoeff"].values

        self.shelf_life = self.products["ShelfLife"].values

        self.lead_time = self.products["LeadTime"].values

      


        # Shelf attractiveness

        self.gamma = self.products[
            "ShelfAttractiveness"
        ].values

        # --------------------------------------------------
        # ACTION SPACE
        # --------------------------------------------------

        """
        Action = shelf facings allocation

        Example:
        [20,15,30,10,25]
        """
        self.max_shelf_capacity = 500
        self.action_space = spaces.Box(
            low=0,
            high=self.max_shelf_capacity,
            shape=(self.num_products,),
            dtype=np.float32
        )

        # --------------------------------------------------
        # OBSERVATION SPACE
        # --------------------------------------------------

        """
        State =
        Inventory
        Demand forecast
        Product age
        """

        low = np.zeros(self.num_products * 3)

        high = np.ones(self.num_products * 3) * 1000

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )

        self.reset()

    # ======================================================
    # RESET
    # ======================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.current_step = 0

        self.inventory = np.random.randint(100, 200, self.num_products).astype(float)

        self.product_age = np.zeros(self.num_products)

        self.forecast_demand = np.random.randint(20, 50, self.num_products).astype(float)

        self.pending_orders = []

        state = self._get_state()

        print("Products :", self.num_products)
        print("State shape :", state.shape)

        return state, {}
     

    # ======================================================
    # STATE
    # ======================================================

    def _get_state(self):

        return np.concatenate([
            self.inventory,
            self.forecast_demand,
            self.product_age
        ]).astype(np.float32)

    # ======================================================
    # DEMAND UNCERTAINTY
    # ======================================================

    def generate_actual_demand(self):

        sigma = self.demand_std

        demand = np.random.normal(
        self.demand_mean,
        sigma
        )

        demand = np.maximum(demand, 0)

        return demand

    # ======================================================
    # STEP
    # ======================================================

    def step(self, action):

        self.current_step += 1

        # ---------------------------------------------
        # Shelf Capacity Constraint
        # ---------------------------------------------

        action = np.maximum(action, 0)

        if np.sum(action) > self.max_shelf_capacity:

            action = (
                action /
                np.sum(action)
            ) * self.max_shelf_capacity

        # ---------------------------------------------
        # Demand Realization
        # ---------------------------------------------

        demand = self.generate_actual_demand()

        # ---------------------------------------------
        # Sales
        # ---------------------------------------------

        sales = np.minimum(
            self.inventory,
            demand
        )

        stockout = np.maximum(
            demand - self.inventory,
            0
        )

        # ---------------------------------------------
        # Revenue
        # ---------------------------------------------

        revenue = np.sum(
            sales * self.selling_price
        )

        purchase_cost = np.sum(
            sales * self.purchase_cost
        )

        gross_profit = (
            revenue -
            purchase_cost
        )

        # ---------------------------------------------
        # Holding Cost
        # ---------------------------------------------

        holding_cost = np.sum(
            self.inventory *
            self.holding_cost
        )

        # ---------------------------------------------
        # Carbon Emission Cost
        # ---------------------------------------------

        carbon_cost = np.sum(
            action *
            self.carbon_factor
        )

        # ---------------------------------------------
        # Customer Satisfaction
        # ---------------------------------------------

        satisfaction = np.sum(
            self.customer_coeff *
            self.gamma *
            action
        )

        # ---------------------------------------------
        # Waste due to Perishability
        # ---------------------------------------------

        self.product_age += 1

        waste = np.where(
            self.product_age >
            self.shelf_life,
            self.inventory,
            0
        )

        waste_cost = np.sum(
            waste *
            self.purchase_cost
        )

        self.inventory -= waste

        self.product_age[
            self.product_age >
            self.shelf_life
        ] = 0

        # ---------------------------------------------
        # Inventory Update
        # ---------------------------------------------

        self.inventory -= sales

        self.inventory = np.maximum(
            self.inventory,
            0
        )

        # ---------------------------------------------
        # Replenishment Decision
        # ---------------------------------------------

        reorder_qty = np.maximum(
            demand - self.inventory,
            0
        )

        for i in range(self.num_products):

            arrival_time = (
                self.current_step +
                self.lead_time[i]
            )

            self.pending_orders.append(
                (
                    arrival_time,
                    i,
                    reorder_qty[i]
                )
            )

        # ---------------------------------------------
        # Receive Orders
        # ---------------------------------------------

        arrivals = []

        for order in self.pending_orders:

            arr_step, prod, qty = order

            if arr_step <= self.current_step:

                self.inventory[prod] += qty

            else:
                arrivals.append(order)

        self.pending_orders = arrivals

        # ---------------------------------------------
        # New Demand Forecast
        # ---------------------------------------------

        self.demand_mean = self.products[
            "DemandMean"
        ].values

        self.demand_std = self.products[
            "DemandStd"
        ].values
        self.forecast_demand = self.demand_mean

        # ---------------------------------------------
        # Objective 1
        # Profit
        # ---------------------------------------------

        profit = (
            gross_profit
            - holding_cost
            - waste_cost
        )

        # ---------------------------------------------
        # Objective 2
        # Carbon
        # ---------------------------------------------

        carbon = carbon_cost

        # ---------------------------------------------
        # Objective 3
        # Satisfaction
        # ---------------------------------------------

        customer_sat = satisfaction

        # ---------------------------------------------
        # Normalize Objectives
        # ---------------------------------------------

        profit_score = profit / 10000

        carbon_score = carbon / 1000

        sat_score = customer_sat / 1000

        stockout_score = np.sum(stockout) / 100

        waste_score = np.sum(waste) / 100

        # ---------------------------------------------
        # Reward Function
        # ---------------------------------------------

        reward = (
            0.4 * profit_score
            + 0.3 * sat_score
            - 0.1 * carbon_score
            - 0.1 * waste_score
            - 0.1 * stockout_score
        )

        # ---------------------------------------------
        # Termination
        # ---------------------------------------------

        terminated = (
            self.current_step >=
            self.max_episode_steps
        )

        truncated = False

        info = {

            "profit": profit,

            "carbon": carbon,

            "satisfaction": customer_sat,

            "waste": np.sum(waste),

            "stockout": np.sum(stockout),

            "reward": reward
        }

        return (
            self._get_state(),
            reward,
            terminated,
            truncated,
            info
        )

    # ======================================================
    # RENDER
    # ======================================================

    def render(self):

        print("\n---------------------")
        print("STEP:", self.current_step)

        print("Inventory:")
        print(self.inventory)

        print("Demand Forecast:")
        print(self.forecast_demand)

        print("---------------------")