# Retail Shelf Space Optimization using Deep Reinforcement Learning and NSGA-II

## Overview

This project presents an intelligent retail shelf space optimization framework that combines:

- Deep Reinforcement Learning (PPO)
- Multi-objective Optimization (NSGA-II)
- TOPSIS Decision Making

The objective is to determine the optimal shelf allocation (facings) for multiple retail products while simultaneously maximizing profit, minimizing carbon emissions, and maximizing customer satisfaction under uncertain demand and inventory conditions.

---

# Problem Statement

Retail stores have limited shelf space and must determine how much space should be allocated to each product.

The proposed model considers

- Dynamic customer demand
- Inventory uncertainty
- Product perishability
- Carbon emission cost
- Customer satisfaction
- Shelf attractiveness
- Lead time
- Supplier service level
- Multi-objective optimization

---

# Objectives

The optimization simultaneously solves three objectives.

### Objective 1

Maximize Total Profit

- Sales revenue
- Inventory cost
- Holding cost
- Waste cost
- Stockout cost

---

### Objective 2

Minimize Carbon Emissions

Carbon emissions generated from

- transportation
- storage
- refrigeration
- inventory handling

---

### Objective 3

Maximize Customer Satisfaction

Customer satisfaction depends upon

- product availability
- shelf attractiveness
- demand fulfillment

---

# Solution Methodology

The project integrates three AI techniques.

## Phase 1

Deep Reinforcement Learning

Algorithm:

PPO (Proximal Policy Optimization)

Purpose:

Learn intelligent shelf allocation policies under uncertain retail environments.

---

## Phase 2

NSGA-II

Generates Pareto-optimal solutions considering

- Profit
- Carbon emission
- Customer satisfaction

---

## Phase 3

TOPSIS

Ranks the Pareto solutions and selects the single best managerial solution.

---

# Project Structure

```
Retail_shelf_optimization/

│
├── config/
│       config.py
│
├── data/
│       products.csv
│
├── env/
│       retail_env.py
│
├── drl/
│       train_ppo.py
│       policy_export.py
│       evaluate.py
│
├── nsga2/
│       retail_problem.py
│       optimize.py
│
├── decision/
│       topsis.py
│
├── utils/
│       demand_generator.py
│       metrics.py
│       plotting.py
│
├── results/
│
├── main.py
│
├── requirements.txt
│
└── README.md
```

---

# Dataset

The dataset contains information for 50 retail products.

Each product includes

- Product
- Category
- Price
- Cost
- Holding Cost
- Carbon Factor
- Satisfaction Coefficient
- Shelf Life
- Lead Time
- Demand Mean
- Demand Standard Deviation
- Shelf Attractiveness
- Minimum Display
- Maximum Display
- Service Level Target

---

# Installation

Create a virtual environment

```bash
python -m venv venv
```

Activate environment

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

Install required packages

```bash
pip install -r requirements.txt
```

---

# Required Libraries

- numpy
- pandas
- matplotlib
- gymnasium
- stable-baselines3
- pymoo
- scipy

---

# Running the Project

## Step 1

Train PPO agent

```bash
python -m drl.train_ppo
```

Output

```
results/
    ppo_retail.zip
```

---

## Step 2

Export learned policies

```bash
python -m drl.policy_export
```

Output

```
results/
    policies.npy
```

---

## Step 3

Run NSGA-II optimization

```bash
python -m nsga2.optimize
```

Output

```
results/
    pareto_front.csv
```

---

## Step 4

Run TOPSIS decision making

```bash
python main.py
```

Output

```
Best Shelf Allocation

Product              Facings

Milk                   45
Bread                  30
Chocolate              18
...
```

---

# Mathematical Model

The model considers

Decision Variables

- Shelf facings
- Inventory level
- Replenishment quantity

Objectives

- Maximize Profit
- Minimize Carbon Emission
- Maximize Customer Satisfaction

Constraints

- Shelf Capacity
- Inventory Balance
- Demand Uncertainty
- Product Expiration
- Lead Time
- Minimum Display
- Maximum Display
- Supplier Service Level
- Carbon Emission Limit

---

# AI Workflow

```
Retail Dataset
        │
        ▼
Retail Environment
        │
        ▼
PPO Training
        │
        ▼
Learned Policies
        │
        ▼
NSGA-II Optimization
        │
        ▼
Pareto Front
        │
        ▼
TOPSIS
        │
        ▼
Best Shelf Allocation
```

---

# Expected Outputs

- Trained PPO model
- Learned shelf allocation policies
- Pareto optimal solutions
- Best managerial solution
- Recommended shelf allocation
- Profit
- Carbon emission
- Customer satisfaction

---

# Future Work

- Real-time IoT inventory integration
- Computer vision shelf monitoring
- Digital twin of retail stores
- Online reinforcement learning
- Multi-store optimization
- Transformer-based demand forecasting

---

# Author

Retail Shelf Space Optimization using Deep Reinforcement Learning and Multi-objective Optimization

Academic Research Project
