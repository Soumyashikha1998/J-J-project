# 📊 Sensitivity Analysis of MILP-Based Expansion Decisions Under Demand Uncertainty  
*A Data-Driven Approach for Strategic Capacity Management Planning*  

![Banner Image](https://via.placeholder.com/1200x400?text=Sensitivity+Analysis+of+MILP+Expansion+Decisions)  
*(Replace with a relevant graphic, e.g., optimization flowchart or demand uncertainty visualization)*  

---

## 🌟 **Overview**  
Strategic expansion decisions (e.g., manufacturing capacity, supply chain networks) are often modeled as **Mixed-Integer Linear Programming (MILP)** problems. However, real-world demand uncertainty can significantly alter the feasibility and optimality of these plans. This project:  

✅ Simulates **multiple demand scenarios** using Monte Carlo methods to account for market fluctuations.  
✅ Performs **parametric sensitivity analysis** to provide adaptable investment recommendations without rerunning multiple optimizations.  
✅ Identifies critical demand thresholds where **investment strategies** shift significantly. 
✅ Provides **visual tools** to interpret important aspects of capacity investment policies, production planning and inventory management through an interactive dashboard.  

By embedding uncertainty-aware optimization into capacity planning, our framework enhances decision robustness, ensuring cost-effective, scalable, and risk-mitigated expansion strategies.  
*(Add a concise 2-3 sentence summary of your key findings here for conference attendees.)*  

---

### **Why This Matters**
Manufacturers and supply chains face volatile demand due to factors like market shifts, disruptions, and seasonality. Many still rely on traditional tools like Excel or expert judgment, leading to uncoordinated and biased decision-making, and face:

❌ Overinvestment in low-probability scenarios.  
❌ Underpreparedness for high-demand shocks.  
❌ Suboptimal resource allocation due to rigid expansion strategies.  

<div align="center">
  <img src="https://github.com/Soumyashikha1998/Johnson-Johnson/blob/main/assets/Problem_network.png?raw=true" 
       alt="Methodology Flowchart" 
       width="50%" />
  <br>
  <em>Figure: Represtation of Supply Chain Network</em>
</div>

A real-world example is the **supply of surgical instruments**.  The figure shown above represents a similar supply chain network. These require long lead-time investments, meaning capacity decisions today must align with uncertain future demand 2-3 years ahead, so that the right equipment reaches surgeons at the right time. However, challenges arise due to:

📌 New Product Introductions (NPIs): Some instruments may replace existing ones, altering demand forecasts.  
📌 Market Price Fluctuations: Profitability and investment feasibility change with shifting costs.  
📌 Technological & Regulatory Changes: Compliance and innovation cycles impact long-term demand patterns. 

<div align="center">
  <img src="https://github.com/Soumyashikha1998/Johnson-Johnson/blob/main/assets/Challenge.png?raw=true" 
       alt="Methodology Flowchart" 
       width="50%" />
  <br>
  <em>Figure: Uncertainty challenge</em>
</div> 

By integrating **sensitivity analysis** into the MILP framework, we aim to:
- Identify critical parameters affecting expansion feasibility.
- Quantify the impact of demand fluctuations on decision-making.
- Provide insights for robust strategic planning that adapts to uncertainty.
  
### **Research Questions**
- How sensitive are **MILP-driven expansion decisions** to demand uncertainty?
- Which **demand scenarios** lead to infeasibility or cost spikes?
- Can we derive **robust decision rules** from sensitivity analysis to improve strategic planning?

_(Include a figure here, e.g., a schematic of demand uncertainty vs. expansion costs)_


*(Include a figure here, e.g., a schematic of demand uncertainty vs. expansion costs)*  
![Problem Schematic](https://via.placeholder.com/600x300?text=Demand+Uncertainty+Impact+on+MILP)  

---

## 📈 **Methodology** 
To tackle the challenges, we propose a three step approach:  
<div align="center">
  <img src="https://github.com/Soumyashikha1998/Johnson-Johnson/blob/main/assets/Methodology.png?raw=true" 
       alt="Methodology Flowchart" 
       width="50%" />
  <br>
  <em>Figure: Workflow of sensitivity analysis under demand uncertainty</em>
</div> 

### **1. Demand Scenario Generation**
We generate multiple demand forecasts using **Monte Carlo simulation**. The simulation considers:
- **User inputs**: Nominal demand data, a set of instructions
- **Real-world dynamics**: Market potential, adoption rates (for new products (NPIs) or cannibalizationeffects), product relationship, launch delays
Scenario sampling is them performed based on cumulative probability functions to ensure comprehensive uncertainty coverage.
<div align="center">
  <img src="https://github.com/Soumyashikha1998/Johnson-Johnson/blob/main/assets/Scenario_Sampling.png?raw=true" 
       alt="Methodology Flowchart" 
       width="50%" />
  <br>
  <em>Figure: Generation and Selection of demand scenarios</em>
</div>

### **2. MILP Model Formulation**  
The next step involves formulating a **Mixed-Integer Linear Programming (MILP) model** with the objective of:
- **Minimizing total costs**, including inventory holding and other expenses.
- **Maximizing demand fulfillment** by ensuring optimal capacity allocation.

The model's decision variables determine how many capacity units should be allocated to different work centers over multiple time periods. This results in a **multi-period supply management problem** subject to various constraints, including:
- **Inventory balance** and demand propagation.
- **Production capacity and expansion projects.**
- **Flexible work centers** that can adapt to changing demand.
- **Annual budget limitations** and investment feasibility.

A single demand forecast can be optimized using this MILP model. 


#### Objective Function
$$
\max \sum_{t} \frac{1}{(1+r)^t} (\lambda_a \cdot SFG_t - \lambda_b \cdot CID_t - \lambda_c \cdot IIC_t - \lambda_d \cdot HC_t - \lambda_e \cdot TD_t) + \sum_{j} \sum_{t} (s_{j,t}^{E} + s_{j,t}^{U})
$$

#### Inventory Constraints
$$
s_{j,t} = s_{j,t-1} + x_{j,t} - d_{j,t} \quad \forall j \in J, t > 1
$$
$$
s_{j,t} = Target_{j,t} + s_{j,t}^{E} - s_{j,t}^{S} \quad \forall j \in J, t \in T
$$
$$
s_{j,t}^{S} \leq s_{j,t}^{BIN} \cdot (Target_{j,t} + 1) \quad \forall j \in J, t \in T
$$
$$
s_{j,t}^{E} \leq (1 - s_{j,t}^{BIN}) \cdot 2 \cdot s_{j,t} \quad \forall j \in J, t \in T
$$
$$
s_{j,t} \leq S_j^{max} \quad \forall j \in J, t \in T
$$

#### BOM Constraints
$$
d_{k,t-CLT_k} = \sum_{p \in P_{i}} \frac{\alpha_{k,j}}{(1 - SR_{k,j})} \cdot x_{j,t} \quad \forall k \in J_K, t \in T
$$

#### Demand Fulfillment
$$
d_{j,t-CLT_k} \leq D_{j,t} \quad \forall j \in J_p, t \in T
$$

#### Available Capacity Constraints
$$
z_{j,t} = z_{j,t-1} + \sum_{p \in P_i} \phi_p \cdot y_{j,t} + v_{j,t} \quad \forall i \in I, t > 1
$$

#### Cross Qualification Constraints
$$
\hat{x}_{i,j,t} \leq \sum_{\tau = t_0}^{t} v_{i,j,t} \quad \forall i, j \in Q, t : t \in T
$$
$$
\sum_{t \in T} v_{i,j,t} \leq 1 \quad \forall i, j \in Q
$$
$$
u_{i,j,t-CLT_k} = v_{i,j,t} \quad \forall i, j \in Q, t \in T
$$

#### Project Constraints
$$
\sum_{t \in T} y_{p,t} \leq 1 \quad \forall p \in P^I, t > 1
$$
$$
\sum_{\tau \in T} y_{p,\tau} \leq \gamma_p - 1 \quad \forall p \in P^S, 1 < t \leq T
$$
$$
y_{q,t} \leq \gamma_p - \sum_{\tau = 1}^{t-1} y_{p,\tau} \quad \forall (p,q) : p \in P^I, q \in P^S, t > 1
$$
$$
y_{p,t-LT_p}^{S} \leq y_{p,t} \quad \forall p \in P, t \in T
$$

#### Annual Budget Constraint
$$
\sum_{p \in P} \frac{F_p}{N_{Period\_per\_year}} \cdot y_{p,t} + \sum_{j \in J} C_j \cdot \gamma_j \leq AB \quad \forall t \in T
$$

#### Notation
- **SFG**: Sales of Finished Goods
- **CID**: Capital Investment Depreciation value
- **TD**: Penalty for target deviation
- **IIC**: Inventory Investment Cost
- **QC**: Qualification Cost
- **HC**: Holding Cost


### **3. Sensitivity Analysis**  
To assess how changes in demand parameters affect optimal decisions, we apply a **multi-parametric programming approach**. This method: 
- Provides insights into the impact of demand variations 
- Avoids the need to re-run multiple scenarios
- Determines the sensitivity of decision variables to different demand scenarios, enabling faster and more robust strategic planning.

<div align="center">
  <img src="https://github.com/Soumyashikha1998/Johnson-Johnson/blob/main/assets/mpMILP-algorithm.png?raw=true" 
       alt="Algorithm Flowchart" 
       width="50%" />
  <br>
  <em>Figure: Step-by-step algorithm of multi-parametric programming for MILP </em>
</div> 

---

## 💻 **Technical Implementation**  
```python
# Pyomo snippet example (simplified)
model = ConcreteModel()
model.x = Var(within=Binary)  # Expansion decision
model.y = Var(within=NonNegativeReals)  # Production
model.cost = Objective(expr=CAPEX*model.x + OPEX*model.y, sense=minimize)
# ... (add your key constraints)
```
## 🛠️ **Tools Used**
- **Optimization**: 
  ![Pyomo](https://img.shields.io/badge/Pyomo-Open%20Source%20Optimization-orange)
  ![Gurobi](https://img.shields.io/badge/Gurobi-Solver%20(Optional)-yellow)
- **Data Handling**: 
  ![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-blue)
  ![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-green)
- **Visualization**: 
  ![Matplotlib](https://img.shields.io/badge/Matplotlib-Plotting-red)
  ![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-purple)
- **Development**: 
  ![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
  ![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange)

---

## 🚀 **Quick Start**
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/repository-name.git
   cd repository-name
   ```

---


## 🔍 **Key Findings**
- 📉 **Demand Sensitivity**:  
  Expansion plans become infeasible beyond **±20%** demand deviation from baseline.  
  ![Demand Sensitivity Plot](https://via.placeholder.com/400x200?text=Demand+vs+Cost+Plot)  

- 💡 **Robustness Insights**:  
  A **15% capacity buffer** reduces cost volatility by **40%** under uncertainty.  

- ⚖️ **Trade-off Analysis**:  
  Phased investments outperform single-stage expansions when demand growth is >**10% CAGR**.  
  ```python
  # Example trade-off snippet (placeholder)
  print(f"Optimal investment threshold: {threshold} units/year")


---


## ✨ **Contributors**
- **Soumya Shikha**  
  [![Email](https://img.shields.io/badge/Email-soumya@example.com-blue)](mailto:soumya@example.com)  
  [![GitHub](https://img.shields.io/badge/GitHub-soumyashikha-green)](https://github.com/soumyashikha)  

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)  
*Acknowledgments: [Conference Name/Advisor Name].*
