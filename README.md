# 📊 Sensitivity Analysis of MILP-Based Expansion Decisions Under Demand Uncertainty  
*A Data-Driven Approach for Strategic Capacity Management Planning*  

---

## 🌟 **Overview**  
Strategic expansion decisions (e.g., manufacturing capacity, supply chain networks) are often modeled as **Mixed-Integer Linear Programming (MILP)** problems. However, real-world demand uncertainty can significantly alter the feasibility and optimality of these plans. This project:  

✅ Simulates **multiple demand scenarios** using Monte Carlo methods to account for market fluctuations.  
✅ Performs **parametric sensitivity analysis** to provide adaptable investment recommendations without rerunning multiple optimizations.  
✅ Identifies critical demand thresholds where **investment strategies** shift significantly. 
✅ Provides **visual tools** to interpret important aspects of capacity investment policies, production planning and inventory management through an interactive dashboard.  

By embedding uncertainty-aware optimization into capacity planning, our framework enhances decision robustness, ensuring cost-effective, scalable, and risk-mitigated expansion strategies.  

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
\min \sum_{t} \frac{1}{(1+r)^t} (\lambda_a \cdot SFG_t - \lambda_b \cdot CID_t - \lambda_c \cdot IIC_t - \lambda_d \cdot HC_t - \lambda_e \cdot TD_t) + \sum_{j} \sum_{t} (s_{j,t}^{E} + s_{j,t}^{U})
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
\hat{x}_{i,j,t} \leq \sum_{\tau = t_0}^{t} v_{i,j,\tau} \quad \forall \; i, j \in Q,\; t \in T
$$

$$
\sum_{t \in T} v_{i,j,t} \leq 1 \quad \forall \; i, j \in Q
$$

$$
u_{i,j,t - \text{CLT}_k} = v_{i,j,t} \quad \forall \; i, j \in Q,\; t \in T
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
We can obtain the optimal expansion decisions from the MILP model foe each demand scenarios. To evaluate how sensitive these decisions are to variations in demand parameters, we apply a multi-parametric programming approach instead of solving multiple MILP optimizations for different demand realizations. This approach allows us to systematically explore the feasible solution space under different demand variations by identifying critical regions where the optimal decisions remain unchanged. This reduces computational complexity and provides decision-makers with direct insights into how investment strategies should adapt dynamically without re-solving the MILP for each scenario.  

To implement this, we follow an iterative parametric programming algorithm, as illustrated in the step-by-step schematic:

<i> **Step 0 (Initialization)** </i>:
Define an initial critical region (CR) with an upper bound for the objective function. Identify an initial integer solution from the MILP model.

<i> **Step 1 (Multiparametric LP Subproblem)** </i>:
Solve the multiparametric LP subproblem for each critical region to obtain parametric upper bounds. If a better feasible solution is found, update the best upper bound and the integer decision variables accordingly. If infeasibility arises, move to Step 2.

<i> **Step 2 (Master MILP Subproblem)** </i>:
Solve the MILP master problem for each region while treating demand uncertainty as a bounded variable. Introduce integer and parametric cuts to refine feasible solutions. Return to Step 1 with newly identified integer solutions and updated critical regions.

<i> **Step 3 (Convergence)** </i>:
The algorithm terminates when no feasible solution exists for further demand variations.
The final solution consists of critical regions with corresponding expansion decisions and optimal capacity investment plans.

There are several techniques for searching the parametric space and determining critical regions where optimal decisions shift:  

<i> Geometric Approach </i>:    
Constructs polyhedral partitions of the parametric space by explicitly solving the optimization problem at different demand levels.  
Computationally expensive for high-dimensional problems.

<i> Graph-based Approach </i>:
Models the solution space as a network where represnt feasible solutions, and edges depict transitions due to parametric changes.
Primarily used in transporattion network problems but less applicable to suplly chains.

<i> **Combinatorial Approach (Chosen Approach)** </i>:
Identifies integer-feasible regions by systematically enumerating integer solutions and evaluating their validity under different demand variations.  
Well suited for supply chain and manufacturing applications, where expansion decisions involve discrete investment choices.  
Efficient in handling large-scale problems with multiple constraints. By adopting the combinatorial approach, we ensure that our sensitivity analysis remains computationally tractable while providing structured decision rules for capacity expansion under demand uncertainty.

---

## 💻 **Technical Implementation**  
**Project Strucutre**
```bash
repository-name/
├── dashboard.py         # Main script to launch the dashboard
├── parametric2.py       # Computational model for the parametric programming implementation 
├── MILP model           # Folder containing the original MILP model codes for the supply chain network
├── datasets/            # Folder containing input data files
├── requirements.txt     # List of dependencies
└──  README.md           # Documentation
```

**Installation & Dependencies**
Ensure you have Python installed, then install the required packages using:
```bash
pip install -r requirements.txt
```
**How the Dashboard Works**
1. The dashboard is built using Streamlit, Matplotlib, Dash and Plotly for visualization.
2. When dashboard.py is executed, it:
Loads datasets from the datasets/ folder.
Calls parametric2.py to perform computations.
Generates interactive charts and tables for analysis.
3. To run the dashboard:
   ```bash
   streamlit run dashboard.py
   ```
**Core Functionalities**
Data Processing: parametric2.py loads and processes the dataset.  
Optimization Model: Uses MILP techniques to evaluate different scenarios.  
Visualization: The dashboard displays results using dynamic graphs and tables.  
**Customization & Extensibility**  
Modify Input Data: Update the files in the datasets/ folder.  
Change Model Parameters: Edit parametric2.py to adjust computational logic.  
Run the Dashboard: View charts and visuals.

---

## 🔍 **Key Findings**
1. Demand Scenario Analysis:
Demand variations were analyzed in a 80-120% range for a test problem with one product, three work centers, and four time periods.
The heatmap categorizes scenarios into Min (80%), Nominal (100%), and Max (120%).
<div align="center">
  <img src="https://github.com/Soumyashikha1998/Johnson-Johnson/blob/main/assets/demand_scenarios_heatmap.png?raw=true" 
       alt="Algorithm Flowchart" 
       width="50%" />
  <br>
  <em>Figure:  Demand Scenarios: Temporal Patterns & Hotspots </em>
</div>
3. Cost Variability Across Critical Regions (CRs):
The objective function follows piecewise-linear MILP solutions, defining cost behavior across CRs. CRs represent distinct, non-overlapping clusters where the optimal investment strategy remains stable. Cost shifts at CR boundaries highlight resource reallocation points critical for investment decisions.  
<div align="center">
  <img src="https://github.com/Soumyashikha1998/Johnson-Johnson/blob/main/assets/critical_region_plot.png?raw=true" 
       alt="Algorithm Flowchart" 
       width="50%" />
  <br>
  <em>Figure: Cost Function Variability across Critical Regions</em>
</div>
4. Capacity Investment Strategies:
Five optimal investment plans exist forour problem, each aligning with different demand scenarios. Plan E is the most probable (58%), while Plan A & D are least likely (3.7% each).  
The heatmap visualizes each plan indicating work center investments over time.  
<div align="center">
  <img src="https://github.com/Soumyashikha1998/Johnson-Johnson/blob/main/assets/investment_plans.png?raw=true" 
       alt="Algorithm Flowchart" 
       width="50%" />
  <br>
  <em>Figure: Capacity Investment Definitions and Selections </em>
</div>
5. Production Planning Insights:
Stacked bar graphs show production levels across time periods and demand scenarios. Helps refine strategies to meet peak demands efficiently.  
<div align="center">
  <img src="https://github.com/Soumyashikha1998/Johnson-Johnson/blob/main/assets/stacked_production.png?raw=true" 
       alt="Algorithm Flowchart" 
       width="50%" />
  <br>
  <em>Figure: Production Levels for Demand Variability </em>
</div>
6. Scenario-Based Investment Optimization:
Model suggests optimal plans based on demand trends across time periods. Take a tour of the dashboard, and try playing with scenarios. If demand is anticipated to be in the mid-lower range in the first two periods, and in mid-higher range lower in the third and fourth periods. The model suggests the best investment plans accordingly. This is where you can experiment with different scenarios and see which investment plans suit your case. For this example, the optimizer suggests going for Plan E mostly and maybe Plan B. Scenario experimentation allows businesses to adapt investment decisions dynamically.
<div align="center">
  <img src="https://github.com/Soumyashikha1998/Johnson-Johnson/blob/main/assets/investment_plans_bubbles.png?raw=true" 
       alt="Algorithm Flowchart" 
       width="50%" />
  <br>
  <em>Figure: Investment Planner</em>
</div>
---

## 🛠️ **Tools Used**
- **Data Handling**: 
  ![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-blue)
  ![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-green)
- **Optimization**: 
  ![Pyomo](https://img.shields.io/badge/Pyomo-Open%20Source%20Optimization-orange)
  ![Gurobi](https://img.shields.io/badge/Gurobi-Solver%20(Optional)-yellow)
- **Visualization**:
  ![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-green)
  ![Matplotlib](https://img.shields.io/badge/Matplotlib-Plotting-red)
  ![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-purple)
- **Development**: 
  ![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
---

## 🚀 **Quick Start**
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/repository-name.git
   cd repository-name
   ```
2. **Download Required Files**:
   Ensure you have the following files and folder in your working directory:
   ```bash
   parametric2.py
   ```
   ```bash
   dashboard.py
   ```
   ```bash
   datasets/ #(folder containing necessary data)
   ``` 
3. **Run the Dashboard**:
   Execute the following command to launch the interactive dashboard:
   ```bash
   streamli run dashboard.py
   ```
   This will start a local server, and you can view the dashboard in your browser.

---

## ✨ **Contributors**
- **Soumya Shikha**  (Lead Researcher)
  [![Email](https://img.shields.io/badge/Email-sshikha@alumni.cmu.edu-blue)](mailto:sshikha@alumni.cmu.edu)  
  [![GitHub](https://img.shields.io/badge/GitHub-Soumyashikha1998-green)](https://github.com/Soumyashikha1998)  

- **Dr. Ignacio Grossmann**  (Advisor)

- **Johnson & Johnson Digital Supply Team** (Industry Partner)
  
<i> INFORMS ANALYTICS+ CONFERENCE, APRIL 6-8 </I>
