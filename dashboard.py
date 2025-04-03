
###############-----SENSITIVITY ANALYSIS OF MILP-BASED CAPABILITY EXPANSION DECISIONS-----###############
######################----------------UNDER DEMAND UNCERTAINTY----------------######################




##########################-----------------DASHBOARD EXECUTION-----------------##########################
















###-------UPLOADING THE DATASETS ON DASHBOARD----------
####Please be patient, to see a demo try one scenario only
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter
from parametric2 import datainput,solve_initial_feasible_milp,solve_multiparametric_lp,iterative_MILP

np.set_printoptions(precision=2)
####-----------------------------TITLE-----------------------------####
st.set_page_config(page_title="Dashboard", 
                   layout="wide")
st.markdown("""
    <h1 style='text-align: left; font-size: 38px;'> Optimization-based <span style='color:orange;'>Sensitivity Analysis</span> for Capacity Expansion Decision Making Under Demand  
    Uncertainty </h1>
""", unsafe_allow_html=True)

st.markdown("""
            <h2 style='text-align: center; font-size: 24px;'><span style='color:lightblue;'>Optimize and Analyze: </span>
            Leverage Data-Driven Insights for Strategic Capacity Management</h2>""", unsafe_allow_html=True)

# Markdown with highlighted section in yellow
st.markdown("""
    <div style='padding: 10px; font-size: 16px;'>
    Welcome to the <span style='color:#800080; font-weight: bold;'>Capacity Management Planning</span> Dashboard. 
    This research tackles the complex challenge of prioritizing investments for capacity expansion amidst uncertain market conditions. 
    It introduces a Mixed-Integer Linear Programming (MILP) model integrated with Monte Carlo simulation to identify the impactful 
    investments optimizing demand coverage and minimizing costs. Parametric Programming approach is employed to develop a comprehensive framework 
    for evaluating investment decisions across uncertain demand scenarios, ensuring alignment with both short-term operational needs and long-term strategic goals 
    </div>
""", unsafe_allow_html=True)

# Add a subheader with more details



st.sidebar.markdown("""
    <div style="
        font-family: 'Arial', sans-serif;
        font-size: 24px;
        color: #f0f0f0;
        background-color: #000080;
        text-align: center;
        text-shadow: 1px 1px 2px grey;
        padding: 10px;
        border-radius: 5px;
        ">
        <strong>DASHBOARD WALKTHROUGH ⋮</strong>
    </div>
    """, unsafe_allow_html=True)
st.sidebar.markdown('###')
st.sidebar.markdown("""
    <div style="font-family: 'Courier New', serif; font-size: 20px; color: #8B0000; text-shadow: 1px 1px 2px grey;">
                    <b>› Build your own Supply Chain Network</b>
                    </div>""", unsafe_allow_html = True)
st.sidebar.markdown("""
    <div style="font-family: 'Courier New', monospace; font-size: 16px; color: #000000;">
                    • Upload Datasets/Parameters
                    </div>""", unsafe_allow_html = True)
st.sidebar.markdown("""
    <div style="font-family: 'Courier New', monospace; font-size: 16px; color: #000000;">
                    • Define Demand Uncertainty
                    </div>""", unsafe_allow_html = True)
st.sidebar.markdown("""
    <div style="font-family: 'Courier New', monospace; font-size: 16px; color: #000000;">
                    • Configure Parametric Programming Steup
                    </div>""", unsafe_allow_html = True)

st.sidebar.markdown('###')
st.sidebar.markdown("""
    <div style="font-family: 'Courier New', serif; font-size: 20px; color: #00008B; text-shadow: 1px 1px 2px grey;">
                    <b>› Visualize your Selections </b>
                    </div>""", unsafe_allow_html = True)
st.sidebar.markdown("""
    <div style="font-family: 'Courier New', monospace; font-size: 16px; color: #000000;">
                    • Demand Scenarios Heat-map
                    </div>""", unsafe_allow_html = True)
st.sidebar.markdown("""
    <div style="font-family: 'Courier New', monospace; font-size: 16px; color: #000000;">
                    • Objective Function Variablity across Critical Regions
                    </div>""", unsafe_allow_html = True)
st.sidebar.markdown('###')
st.sidebar.markdown("""
    <div style="font-family: 'Courier New', serif; font-size: 20px; color: #00008B; text-shadow: 1px 1px 2px grey;">
                    <b>› Discover Key Insights</b>
                    </div>""", unsafe_allow_html = True)
st.sidebar.markdown("""
    <div style="font-family: 'Courier New', serif; font-size: 16px; color: #000000;">
                    • Capacity Expansion Plans & High Probability Selections
                    </div>""", unsafe_allow_html = True)
st.sidebar.markdown("""
    <div style="font-family: 'Courier New', monospace; font-size: 16px; color: #000000;">
                    • Strategic Production Levels 
                    </div>""", unsafe_allow_html = True)

st.sidebar.markdown('###')
st.sidebar.markdown("""
    <div style="font-family: 'Courier New', serif; font-size: 20px; color: #00008B; text-shadow: 1px 1px 2px grey;">
                    <b>› Analyze the Sensitivity</b>
                    </div>""", unsafe_allow_html = True)
st.sidebar.markdown("""
    <div style="font-family: 'Courier New', monospace; font-size: 16px; color: #000000;">
                    • Interactive Investment Planner
                    </div>""", unsafe_allow_html = True)

st.sidebar.markdown('###')
st.sidebar.markdown("""
    <div style="font-family: 'Courier New', serif; font-size: 20px; color: #8B4513;">
                    <b>›Additional Resources</b>
                    </div>""", unsafe_allow_html = True)


if 'startclicked' not in st.session_state:
        st.session_state.startclicked = False

def click_start():
        st.session_state.startclicked = True

st.sidebar.markdown('####')
st.sidebar.button('Click to start', on_click=click_start)

####-----------------------------UPLOADING DATASETS-----------------------------####
data = {}
if st.session_state.startclicked:
    st.markdown(f"""
                    <div style="
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        padding: 3px;
                        background-color: #ADD8E6;
                        text-align: center;
                        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                        margin: 3px;
                        height: 50px;  
                        overflow: hidden; 
                        ">
                        <h2 style="margin: 0; font-size: 24px; color: #333;"><svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#5f6368"><path d="M160-120q-33 0-56.5-23.5T80-200v-560q0-33 23.5-56.5T160-840h240q33 0 56.5 23.5T480-760v80h320q33 0 56.5 23.5T880-600v400q0 33-23.5 56.5T800-120H160Zm0-80h240v-80H160v80Zm0-160h240v-80H160v80Zm0-160h240v-80H160v80Zm0-160h240v-80H160v80Zm320 480h320v-400H480v400Zm120-240q-17 0-28.5-11.5T560-480q0-17 11.5-28.5T600-520h80q17 0 28.5 11.5T720-480q0 17-11.5 28.5T680-440h-80Zm0 160q-17 0-28.5-11.5T560-320q0-17 11.5-28.5T600-360h80q17 0 28.5 11.5T720-320q0 17-11.5 28.5T680-280h-80Z"/></svg> BUILD YOUR OWN SUPPLY CHAIN NETWORK <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#5f6368"><path d="M160-120q-33 0-56.5-23.5T80-200v-560q0-33 23.5-56.5T160-840h240q33 0 56.5 23.5T480-760v80h320q33 0 56.5 23.5T880-600v400q0 33-23.5 56.5T800-120H160Zm0-80h240v-80H160v80Zm0-160h240v-80H160v80Zm0-160h240v-80H160v80Zm0-160h240v-80H160v80Zm320 480h320v-400H480v400Zm120-240q-17 0-28.5-11.5T560-480q0-17 11.5-28.5T600-520h80q17 0 28.5 11.5T720-480q0 17-11.5 28.5T680-440h-80Zm0 160q-17 0-28.5-11.5T560-320q0-17 11.5-28.5T600-360h80q17 0 28.5 11.5T720-320q0 17-11.5 28.5T680-280h-80Z"/></svg></h2>
                        </div>
                        """, unsafe_allow_html = True)

    st.markdown(f"""
                <div style="
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 3px;
                    background-color: #f9f9f9;
                    text-align: left;
                    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                    margin: 3px;
                    height: 50px;  
                    overflow: hidden; 
                ">
                <h2 style="margin: 0; font-size: 20px; color: #333;">⋮ Upload datasets</h2>
                </div>
                """, unsafe_allow_html=True)
    with st.expander('Sample Input file'):
        columna, columnb, columnc = st.columns(3)
        with columna:
            st.download_button(
                            label="Sample Work Centers Dataset",
                            data="../dataset/parameters_work_centers.xlsx",
                            file_name='parameters_work_centers.xlsx',
                            mime='text/xlsx'
                        )
        with columnb:
            st.download_button(
                            label="Sample Product Dataset",
                            data="../dataset/parameters_products.xlsx",
                            file_name='parameters_products.xlsx',
                            mime='text/xlsx'
                        )

        with columnc:
            st.download_button(
                            label="Sample Products_Demand Dataset",
                            data="../dataset/parameters_products_demand.xlsx",
                            file_name='parameters_products_demand.xlsx',
                            mime='text/xlsx'
                        )

        
    columna, columnb, columnc = st.columns(3)

    with columna:
        work_centers_file = st.file_uploader('Upload Work Centers Dataset', type = 'xlsx')
        #vehicles_fuels_file = r'C:\Users\user\Documents\GitHub\shell_hackathon\dataset\vehicles_fuels.xlsx'
        if work_centers_file:
            work_centers = pd.read_excel(work_centers_file)
            data['work_centers'] = work_centers
            st.markdown('<b>Work_centers dataset uploaded ✅</b>', unsafe_allow_html = True)

    with columnb:
        products_file = st.file_uploader('Upload Products Dataset', type = 'xlsx')
            #vehicles_file = r'C:\Users\user\Documents\GitHub\shell_hackathon\dataset\vehicles.xlsx'
        if products_file:
            products = pd.read_excel(products_file)
            data['products'] = products
            st.markdown('<b> Products dataset uploaded ✅</b>', unsafe_allow_html = True)
    
    with columnc:
        products_demand_file = st.file_uploader('Upload Products_Demand Dataset', type = 'xlsx')
            #vehicles_file = r'C:\Users\user\Documents\GitHub\shell_hackathon\dataset\vehicles.xlsx'
    if products_demand_file:
        products_demand = pd.read_excel(products_demand_file)
        data['products_demand'] = products_demand
        with columnc:
            st.markdown('<b> Products_Demand dataset uploaded ✅</b>', unsafe_allow_html = True)

########-----------------------------SELECTING THE DEMAND AND UNCERTAINTY-----------------------------####


        st.markdown('###')

        if 'clicked' not in st.session_state:
                st.session_state.clicked = False

        def click_button():
                st.session_state.clicked = True


        st.markdown(f"""
                        <div style="
                            border: 1px solid #ddd;
                            border-radius: 8px;
                            padding: 3px;
                            background-color: #f9f9f9;
                            text-align: left;
                            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                            margin: 3px;
                            height: 50px;  
                            overflow: hidden; 
                        ">
                        <h2 style="margin: 0; font-size: 20px; color: #333;">⋮ Define Demand Uncertainty </h2>
                        </div>
                        """, unsafe_allow_html=True)
                
            
        if 'clicked' not in st.session_state:
                    st.session_state.clicked = False

        def click_button():
                    st.session_state.clicked = True

        col1, col2 = st.columns(2)
        Min_value = '0'
        Min_value = '120'
        with col1:
                    model = st.selectbox("Choose uncertainty range for products demand", options = ["None","0-40%", "40-80%", "80-120%" ], index=0)
                    if model == "None":
                        st.warning("Please select a range")
                    else:
                        st.success(f"You selected: {model} coverage")
        if model == '0-40%':
            data['Min_value'] = 0
            data['Max_value'] = 40
            with col2:
                    probd = st.selectbox("Choose Probability Distribution", options = ["None","Uniform"], index=0)
                    if probd == "None":
                                  st.warning("Please select an option")
                                #   st.button('Solve', on_click=click_button)
                    else:
                                  st.success(f"You selected: {probd}")

            
        if model == '40-80%':
            data['Min_value'] = 40
            data['Max_value'] = 80
            with col2:
                    probd = st.selectbox("Choose Probability Distribution", options = ["None","Uniform"], index=0)
                    if probd == "None":
                                  st.warning("Please select an option")
                                #   st.button('Solve', on_click=click_button)
                    else:
                                 st.success(f"You selected: {probd} Distribution")

        if model == '80-120%':
            data['Min_value'] = 80
            data['Max_value'] = 120
            with col2:
                probd = st.selectbox("Choose Probability Distribution", options = ["None","Uniform"], index=0)
            if probd == "None":
                    with col2:
                                  st.warning("Please select an option")
                                #   st.button('Solve', on_click=click_button)
            if probd == "Uniform":
                with col2:
                    st.success(f"You selected: {probd} Distribution")
                
            if 'clicked' not in st.session_state:
                        st.session_state.clicked = False

            def click_button():
                        st.session_state.clicked = True
                    
            st.markdown(f"""
                            <div style="
                            border: 1px solid #ddd;
                            border-radius: 8px;
                            padding: 3px;
                            background-color: #f9f9f9;
                            text-align: left;
                                        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                                        margin: 3px;
                                        height: 50px;  
                                        overflow: hidden; 
                                        ">
                                        <h2 style="margin: 0; font-size: 20px; color: #333;">⋮ Configure the Parametric Programming Setup </h2>
                                        </div>
                                        """, unsafe_allow_html=True)
                                  
            with st.expander('About Parametric Programming Algorithm and Space Search Approach', icon = ':material/browse_activity:'):
                    explanation = """
                    <div style="font-size: 14px; line-height: 1.6; text-align: justify;">
                    <b>- Parametric Programming Algorithm:</b><br>
                    A problem-solving approach where solutions change as input parameters vary. 
                    Used extensively in optimization to understand how outputs change with shifting conditions.
                    Key Types: <br>
                    Geometric (common for 3D modeling, graphics), 
                    Combinatorial (common for scheduling, supply chains), 
                    Graph-based (common for transportation networks)<br>
                    <b>- Parametric Space Search Approach:</b><br>
                    A technique to efficiently find optimal solutions by exploring how parameters influence outcomes. 
                    It evaluates points where solutions change fundamentally.
                    Common Approaches:<br>
                    Continuous: Sensitivity analysis;
                    Discrete: Scenario-based; 
                    Binary: Yes/No decisions
                    </div>
                    """
                    st.write(explanation, unsafe_allow_html=True)
            if 'clicked' not in st.session_state:
                    st.session_state.clicked = False

            def click_button():
                    st.session_state.clicked = True
                               
            col1,col2 = st.columns(2)
            with col1:                    
                algorithm= st.selectbox("Select parametric programming algorithm",options = ["None","Geometrical","Combinatorial","Graph-Based"], index=0)

                if algorithm == "None":
                                  st.warning("Please select an algorithm. Default option is Geometrical")               
                if algorithm == "Geometrical":
                                st.warning("Not Available")
                if algorithm == "Graph-Based":
                                st.warning("Not Available")  
            if  algorithm == "Combinatorial":
                with col1:
                    st.success(f"You selected: {algorithm} Algorithm")
                with col2:
                    approach = st.selectbox("Select parametric space search approach",options = ["None","Discrete","Continuous","Binary"], index=0)
                    if approach == "None":
                       st.warning("Please select an approach. Default option is Discrete")
                    if approach == "Continuous":
                       st.warning("Not Available")
                if approach == "Discrete":
                    with col2:
                           st.success(f"You selected: {approach} Approach")
                    st.button('Solve', on_click=click_button)
                    
                          
                                  
                                
         
########----------------------------------------- SOLVING THE MODEL ------------------------------------------########
            if st.session_state.clicked:
                solving_message = st.empty()
                solving_message.write("Solving for results...")
                periods, products, c,r,h,s0, machines, C, Cost_m, Fix_InvCost_m, Var_InvCost_m, d_nom, d_min, d_max, M = datainput(data)
                y_bin_insolution,y_insolution,y_l_insolution = solve_initial_feasible_milp(data)
                critical_regions,infeasible_regions,df_cr= solve_multiparametric_lp(data,y_bin_insolution,y_insolution,y_l_insolution)
                demand_values, y_solution, y_bin_solution, obj, df = iterative_MILP(data,df_cr, critical_regions,y_bin_insolution,y_insolution,y_l_insolution)
                 
                # if "df" not in st.session_state:
                # # if st.session_state.clicked:     
                #     #  st.write("Model Computation Ran")
                #     solving_message = st.empty()
                #     periods, products, c,r,h,s0, machines, C, Cost_m, Fix_InvCost_m, Var_InvCost_m, d_nom, d_min, d_max, M = datainput(data)
                #     y_bin_insolution,y_insolution,y_l_insolution = solve_initial_feasible_milp(data)
                #     critical_regions,infeasible_regions,df_cr= solve_multiparametric_lp(data,y_bin_insolution,y_insolution,y_l_insolution)
                #     demand_values, y_solution, y_bin_solution, obj, df_result = iterative_MILP(data,df_cr, critical_regions,y_bin_insolution,y_insolution,y_l_insolution)
                #     st.session_state.df = df_result
                    
                    # df = st.session_state.df
                # st.text('The sub-final file')
                # st.write(df_cr)
                # st.text('The final file')
                # st.write(df)
                # st.write("df type before modification:", type(df))
                solving_message.empty()

########----------------------------- GRAPHICS PRESENTATION & ANALYSIS -----------------------------########
                st.markdown('---')
                # Ia,IIa,IIIa = st.columns((0.00000001,0.99999998,0.00000001))
                # with IIa:


                st.markdown(f"""
                        <div style="
                            border: 1px solid #ddd;
                            border-radius: 8px;
                            padding: 3px;
                            background-color: #ADD8E6;
                            text-align: center;
                            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                            margin: 3px;
                            height: 50px;  
                            overflow: hidden; 
                        ">
                        <h2 style="margin: 0; font-size: 24px; color: #333;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-gpu-card" viewBox="0 0 16 16">
  <path d="M4 8a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0m7.5-1.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3"/>
  <path d="M0 1.5A.5.5 0 0 1 .5 1h1a.5.5 0 0 1 .5.5V4h13.5a.5.5 0 0 1 .5.5v7a.5.5 0 0 1-.5.5H2v2.5a.5.5 0 0 1-1 0V2H.5a.5.5 0 0 1-.5-.5m5.5 4a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5M9 8a2.5 2.5 0 1 0 5 0 2.5 2.5 0 0 0-5 0"/>
  <path d="M3 12.5h3.5v1a.5.5 0 0 1-.5.5H3.5a.5.5 0 0 1-.5-.5zm4 1v-1h4v1a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5"/></svg>                            
                            Visualize Your Selections
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-gpu-card" viewBox="0 0 16 16">
  <path d="M4 8a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0m7.5-1.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3"/>
  <path d="M0 1.5A.5.5 0 0 1 .5 1h1a.5.5 0 0 1 .5.5V4h13.5a.5.5 0 0 1 .5.5v7a.5.5 0 0 1-.5.5H2v2.5a.5.5 0 0 1-1 0V2H.5a.5.5 0 0 1-.5-.5m5.5 4a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5M9 8a2.5 2.5 0 1 0 5 0 2.5 2.5 0 0 0-5 0"/>
  <path d="M3 12.5h3.5v1a.5.5 0 0 1-.5.5H3.5a.5.5 0 0 1-.5-.5zm4 1v-1h4v1a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5"/></svg></h2>                        </div>
                        """, unsafe_allow_html=True)

                # Ia,IIb = st.columns((0.9999999,0.00000001))
                # with Ia:
########1. Heat map########
                # st.markdown('###')
                st.markdown(f"""
                        <div style="
                            border: 1px solid #ddd;
                            border-radius: 8px;
                            padding: 3px;
                            background-color: #f9f9f9;
                            text-align: center;
                            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                            margin: 3px;
                            height: 50px;  
                            overflow: hidden; 
                        ">
                        <h2 style="margin: 0; font-size: 20px; color: #333;"> Demand Scenarios: Temporal Patterns and Hotspots </h2>
                        </div>
                        """, unsafe_allow_html=True)
                # st.markdown(""" <div
                #         style="font-size: 20px; text-align: center;">
                #         <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#5f6368"><path d="M320-800q-17 0-28.5-11.5T280-840q0-17 11.5-28.5T320-880h215q20 0 29.5 12.5T574-840q0 15-10 27.5T534-800H320ZM200-640q-17 0-28.5-11.5T160-680q0-17 11.5-28.5T200-720h175q20 0 29.5 12.5T414-680q0 15-10 27.5T374-640H200ZM80-480q-17 0-28.5-11.5T40-520q0-17 11.5-28.5T80-560h135q20 0 29.5 12.5T254-520q0 15-10 27.5T214-480H80Zm504 80L480-504 280-304l104 104 200-200Zm-47-161 104 104 199-199-104-104-199 199Zm-84-28 216 216-229 229q-24 24-56 24t-56-24l-2-2-14 14q-6 6-13.5 9t-15.5 3H148q-14 0-19-12t5-22l92-92-2-2q-24-24-24-56t24-56l229-229Zm0 0 227-227q24-24 56-24t56 24l104 104q24 24 24 56t-24 56L669-373 453-589Z"/></svg> 
                            # Use this for normal font without a heading bar
                #         </div>
                #         """, unsafe_allow_html = True) 
                theta_cols = ["Theta 1", "Theta 2", "Theta 3", "Theta 4"]

                df["Scenario Name"] = pd.Categorical(df["Scenario Name"], 
                                     categories=sorted(df["Scenario Name"].unique(), key=lambda x: int(x[2:])),
                                     ordered=True)

                df_melted = df.melt(id_vars=["Scenario Name"], value_vars=theta_cols,
                        var_name="Theta", value_name="Value")

                df_pivot = df_melted.pivot(index="Theta", columns="Scenario Name", values="Value")
                df_pivot = df_pivot.reindex(columns=df["Scenario Name"].unique(), fill_value=0)

                colors = [(0, 0.2, 0.6), (1, 1, 0.9), (0.8, 0, 0)]
                cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
                vmin, vmax = df_pivot.min().min(), df_pivot.max().max()
                center = (vmax + vmin) / 2

                cbar_ticks = [vmin, center, vmax]
                cbar_labels = ["Min (80%)", "Nominal (100%)", "Max (120%)"]
                box_size = 0.3  # Size of each square in inches
                fig_width = box_size * len(df_pivot.columns)
                fig_height = 2 * box_size * len(df_pivot.index)


                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                sns.heatmap(df_pivot, cmap=cmap, center=center, annot=False, fmt=".0f",
                linewidths=0.1, ax=ax, cbar_kws={'label': 'Demand Level','ticks': cbar_ticks})
                    
                cbar = ax.collections[0].colorbar
                cbar.set_ticklabels(cbar_labels)
                cbar.outline.set_edgecolor('black')
                cbar.ax.yaxis.label.set_fontproperties({'weight': 'bold'})
                for label in cbar.ax.get_yticklabels():
                    label.set_fontproperties({'weight': 'bold'})
                                                       
                    ax.set_xlabel("Demand Scenario Combinations", fontsize=12, fontweight="bold")
                    ax.set_ylabel("Planning periods", fontsize=12, fontweight="bold")
                    ax.set_title("Heatmap of Theta Values Across Scenarios", fontsize=14, fontweight="bold")

                    ax.set_yticklabels([f"t{t}" for t in periods], rotation=0,
                                   fontproperties={'weight': 'bold', 'size': 10})
                    ax.set_xticks(range(len(df_pivot.columns)))
                    ax.set_xticklabels(df_pivot.columns, rotation=90, ha='center', 
                                    fontproperties={'family': 'Times New Roman', 'size': 8})
                    ax.set_title("Demand Scenarios Heatmap", pad=20, 
                                fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 16})

                st.pyplot(fig)
                st.markdown('---')

########2. Criitcal Region v/s Obj function########
                
                col1,col2 = st.columns((0.75,0.25))
                with col2:
                      cr_explanation = """
<div style="font-size: 16px; line-height: 1.6; text-align: justify;">
    <b>Interpretation:</b><br>
    This plot reveals how the objective function (total cost) behaves <i>discretely</i> across 
    Critical Regions (CRs) — distinct, non-overlapping demand-scenario clusters where the 
    optimal investment strategy remains consistent. Costs jump at CR boundaries, reflecting 
    sudden shifts in resource allocation. Use these transitions to identify cost-sensitive 
    decision points.
</div>
"""
                      st.write(cr_explanation, unsafe_allow_html=True)
                with col1:
                    st.markdown(f"""
                        <div style="
                            border: 1px solid #ddd;
                            border-radius: 8px;
                            padding: 3px;
                            background-color: #f9f9f9;
                            text-align: center;
                            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                            margin: 3px;
                            height: 50px;  
                            overflow: hidden; 
                        ">
                        <h2 style="margin: 0; font-size: 20px; color: #333;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-graph-up-arrow" viewBox="0 0 16 16">
  <path fill-rule="evenodd" d="M0 0h1v15h15v1H0zm10 3.5a.5.5 0 0 1 .5-.5h4a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-1 0V4.9l-3.613 4.417a.5.5 0 0 1-.74.037L7.06 6.767l-3.656 5.027a.5.5 0 0 1-.808-.588l4-5.5a.5.5 0 0 1 .758-.06l2.609 2.61L13.445 4H10.5a.5.5 0 0 1-.5-.5"/></svg>
                                 Cost Function Variability Across Critical Regions <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-graph-up-arrow" viewBox="0 0 16 16">
  <path fill-rule="evenodd" d="M0 0h1v15h15v1H0zm10 3.5a.5.5 0 0 1 .5-.5h4a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-1 0V4.9l-3.613 4.417a.5.5 0 0 1-.74.037L7.06 6.767l-3.656 5.027a.5.5 0 0 1-.808-.588l4-5.5a.5.5 0 0 1 .758-.06l2.609 2.61L13.445 4H10.5a.5.5 0 0 1-.5-.5"/></svg></h2>
                        </div>
                        """, unsafe_allow_html=True)
                    # st.markdown(""" <div
                    #     style="font-size: 20px; text-align: center;">
                    #     <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#5f6368"><path d="M320-800q-17 0-28.5-11.5T280-840q0-17 11.5-28.5T320-880h215q20 0 29.5 12.5T574-840q0 15-10 27.5T534-800H320ZM200-640q-17 0-28.5-11.5T160-680q0-17 11.5-28.5T200-720h175q20 0 29.5 12.5T414-680q0 15-10 27.5T374-640H200ZM80-480q-17 0-28.5-11.5T40-520q0-17 11.5-28.5T80-560h135q20 0 29.5 12.5T254-520q0 15-10 27.5T214-480H80Zm504 80L480-504 280-304l104 104 200-200Zm-47-161 104 104 199-199-104-104-199 199Zm-84-28 216 216-229 229q-24 24-56 24t-56-24l-2-2-14 14q-6 6-13.5 9t-15.5 3H148q-14 0-19-12t5-22l92-92-2-2q-24-24-24-56t24-56l229-229Zm0 0 227-227q24-24 56-24t56 24l104 104q24 24 24 56t-24 56L669-373 453-589Z"/></svg> 
                    #         Cost Function Variability Across Critical Regions
                    #     </div>
                    #     """, unsafe_allow_html = True) 
                    CR_data_1 = {
                    "CR1": [7628, 7993, 8273, 8363, 8638, 9008],
                    "CR2": [7948, 8268, 8305, 8313, 8348, 8433, 8593, 8625, 
            8633, 8668, 8675, 8683, 8705, 8713, 8753, 8753, 
            8798, 8950, 8958, 8993, 8995, 9003, 9025, 9033, 
            9035, 9073, 9075, 9078, 9083, 9110, 9118, 9118, 
            9168, 9320, 9328, 9350, 9355, 9358, 9395, 9398, 
            9403, 9430, 9435, 9438, 9443, 9480, 9488, 9488, 
            9680, 9720, 9728, 9755, 9755, 9763, 9800, 9808, 
            9813, 9840, 10080, 10125, 10133, 10160, 10485],
            "CR3": [9408, 9083, 9038, 8763, 8713, 8683, 
            8673, 8393, 8348, 8313, 8028, 7948]
                                }
                
                    base_width = 1.0
                    width_factors = {CR: base_width * (1 + 0.02*len(values)) for CR, values in CR_data_1.items()}
                    x_positions = {}
                    current_x = 0
                    for CR in CR_data_1:
                        x_positions[CR] = current_x + width_factors[CR]/2
                        current_x += width_factors[CR]
                    fig, ax = plt.subplots(figsize=(18, 6))
#colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Distinct colors
                    colors = ["#00BFFF", "#4169E1", "#87CEEB"]
                    endpoints = {}
                    for (CR, values), color in zip(CR_data_1.items(), colors):
                        x_min = x_positions[CR] - width_factors[CR]/2
                        x_max = x_positions[CR] + width_factors[CR]/2
                        x = np.linspace(x_min, x_max, len(values))

                        cr_index = CR[2:]  # Removes "CR" prefix
                        label = rf"$\hat{{z}}(\theta)^{{{cr_index}}}$"
    
                        ax.scatter(x, values, color=color, label=label, alpha=0.7)
                        ax.plot(x, values, color=color, alpha=0.3, linestyle='-')
                        endpoints[CR] = (x[-1], values[-1], x[0], values[0])

                    ax.set_xticks(list(x_positions.values()), list(x_positions.keys()), fontsize=12, fontweight='bold')
                    ax.set_xlabel("Critical Regions", fontsize=14,fontweight="bold")
                    ax.set_ylabel(r'Objective Value ($\hat{z}(\theta)$)', fontsize=14,fontweight="bold")
                    # ax.set_title("Cost Variation in each Critical Region", fontsize=16, fontweight='bold', pad=20)
                    ax.grid(True, linestyle=':', alpha=0.4)

                    ax.set_xlim(0, current_x)
                    ax.legend(title="Explicit functions", fontsize=12, title_fontsize=13, loc="upper left")

                    plt.tight_layout()
                    st.pyplot(fig)
                    note = """
<div style="font-size: 14px; line-height: 1.6; text-align: justify;">
    <b>Note:</b><br>
    <i>The critical Regions arise from piecewise-linear MILP solutions. Each CR's cost function is defined, and the solution is piecewise due to constraint activations.</i>
</div>
"""
                    st.write(note, unsafe_allow_html=True)

                # Ia,IIa,IIIa = st.columns((0.00000001,0.99999998,0.00000001))
                # with IIa:
                # st.markdown(f"""
                #         <div style="
                #             border: 1px solid #ddd;
                #             border-radius: 8px;
                #             padding: 3px;
                #             background-color: #ADD8E6;
                #             text-align: center;
                #             box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                #             margin: 3px;
                #             height: 50px;  
                #             overflow: hidden; 
                #         ">
                #         <h2 style="margin: 0; font-size: 24px; color: #333;"><svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#5f6368">
                #             <path d="M160-120q-33 0-56.5-23.5T80-200v-200q0-33 23.5-56.5T160-480h166q15 0 26 10t13 24q5 34 40 60t75 26q40 0 75-26t40-60q2-14 13-24t26-10h166q33 0 56.5 23.5T880-400v200q0 33-23.5 56.5T800-120H160Zm0-80h640v-200H664q-25 55-74.5 87.5T480-280q-60 0-109.5-32.5T296-400H160v200Zm516-356q-11-11-11-28t11-28l86-86q11-11 28-11t28 11q11 11 11 28t-11 28l-86 86q-11 11-28 11t-28-11Zm-392 0q-11 11-28 11t-28-11l-86-86q-11-11-11-28t11-28q11-11 28-11t28 11l86 86q11 11 11 28t-11 28Zm196-84q-17 0-28.5-11.5T440-680v-120q0-17 11.5-28.5T480-840q17 0 28.5 11.5T520-800v120q0 17-11.5 28.5T480-640ZM160-200h640-640Z"/></svg> 
                #             DASHBOARD: Visualizing the big picture 
                #             <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#5f6368">
                #             <path d="M160-120q-33 0-56.5-23.5T80-200v-200q0-33 23.5-56.5T160-480h166q15 0 26 10t13 24q5 34 40 60t75 26q40 0 75-26t40-60q2-14 13-24t26-10h166q33 0 56.5 23.5T880-400v200q0 33-23.5 56.5T800-120H160Zm0-80h640v-200H664q-25 55-74.5 87.5T480-280q-60 0-109.5-32.5T296-400H160v200Zm516-356q-11-11-11-28t11-28l86-86q11-11 28-11t28 11q11 11 11 28t-11 28l-86 86q-11 11-28 11t-28-11Zm-392 0q-11 11-28 11t-28-11l-86-86q-11-11-11-28t11-28q11-11 28-11t28 11l86 86q11 11 11 28t-11 28Zm196-84q-17 0-28.5-11.5T440-680v-120q0-17 11.5-28.5T480-840q17 0 28.5 11.5T520-800v120q0 17-11.5 28.5T480-640ZM160-200h640-640Z"/></svg></h2>
                #         </div>
                #         """, unsafe_allow_html=True)
   
                st.markdown(f"""
                        <div style="
                            border: 1px solid #ddd;
                            border-radius: 8px;
                            padding: 3px;
                            background-color: #ADD8E6;
                            text-align: center;
                            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                            margin: 3px;
                            height: 50px;  
                            overflow: hidden; 
                        ">
                        <h2 style="margin: 0; font-size: 24px; color: #333;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-nintendo-switch" viewBox="0 0 16 16">
  <path d="M9.34 8.005c0-4.38.01-7.972.023-7.982C9.373.01 10.036 0 10.831 0c1.153 0 1.51.01 1.743.05 1.73.298 3.045 1.6 3.373 3.326.046.242.053.809.053 4.61 0 4.06.005 4.537-.123 4.976-.022.076-.048.15-.08.242a4.14 4.14 0 0 1-3.426 2.767c-.317.033-2.889.046-2.978.013-.05-.02-.053-.752-.053-7.979m4.675.269a1.62 1.62 0 0 0-1.113-1.034 1.61 1.61 0 0 0-1.938 1.073 1.9 1.9 0 0 0-.014.935 1.63 1.63 0 0 0 1.952 1.107c.51-.136.908-.504 1.11-1.028.11-.285.113-.742.003-1.053M3.71 3.317c-.208.04-.526.199-.695.348-.348.301-.52.729-.494 1.232.013.262.03.332.136.544.155.321.39.556.712.715.222.11.278.123.567.133.261.01.354 0 .53-.06.719-.242 1.153-.94 1.03-1.656-.142-.852-.95-1.422-1.786-1.256"/>
  <path d="M3.425.053a4.14 4.14 0 0 0-3.28 3.015C0 3.628-.01 3.956.005 8.3c.01 3.99.014 4.082.08 4.39.368 1.66 1.548 2.844 3.224 3.235.22.05.497.06 2.29.07 1.856.012 2.048.009 2.097-.04.05-.05.053-.69.053-7.94 0-5.374-.01-7.906-.033-7.952-.033-.06-.09-.063-2.03-.06-1.578.004-2.052.014-2.26.05Zm3 14.665-1.35-.016c-1.242-.013-1.375-.02-1.623-.083a2.81 2.81 0 0 1-2.08-2.167c-.074-.335-.074-8.579-.004-8.907a2.85 2.85 0 0 1 1.716-2.05c.438-.176.64-.196 2.058-.2l1.282-.003v13.426Z"/></svg>                            
                            Discover Key Insights
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-nintendo-switch" viewBox="0 0 16 16">
  <path d="M9.34 8.005c0-4.38.01-7.972.023-7.982C9.373.01 10.036 0 10.831 0c1.153 0 1.51.01 1.743.05 1.73.298 3.045 1.6 3.373 3.326.046.242.053.809.053 4.61 0 4.06.005 4.537-.123 4.976-.022.076-.048.15-.08.242a4.14 4.14 0 0 1-3.426 2.767c-.317.033-2.889.046-2.978.013-.05-.02-.053-.752-.053-7.979m4.675.269a1.62 1.62 0 0 0-1.113-1.034 1.61 1.61 0 0 0-1.938 1.073 1.9 1.9 0 0 0-.014.935 1.63 1.63 0 0 0 1.952 1.107c.51-.136.908-.504 1.11-1.028.11-.285.113-.742.003-1.053M3.71 3.317c-.208.04-.526.199-.695.348-.348.301-.52.729-.494 1.232.013.262.03.332.136.544.155.321.39.556.712.715.222.11.278.123.567.133.261.01.354 0 .53-.06.719-.242 1.153-.94 1.03-1.656-.142-.852-.95-1.422-1.786-1.256"/>
  <path d="M3.425.053a4.14 4.14 0 0 0-3.28 3.015C0 3.628-.01 3.956.005 8.3c.01 3.99.014 4.082.08 4.39.368 1.66 1.548 2.844 3.224 3.235.22.05.497.06 2.29.07 1.856.012 2.048.009 2.097-.04.05-.05.053-.69.053-7.94 0-5.374-.01-7.906-.033-7.952-.033-.06-.09-.063-2.03-.06-1.578.004-2.052.014-2.26.05Zm3 14.665-1.35-.016c-1.242-.013-1.375-.02-1.623-.083a2.81 2.81 0 0 1-2.08-2.167c-.074-.335-.074-8.579-.004-8.907a2.85 2.85 0 0 1 1.716-2.05c.438-.176.64-.196 2.058-.2l1.282-.003v13.426Z"/></svg></h2>                        
                            </div>
                        """, unsafe_allow_html=True)

                Pa,Pb = st.columns((0.75,0.25))
                with Pa:
########3. Investment Plan Heat Maps########
                    
                    st.markdown(f"""
                        <div style="
                            border: 1px solid #ddd;
                            border-radius: 8px;
                            padding: 3px;
                            background-color: #f9f9f9;
                            text-align: center;
                            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                            margin: 3px;
                            height: 50px;  
                            overflow: hidden; 
                        ">
                        <h2 style="margin: 0; font-size: 20px; color: #333;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-clipboard2-pulse-fill" viewBox="0 0 16 16">
  <path d="M10 .5a.5.5 0 0 0-.5-.5h-3a.5.5 0 0 0-.5.5.5.5 0 0 1-.5.5.5.5 0 0 0-.5.5V2a.5.5 0 0 0 .5.5h5A.5.5 0 0 0 11 2v-.5a.5.5 0 0 0-.5-.5.5.5 0 0 1-.5-.5"/>
  <path d="M4.085 1H3.5A1.5 1.5 0 0 0 2 2.5v12A1.5 1.5 0 0 0 3.5 16h9a1.5 1.5 0 0 0 1.5-1.5v-12A1.5 1.5 0 0 0 12.5 1h-.585q.084.236.085.5V2a1.5 1.5 0 0 1-1.5 1.5h-5A1.5 1.5 0 0 1 4 2v-.5q.001-.264.085-.5M9.98 5.356 11.372 10h.128a.5.5 0 0 1 0 1H11a.5.5 0 0 1-.479-.356l-.94-3.135-1.092 5.096a.5.5 0 0 1-.968.039L6.383 8.85l-.936 1.873A.5.5 0 0 1 5 11h-.5a.5.5 0 0 1 0-1h.191l1.362-2.724a.5.5 0 0 1 .926.08l.94 3.135 1.092-5.096a.5.5 0 0 1 .968-.039Z"/></svg>
                                 Capacity Expansion Strategies <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-clipboard2-pulse-fill" viewBox="0 0 16 16">
  <path d="M10 .5a.5.5 0 0 0-.5-.5h-3a.5.5 0 0 0-.5.5.5.5 0 0 1-.5.5.5.5 0 0 0-.5.5V2a.5.5 0 0 0 .5.5h5A.5.5 0 0 0 11 2v-.5a.5.5 0 0 0-.5-.5.5.5 0 0 1-.5-.5"/>
  <path d="M4.085 1H3.5A1.5 1.5 0 0 0 2 2.5v12A1.5 1.5 0 0 0 3.5 16h9a1.5 1.5 0 0 0 1.5-1.5v-12A1.5 1.5 0 0 0 12.5 1h-.585q.084.236.085.5V2a1.5 1.5 0 0 1-1.5 1.5h-5A1.5 1.5 0 0 1 4 2v-.5q.001-.264.085-.5M9.98 5.356 11.372 10h.128a.5.5 0 0 1 0 1H11a.5.5 0 0 1-.479-.356l-.94-3.135-1.092 5.096a.5.5 0 0 1-.968.039L6.383 8.85l-.936 1.873A.5.5 0 0 1 5 11h-.5a.5.5 0 0 1 0-1h.191l1.362-2.724a.5.5 0 0 1 .926.08l.94 3.135 1.092-5.096a.5.5 0 0 1 .968-.039Z"/></svg></h2>
                        </div>
                        """, unsafe_allow_html=True)                
                    data = {
                        "t1": [2, 2, 2, 2, 2],
                        "t2": [2, 2, 2, 2, 2],
                        "t3": [1, 1, 2, 1, 2],
                        "t4": [2, 0, 0, 2, 1]
                        }

                # Create a DataFrame for each plan (Plan A, Plan B, Plan C, Plan D, Plan E)
                    plans = ['Plan A', 'Plan B', 'Plan C', 'Plan D', 'Plan E']
                    work_centers = ['WC1', 'WC2', 'WC3']

                    plan_data = {
                        'Plan A': [[2, 0, 0], [2, 0, 0], [1, 0, 1], [2, 0, 0]],
                        'Plan B': [[2, 0, 0], [2, 0, 0], [1, 0, 0], [0, 0, 1]],
                        'Plan C': [[2, 0, 0], [2, 0, 0], [2, 0, 0], [0, 0, 1]],
                        'Plan D': [[2, 0, 0], [2, 0, 0], [1, 0, 0], [2, 0, 0]],
                        'Plan E': [[2, 0, 0], [2, 0, 0], [2, 0, 0], [1, 0, 0]]
                        }
                    fig, axes = plt.subplots(1, 5, figsize=(25, 6), gridspec_kw={'wspace': 0.4})

                    for i, plan in enumerate(plans):
                        df_3 = pd.DataFrame(plan_data[plan], columns=work_centers, index=["t1", "t2", "t3", "t4"])
    
                        sns.heatmap(df_3, annot=True, cmap="YlGnBu", cbar=False, ax=axes[i], fmt="d", annot_kws={"size": 12, "weight": "bold"}, linewidths=0.5, linecolor='gray')
    
                        axes[i].set_title(plan, fontsize=16, fontweight='bold',pad=20)
                        axes[i].set_xlabel('Work Centers', fontsize=12,labelpad=10)
                        axes[i].set_ylabel('Time Periods', fontsize=12,labelpad=10)

                    plt.subplots_adjust(top=0.85, bottom=0.15)
                    plt.tight_layout()
                    st.pyplot(fig)

                
########4. Pie-Charts for the Investment Plan########
                # Given list of investment plans
                with Pb:    
                    # st.markdown(f"""
                    #     <div style="
                    #         border: 1px solid #ddd;
                    #         border-radius: 8px;
                    #         padding: 3px;
                    #         background-color: #f9f9f9;
                    #         text-align: center;
                    #         box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                    #         margin: 3px;
                    #         height: 50px;  
                    #         overflow: hidden; 
                    #     ">
                    #     <h2 style="margin: 0; font-size: 20px; color: #333;"> High Probability Selections </h2>
                    #     </div>
                    #     """, unsafe_allow_html=True)
                    st.markdown(""" <div
                        style="font-size: 16px; text-align: center;">
                        <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#5f6368">
                                <path d="M320-800q-17 0-28.5-11.5T280-840q0-17 11.5-28.5T320-880h215q20 0 29.5 12.5T574-840q0 15-10 27.5T534-800H320ZM200-640q-17 0-28.5-11.5T160-680q0-17 11.5-28.5T200-720h175q20 0 29.5 12.5T414-680q0 15-10 27.5T374-640H200ZM80-480q-17 0-28.5-11.5T40-520q0-17 11.5-28.5T80-560h135q20 0 29.5 12.5T254-520q0 15-10 27.5T214-480H80Zm504 80L480-504 280-304l104 104 200-200Zm-47-161 104 104 199-199-104-104-199 199Zm-84-28 216 216-229 229q-24 24-56 24t-56-24l-2-2-14 14q-6 6-13.5 9t-15.5 3H148q-14 0-19-12t5-22l92-92-2-2q-24-24-24-56t24-56l229-229Zm0 0 227-227q24-24 56-24t56 24l104 104q24 24 24 56t-24 56L669-373 453-589Z"/></svg> 
                                <b>High Probability Selections</b>
                        </div>
                        """, unsafe_allow_html = True)
                    investment_plans = df['Plan']

                    plan_counts = Counter(investment_plans)
                    labels = list(plan_counts.keys())
                    sizes = list(plan_counts.values())
                    colors = ["#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"]

                    fig, ax = plt.subplots(figsize=(8, 8))
                    wedges, texts, autotexts = ax.pie(
                                sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140,
                                wedgeprops={"edgecolor": "black", "linewidth": 1.2, "linestyle": "--"}
                                )

                    for text, autotext, wedge in zip(texts, autotexts, wedges):
                        text.set_fontsize(12)
                        text.set_fontweight("bold")
                        autotext.set_fontsize(12)
                        autotext.set_fontweight("bold")
                        autotext.set_color("black")

                        angle = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
                        x = wedge.r * 1.2 * np.cos(angle * (3.14159 / 180))
                        y = wedge.r * 1.2 * np.sin(angle * (3.14159 / 180))
                    # ax.set_title("Frequency of Investment Plans", fontsize=18, fontweight="bold")
                    st.pyplot(fig)
                st.markdown('---')

########5. Stacked Bar Graph for Production level########

                st.markdown(f"""
                        <div style="
                            border: 1px solid #ddd;
                            border-radius: 8px;
                            padding: 3px;
                            background-color: #f9f9f9;
                            text-align: center;
                            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                            margin: 3px;
                            height: 50px;  
                            overflow: hidden; 
                        ">
                        <h2 style="margin: 0; font-size: 20px; color: #333;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-reception-4" viewBox="0 0 16 16">
  <path d="M0 11.5a.5.5 0 0 1 .5-.5h2a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1-.5-.5zm4-3a.5.5 0 0 1 .5-.5h2a.5.5 0 0 1 .5.5v5a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1-.5-.5zm4-3a.5.5 0 0 1 .5-.5h2a.5.5 0 0 1 .5.5v8a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1-.5-.5zm4-3a.5.5 0 0 1 .5-.5h2a.5.5 0 0 1 .5.5v11a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1-.5-.5z"/></svg> 
                            Strategic Production Levels for Demand Variability <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-reception-4" viewBox="0 0 16 16">
  <path d="M0 11.5a.5.5 0 0 1 .5-.5h2a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1-.5-.5zm4-3a.5.5 0 0 1 .5-.5h2a.5.5 0 0 1 .5.5v5a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1-.5-.5zm4-3a.5.5 0 0 1 .5-.5h2a.5.5 0 0 1 .5.5v8a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1-.5-.5zm4-3a.5.5 0 0 1 .5-.5h2a.5.5 0 0 1 .5.5v11a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1-.5-.5z"/></svg></h2>
                        </div>
                        """, unsafe_allow_html=True)
#                 st.markdown(""" <div
#                         style="font-size: 20px; text-align: center;">
#                         <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-file-earmark-bar-graph" viewBox="0 0 16 16">
#   <path d="M10 13.5a.5.5 0 0 0 .5.5h1a.5.5 0 0 0 .5-.5v-6a.5.5 0 0 0-.5-.5h-1a.5.5 0 0 0-.5.5zm-2.5.5a.5.5 0 0 1-.5-.5v-4a.5.5 0 0 1 .5-.5h1a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-.5.5zm-3 0a.5.5 0 0 1-.5-.5v-2a.5.5 0 0 1 .5-.5h1a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-.5.5z"/>
#   <path d="M14 14V4.5L9.5 0H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2M9.5 3A1.5 1.5 0 0 0 11 4.5h2V14a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1h5.5z"/>
# </svg> Strategic Production Buffers for Demand Variability
#                         </div>
#                         """, unsafe_allow_html = True)
                columns_to_use = ["Scenario Name", "X (P1,1)", "X (P1,2)", "X (P1,3)", "X (P1,4)"]
                df_5 = df[columns_to_use]
                # st.dataframe(df_5)
                for col in columns_to_use[1:]:
                      df_5[col] = pd.to_numeric(df_5[col], errors="coerce")

                fig, ax = plt.subplots(figsize=(10, 6))
                df_5.set_index("Scenario Name").plot(kind="bar", stacked=True, ax=ax)
                plt.xlabel("Demand Scenarios",fontweight="bold")
                plt.ylabel("Production Levels (unit tonnes) with time",fontweight="bold")
                # plt.title("Stacked Production Levels across Demand Scenarios")
                plt.xticks([])
                plt.legend(title="Production Levels",title_fontsize=12,fontsize=10,loc='upper left',bbox_to_anchor=(1, 1),frameon=True,framealpha=1,edgecolor='black')
                st.pyplot(fig)
                st.markdown('---')



########2. Scatter Plot for Investment Plan########
                
                st.markdown(f"""
                        <div style="
                            border: 1px solid #ddd;
                            border-radius: 8px;
                            padding: 3px;
                            background-color: #ADD8E6;
                            text-align: center;
                            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                            margin: 3px;
                            height: 50px;  
                            overflow: hidden; 
                        ">
                        <h2 style="margin: 0; font-size: 24px; color: #333;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-sliders2-vertical" viewBox="0 0 16 16">
  <path fill-rule="evenodd" d="M0 10.5a.5.5 0 0 0 .5.5h4a.5.5 0 0 0 0-1H3V1.5a.5.5 0 0 0-1 0V10H.5a.5.5 0 0 0-.5.5M2.5 12a.5.5 0 0 0-.5.5v2a.5.5 0 0 0 1 0v-2a.5.5 0 0 0-.5-.5m3-6.5A.5.5 0 0 0 6 6h1.5v8.5a.5.5 0 0 0 1 0V6H10a.5.5 0 0 0 0-1H6a.5.5 0 0 0-.5.5M8 1a.5.5 0 0 0-.5.5v2a.5.5 0 0 0 1 0v-2A.5.5 0 0 0 8 1m3 9.5a.5.5 0 0 0 .5.5h4a.5.5 0 0 0 0-1H14V1.5a.5.5 0 0 0-1 0V10h-1.5a.5.5 0 0 0-.5.5m2.5 1.5a.5.5 0 0 0-.5.5v2a.5.5 0 0 0 1 0v-2a.5.5 0 0 0-.5-.5"/></svg> 
                           Analyze the Sensitivity
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-sliders2-vertical" viewBox="0 0 16 16">
  <path fill-rule="evenodd" d="M0 10.5a.5.5 0 0 0 .5.5h4a.5.5 0 0 0 0-1H3V1.5a.5.5 0 0 0-1 0V10H.5a.5.5 0 0 0-.5.5M2.5 12a.5.5 0 0 0-.5.5v2a.5.5 0 0 0 1 0v-2a.5.5 0 0 0-.5-.5m3-6.5A.5.5 0 0 0 6 6h1.5v8.5a.5.5 0 0 0 1 0V6H10a.5.5 0 0 0 0-1H6a.5.5 0 0 0-.5.5M8 1a.5.5 0 0 0-.5.5v2a.5.5 0 0 0 1 0v-2A.5.5 0 0 0 8 1m3 9.5a.5.5 0 0 0 .5.5h4a.5.5 0 0 0 0-1H14V1.5a.5.5 0 0 0-1 0V10h-1.5a.5.5 0 0 0-.5.5m2.5 1.5a.5.5 0 0 0-.5.5v2a.5.5 0 0 0 1 0v-2a.5.5 0 0 0-.5-.5"/></svg>                        
                            </div>
                        """, unsafe_allow_html=True)
                st.markdown(f"""
                        <div style="
                            border: 1px solid #ddd;
                            border-radius: 8px;
                            padding: 3px;
                            background-color: #f9f9f9;
                            text-align: center;
                            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                            margin: 3px;
                            height: 50px;  
                            overflow: hidden; 
                        ">
                        <h2 style="margin: 0; font-size: 20px; color: #333;"> Interactive Investment Planner: Map Costs to Your Demand Scenarios </h2>
                        </div>
                        """, unsafe_allow_html=True)
                S1,S2 = st.columns((0.25,0.75))
                plan_order = ["Plan A", "Plan B", "Plan C", "Plan D", "Plan E"]
                df['Plan'] = pd.Categorical(df['Plan'], categories=plan_order, ordered=True)

                with S1:
                    # scenario_options = df["Scenario Name"].unique().tolist()
                    theta_mapping = {
                                    "Min": {"Theta 1": 64, "Theta 2": 80, "Theta 3": 72, "Theta 4": 64},
                                    "Nominal": {"Theta 1": 80, "Theta 2": 100, "Theta 3": 90, "Theta 4": 80},
                                    "Max": {"Theta 1": 96, "Theta 2": 120, "Theta 3": 108, "Theta 4": 96}
                                    }
                    
                    theta_1_selection = st.multiselect("Select Product demand for 1st time period", options=["Min", "Nominal", "Max"], default=["Nominal"])
                    theta_2_selection = st.multiselect("Select Product demand for 2nd time period", options=["Min", "Nominal", "Max"], default=["Nominal"])
                    theta_3_selection = st.multiselect("Select Product demand for 3rd time period", options=["Min", "Nominal", "Max"], default=["Nominal"])
                    theta_4_selection = st.multiselect("Select Product demand for 4th time period", options=["Min", "Nominal", "Max"], default=["Nominal"])

                    theta_1_values = [theta_mapping[selection]["Theta 1"] for selection in theta_1_selection]
                    theta_2_values = [theta_mapping[selection]["Theta 2"] for selection in theta_2_selection]
                    theta_3_values = [theta_mapping[selection]["Theta 3"] for selection in theta_3_selection]
                    theta_4_values = [theta_mapping[selection]["Theta 4"] for selection in theta_4_selection]

                    filtered_df = df[(df["Theta 1"].isin(theta_1_values)) & (df["Theta 2"].isin(theta_2_values)) & (df["Theta 3"].isin(theta_3_values)) & (df["Theta 4"].isin(theta_4_values))]
                    # st.write("Filtered Data:", filtered_df)
                
                    scenarios = filtered_df['Scenario Name']
                    updated_obj = filtered_df['Updated Obj']
                    invtplan = filtered_df['Plan']

                if not filtered_df.empty:
                    with S2:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        scatter = sns.scatterplot(data=filtered_df,
                            x=scenarios,
                            y=updated_obj,
                            hue=invtplan,  # Color by Plan (which is now sorted)
                            hue_order=plan_order,  # Ensure correct order in legend
                            palette="viridis",
                            s=100,  # Dot size
                            ax=ax)

                # Customize the plot
                        ax.set_title("Optimal Investment Plans across Demand Scenarios")
                        ax.set_xlabel("Demand Scenarios",fontweight='bold')
                        ax.set_ylabel("Cost (×10³, dollars)",fontweight='bold')
                        ax.legend(title="Investment Plan")
                        ax.grid(True, linestyle='--', alpha=0.7)
                        ax.set_xticks([])         
                        st.pyplot(fig)

#######End section######
                st.markdown('---')
                st.markdown(f"""
                        <div style="
                            border: 1px solid #ddd;
                            border-radius: 8px;
                            padding: 3px;
                            background-color: #ADD8E6;
                            text-align: center;
                            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                            margin: 3px;
                            height: 50px;  
                            overflow: hidden; 
                        ">
                        <h2 style="margin: 0; font-size: 24px; color: #333;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-credit-card-2-front-fill" viewBox="0 0 16 16">
                            <path d="M0 4a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2zm2.5 1a.5.5 0 0 0-.5.5v1a.5.5 0 0 0 .5.5h2a.5.5 0 0 0 .5-.5v-1a.5.5 0 0 0-.5-.5zm0 3a.5.5 0 0 0 0 1h5a.5.5 0 0 0 0-1zm0 2a.5.5 0 0 0 0 1h1a.5.5 0 0 0 0-1zm3 0a.5.5 0 0 0 0 1h1a.5.5 0 0 0 0-1zm3 0a.5.5 0 0 0 0 1h1a.5.5 0 0 0 0-1zm3 0a.5.5 0 0 0 0 1h1a.5.5 0 0 0 0-1z"/>
                            </svg>
                            Additional Resources
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-credit-card-2-front-fill" viewBox="0 0 16 16">
                        <path d="M0 4a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2zm2.5 1a.5.5 0 0 0-.5.5v1a.5.5 0 0 0 .5.5h2a.5.5 0 0 0 .5-.5v-1a.5.5 0 0 0-.5-.5zm0 3a.5.5 0 0 0 0 1h5a.5.5 0 0 0 0-1zm0 2a.5.5 0 0 0 0 1h1a.5.5 0 0 0 0-1zm3 0a.5.5 0 0 0 0 1h1a.5.5 0 0 0 0-1zm3 0a.5.5 0 0 0 0 1h1a.5.5 0 0 0 0-1zm3 0a.5.5 0 0 0 0 1h1a.5.5 0 0 0 0-1z"/></svg>                        
                            </div>
                        """, unsafe_allow_html=True)
                f1, f2 = st.columns(2)
                with f2:
                    st.markdown("""
                    <div style="text-align: right; font-family: 'Courier New', monospace; font-size: 18px; color: #000000;">
                    Any Questions <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#5f6368"><path d="M470-200h-10q-142 0-241-99t-99-241q0-142 99-241t241-99q71 0 132.5 26.5t108 73q46.5 46.5 73 108T800-540q0 134-75.5 249T534-111q-10 5-20 5.5t-18-4.5q-8-5-14-13t-7-19l-5-58Zm90-26q71-60 115.5-140.5T720-540q0-109-75.5-184.5T460-800q-109 0-184.5 75.5T200-540q0 109 75.5 184.5T460-280h100v54Zm-101-95q17 0 29-12t12-29q0-17-12-29t-29-12q-17 0-29 12t-12 29q0 17 12 29t29 12Zm-87-304q11 5 22 .5t18-14.5q9-12 21-18.5t27-6.5q24 0 39 13.5t15 34.5q0 13-7.5 26T480-558q-25 22-37 41.5T431-477q0 12 8.5 20.5T460-448q12 0 20-9t12-21q5-17 18-31t24-25q21-21 31.5-42t10.5-42q0-46-31.5-74T460-720q-32 0-59 15.5T357-662q-6 11-1.5 21.5T372-625Zm88 112Z"/></svg>
                    </div>""", unsafe_allow_html = True)
                    st.markdown("""
                    <div style="text-align: right; font-family: 'Courier New', monospace; font-size: 16px; color: #000000;">
                    Reach out to sshikha@andrew.cmu.edu<br>
                    <span style="display: inline-flex; align-items: center;">
                    For more info 
                    <svg xmlns="http://www.w3.org/2000/svg" height="16px" viewBox="0 -960 960 960" width="16px" fill="#5f6368" style="margin-left: 4px;">
                    <path d="M480-280q17 0 28.5-11.5T520-320v-160q0-17-11.5-28.5T480-520q-17 0-28.5 11.5T440-480v160q0 17 11.5 28.5T480-280Zm0-320q17 0 28.5-11.5T520-640q0-17-11.5-28.5T480-680q-17 0-28.5 11.5T440-640q0 17 11.5 28.5T480-600Zm0 520q-83 0-156-31.5T197-197q-54-54-85.5-127T80-480q0-83 31.5-156T197-763q54-54 127-85.5T480-880q83 0 156 31.5T763-763q54 54 85.5 127T880-480q0 83-31.5 156T763-197q-54 54-127 85.5T480-80Zm0-80q134 0 227-93t93-227q0-134-93-227t-227-93q-134 0-227 93t-93 227q0 134 93 227t227 93Zm0-320Z"/>
                    </svg>
                    </span>
                    </div>
                    """, unsafe_allow_html=True)

                with f1:

                    st.markdown("""
                    <div style="font-family: 'Courier New', monospace; font-size: 20px; color: #000000;">
                    Download the Result files <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#5f6368"><path d="m680-272-36-36q-11-11-28-11t-28 11q-11 11-11 28t11 28l104 104q12 12 28 12t28-12l104-104q11-11 11-28t-11-28q-11-11-28-11t-28 11l-36 36v-127q0-17-11.5-28.5T720-439q-17 0-28.5 11.5T680-399v127ZM600-80h240q17 0 28.5 11.5T880-40q0 17-11.5 28.5T840 0H600q-17 0-28.5-11.5T560-40q0-17 11.5-28.5T600-80Zm-360-80q-33 0-56.5-23.5T160-240v-560q0-33 23.5-56.5T240-880h247q16 0 30.5 6t25.5 17l194 194q11 11 17 25.5t6 30.5v48q0 17-11.5 28.5T720-519q-17 0-28.5-11.5T680-559v-41H540q-25 0-42.5-17.5T480-660v-140H240v560h200q17 0 28.5 11.5T480-200q0 17-11.5 28.5T440-160H240Zm0-80v-560 560Z"/></svg>
                    </div>""", unsafe_allow_html = True)

                    # Convert DataFrame to CSV
                    results_updated = df.to_csv(index=False)

                    # Create a download button
                    st.download_button(
                        label="Download Result file",
                        data=results_updated,
                        file_name='submission_buy.csv',
                        mime='text/csv'
                    )

            ###---Hide Streamlit Style
                hide_st_style = """
                                <style>
                                #MainMenu {visibility: hidden}
                                footer {visibility: hidden}
                                header {visibility: hidden}
                                </style>
                                """
                st.write(hide_st_style, unsafe_allow_html=True)

                st.markdown('---')
                st.markdown('---')



                

