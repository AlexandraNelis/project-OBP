import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
import plotly.express as px
import io
import altair as alt
import time
import numpy as np
import matplotlib.pyplot as plt
import base64
import gurobipy as gp
from gurobipy import Model, GRB, quicksum
from itertools import combinations
import seaborn as sns


def solve_gurobi_scheduling_problem(df, machine_columns):
    """
    Given a DataFrame `df` with columns:
      - 'TaskID' (unique identifier)
      - 'ReleaseDate'
      - 'DueDate'
      - 'Weight'
      - columns in `machine_columns` for processing times on each machine

    Returns:
      A dictionary containing:
        - 'status': solver status
        - 'objective': objective function value (sum of weighted tardiness)
        - 'schedule': list of dicts describing the schedule, with fields:
            'task_id', 'finish_time', 'tardiness', 'machine_times'
    """
    tasks = df.to_dict('records')
    model = gp.Model("WeightedTardinessScheduling")

    num_tasks = len(tasks)
    jobs = list(range(num_tasks)) # e.g. [0,1,2] if there are 3 jobs
    machines = list(range(len(machine_columns)))  # e.g. [0,1,2] if there are 3 machine columns

    # Build a 2D list for processing times: times[task_idx][machine_idx]
    times = [[t[col] for col in machine_columns] for t in tasks]

    # Define horizon as sum of all processing times (a crude upper bound)
    horizon = sum(sum(t_row) for t_row in times)


    x = model.addVars(jobs, machines, vtype=GRB.INTEGER, name="x")  # Start times for a job at a given machine

    #z = model.addVars(jobs,jobs, machines, vtype=GRB.BINARY, name="z")  # Binary variables (1 if job j is processed before job i at machine k)
    #z1 = model.addVars(machines, machines, jobs, vtype=GRB.BINARY, name="z")  # Binary variables (1 if job j is processed at machine i before machine j)

    z = model.addVars(combinations(jobs, 2), machines, vtype=GRB.BINARY, name="z")  # Binary variables (1 if job j is processed before job i at machine k)
    z1 = model.addVars(combinations(machines, 2), jobs, vtype=GRB.BINARY, name="z")  # Binary variables (1 if job j is processed at machine i before machine j)
    T = model.addVars(jobs, lb = 0, vtype=GRB.INTEGER, name="T")  # Tardiness for each job

    model.setObjective(quicksum(tasks[j]['Weight'] * T[j] for j in jobs), GRB.MINIMIZE)
    
    # Constraints
    for i in jobs:
        for k in machines:
            # Start times must be later than the release date
            model.addConstr(x[i, k] >= tasks[i]['ReleaseDate'], name=f"start_time_nonneg_{i}_{k}")
            # Tardiness
            model.addConstr(T[i] >= x[i, k] + times[i][k] - tasks[i]['DueDate'], name=f"tardiness_{i}_{k}")
    
    for k in machines:
        for i in jobs:
            for j in jobs:
                if i < j:  # Avoid duplicate pairs
                    # Disjunctive constraints(the same machine can not process simultaneously multiple machines)
                    model.addConstr(
                        x[i, k] + times[i][k] <= x[j, k] + horizon * (1 - z[i, j, k]),
                        name=f"job_{i}_before_{j}_on_machine_{k}"
                    )
                    model.addConstr(
                        x[j, k] + times[j][k] <= x[i, k] + horizon * z[i, j, k],
                        name=f"job_{i}_before_{j}_on_machine_{k}"
                    )
    
    for k in jobs:
        for i in machines:
            for j in machines:
                if i < j:  # Avoid duplicate pairs
                    # Disjunctive constraints (the same job can not be processed simultaneously by multiple machines)
                    model.addConstr(
                        x[k, i] + times[k][i] <= x[k, j] + horizon * (1 - z1[i, j, k]),
                        name=f"job_{i}_before_{j}_on_machine_{k}"
                    )
                    model.addConstr(
                        x[k, j] + times[k][j] <= x[k, i] + horizon * z1[i, j, k],
                        name=f"machine_{i}_before_{j}_on_job_{k}"
                    )

    # Solve
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    print(end_time - start_time)

    status = model.status
    results = {
        'status': status,
        'objective': None,
        'schedule': []
    }

    if status == GRB.OPTIMAL or status == GRB.SUBOPTIMAL:
        results['objective'] = model.ObjVal

        # Build output schedule info
        for t_idx in range(num_tasks):
            t_id = tasks[t_idx]['TaskID']
            tardiness = T[t_idx].X

            machine_times = []
            for m_idx in machines:
                start_time = x[t_idx, m_idx].X
                end_time = x[t_idx, m_idx].X + times[t_idx][m_idx]
                machine_times.append((m_idx, start_time, end_time))

            results['schedule'].append({
                'task_id': t_id,
                'tardiness': tardiness,
                'machine_times': machine_times
            })

    return results


def create_model_variables(model, tasks, machines, machine_columns, horizon):
    """Create and return all model variables."""
    times = [[t[col] for col in machine_columns] for t in tasks]
    
    variables = {
        'start': {},
        'intervals': {},
        'task_end': {},
    }
    
    for t_idx in range(len(tasks)):
        for m_idx in machines:
            duration = times[t_idx][m_idx]
            
            start_var = model.NewIntVar(0, horizon, f'start_{t_idx}_m{m_idx}')
            interval_var = model.NewIntervalVar(
                start_var, duration, start_var + duration, f'interval_{t_idx}_m{m_idx}'
            )
            
            variables['start'][(t_idx, m_idx)] = start_var
            variables['intervals'][(t_idx, m_idx)] = interval_var
        
        variables['task_end'][t_idx] = model.NewIntVar(0, horizon, f'end_time_task{t_idx}')
    
    return variables, times

def add_scheduling_constraints(model, tasks, machines, variables, times):
    """Add all scheduling constraints to the model."""
    # No overlap on machines
    for m_idx in machines:
        machine_intervals = [variables['intervals'][(t_idx, m_idx)] 
                           for t_idx in range(len(tasks))]
        model.AddNoOverlap(machine_intervals)
    
    # No simultaneous processing
    for t_idx in range(len(tasks)):
        task_intervals = [variables['intervals'][(t_idx, m_idx)] 
                         for m_idx in machines]
        model.AddNoOverlap(task_intervals)
    
    # Release date constraints
    for t_idx in range(len(tasks)):
        for m_idx in machines:
            model.Add(variables['start'][(t_idx, m_idx)] >= 
                     tasks[t_idx]['ReleaseDate'])
    
    # End time constraints
    for t_idx in range(len(tasks)):
        end_times_task = []
        for m_idx in machines:
            end_time = variables['start'][(t_idx, m_idx)] + times[t_idx][m_idx]
            end_times_task.append(end_time)
        model.AddMaxEquality(variables['task_end'][t_idx], end_times_task)


def create_objective_variables(model, tasks, variables, horizon):
    """Create and return tardiness variables for the objective function."""

    weighted_tardiness = []
    
    for t_idx, task in enumerate(tasks):
        due_date = task['DueDate']
        weight = task['Weight']
        
        tardiness_var = model.NewIntVar(0, horizon, f'lateness_task{t_idx}')
        model.Add(tardiness_var >= variables['task_end'][t_idx] - due_date)
        model.Add(tardiness_var >= 0)
        
        weighted_tardiness.append(tardiness_var * weight)
    
    return weighted_tardiness


def extract_solution(solver, tasks, machines, variables, times):
    """Extract the solution from the solver."""
    schedule = []
    
    for t_idx, task in enumerate(tasks):
        task_end_time = solver.Value(variables['task_end'][t_idx])
        tardiness = max(0, task_end_time - task['DueDate'])
        machine_times = [
        # Increment machine index by 1 for display purposes
        (m_idx + 1,
         solver.Value(variables['start'][(t_idx, m_idx)]),  # Start time from the solver
         solver.Value(variables['start'][(t_idx, m_idx)]) + times[t_idx][m_idx])  # End time calculated as start + processing time
        for m_idx in machines
        ]
        
        schedule.append({
            'task_id': task['TaskID'],
            'finish_time': task_end_time,
            'tardiness': tardiness,
            'machine_times': machine_times,
            'weight': task['Weight']
        })
    
    return schedule


def solve_scheduling_problem(df, machine_columns):
    """
    Optimized version of the scheduling problem Google CP-SAT solver.
    """
    tasks = df.to_dict('records')
    machines = list(range(len(machine_columns)))
    
    # Calculate horizon
    horizon = sum(
    t['DueDate'] - t['ReleaseDate'] + sum(t[col] for col in machine_columns)
    for t in tasks)

    # Initialize model
    model = cp_model.CpModel()
    
    # Create variables
    variables, times = create_model_variables(model, tasks, machines, 
                                           machine_columns, horizon)
    
    # Add constraints
    add_scheduling_constraints(model, tasks, machines, variables, times)
    
    # Create objective function
    tardiness_vars = create_objective_variables(model, tasks, variables, horizon)
    model.Minimize(sum(tardiness_vars))
    
    # Solve
    solver = cp_model.CpSolver()
    
    # Add time limit and other parameters to improve performance
    solver.parameters.max_time_in_seconds = 60.0  # 5 minute timeout
    solver.parameters.num_search_workers = 8  # Use multiple cores
    solver.parameters.log_search_progress = True  # Enable logging for debugging
    
    start_time = time.time()
    status = solver.Solve(model)
    solve_time = time.time() - start_time
    
    results = {'status': status, 'objective': None, 'schedule': [], 'solve_time': solve_time}
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        results['objective'] = solver.ObjectiveValue()
        results['schedule'] = extract_solution(solver, tasks, machines, variables, times)
    
    return results



def generate_test_case(num_jobs, num_machines):
    """
    Generate a synthetic test case with the specified number of jobs and machines.
    """
    ratio = 1
    min_time_service = 1
    max_time_service = 10
    average_service =(min_time_service+max_time_service)/2

    end_release =num_machines*ratio*average_service
    begin_due = end_release + max_time_service*ratio*num_machines
    end_due = begin_due + max_time_service*ratio*num_machines
    average_service =(min_time_service+max_time_service)/2
    data = {
        "TaskID": list(range(1, num_jobs + 1)),
        "ReleaseDate": np.random.randint(0,end_release, size=num_jobs),  # 0,11
        "DueDate": np.random.randint(begin_due,end_due, size=num_jobs), #55,110
        "Weight": np.random.randint(1, 3, size=num_jobs),       # Random weights
    }
    for i in range(1, num_machines + 1):
        data[f"Machine {i}"] = np.random.randint(min_time_service, max_time_service, size=num_jobs)  # Random processing times

    return pd.DataFrame(data)

def evaluate_solver(max_jobs,max_machines,time_limit,batch):
    """
    Run the solver with increasing job and machine sizes, and record results.
    """
    results = []
    largest_set_of_jobs = {}
      
    for num_machines in range(2, max_machines + 1): # Increment machines by 1
        best_job = 0
        largest_set_of_jobs[num_machines]=best_job        
        for num_jobs in range(2, max_jobs + 1, 2):# Increment jobs by 10 
            st.write(f"Testing with {num_jobs} jobs and {num_machines} machines...")
            df = generate_test_case(num_jobs, num_machines)
            solving_set=[]
            Time_capped = False

            for i in range(batch):#uneven number so the ratio will always tip to one side
                st.write(f"trial {i+1}")
                machine_columns = [f"Machine {i}" for i in range(1, num_machines + 1)]
                start_time = time.time()
                result = solve_scheduling_problem(df, machine_columns)
                end_time = time.time()
                solving_time =  end_time - start_time
                if solving_time >= time_limit:
                    solving_set.append((result,False,solving_time))
                    st.write(f"Not in time")
                else:
                    solving_set.append((result,True,solving_time))
                    st.write(f"Within time")

            false_count = sum(1 for _, is_false, _ in solving_set if not is_false)
            solving_set.sort(key=lambda x: not x[1])
            if false_count > batch/2:#more than half of the tries failed at the instance
                Time_capped = True
                solving_set.sort(key=lambda x: x[1])
            # Record performance
            print(Time_capped)
            results.append({
                "NumJobs": num_jobs,
                "NumMachines": num_machines,
                "SolverStatus": solving_set[0][0]["status"],
                "ObjectiveValue": solving_set[0][0]["objective"],
                "SolveTime":np.mean([time_value for _, condition, time_value in solving_set if condition!=Time_capped]),
                "TimeCapped": Time_capped,
                "Horizon": max(df["DueDate"])
            })
            if  results[-1]['SolveTime']<=60:
                best_job = num_jobs
            largest_set_of_jobs[num_machines] = best_job
            if len(results)>1:
                if results[-2]['TimeCapped'] and  results[-1]['TimeCapped']:
                    print("no improvement")
                    st.write(f"No improvement")
                    break          
    return pd.DataFrame(results),pd.DataFrame(list(largest_set_of_jobs.items()), columns=["Number of machines", "Number of jobs"])

def run_batch(df, machine_columns,solver,solving_set,time_limit):
    start_time = time.time()
    if solver:#or tools
        result = solve_scheduling_problem(df, machine_columns)
    else:# gurobi
        result = solve_gurobi_scheduling_problem(df,machine_columns)
    end_time = time.time()
    solving_time =  end_time - start_time
    if solving_time >= time_limit:
        solving_set.append((result,False,solving_time))
        st.write(f"Not in time")
    else:
        solving_set.append((result,True,solving_time))
        st.write(f"Within time")
    return solving_set


def compare_solvers(test_jobs,test_machines):
    or_results = []
    gur_results=[]
    time_limit = 60 
    or_largest_set_of_jobs = {}
    gur_largest_set_of_jobs = {}
    batch = 5
    for num_machines in test_machines:
        or_best_job = 0
        gur_best_job = 0
        or_largest_set_of_jobs[num_machines]=or_best_job    
        gur_largest_set_of_jobs[num_machines]=gur_best_job     
        for num_jobs in test_jobs:
            st.write(f"Testing with {num_jobs} jobs and {num_machines} machines...")
            df = generate_test_case(num_jobs, num_machines)
            or_solving_set=[]
            gur_solving_set=[]
            or_Time_capped = False
            gur_Time_capped = False
            for i in range(batch):#uneven number so the ratio will always tip to one side
                st.write(f"trial {i+1}")
                machine_columns = [f"Machine {i}" for i in range(1, num_machines + 1)]
                or_solving_set = run_batch(df, machine_columns,True,or_solving_set,time_limit)
                gur_solving_set = run_batch(df, machine_columns,True,gur_solving_set,time_limit)

            or_false_count = sum(1 for _, is_false, _ in or_solving_set if not is_false)
            gur_false_count =  sum(1 for _, is_false, _ in gur_solving_set if not is_false)
            or_solving_set.sort(key=lambda x: not x[1])
            gur_solving_set.sort(key=lambda x: not x[1])
            if or_false_count > batch/2:#more than half of the tries failed at the instance
                or_Time_capped = True
                or_solving_set.sort(key=lambda x: x[1])
            if gur_false_count > batch/2:#more than half of the tries failed at the instance
                gur_Time_capped = True
                gur_solving_set.sort(key=lambda x: x[1])
            
            # Record performance
            or_results.append({
                "NumJobs": num_jobs,
                "NumMachines": num_machines,
                "SolverStatus": or_solving_set[0][0]["status"],
                "ObjectiveValue": or_solving_set[0][0]["objective"],
                "SolveTime":np.mean([time_value for _, condition, time_value in or_solving_set if condition!=or_Time_capped]),
                "TimeCapped": or_Time_capped,
                "Horizon": max(df["DueDate"])
            })
            gur_results.append({
                "NumJobs": num_jobs,
                "NumMachines": num_machines,
                "SolverStatus": gur_solving_set[0][0]["status"],
                "ObjectiveValue": gur_solving_set[0][0]["objective"],
                "SolveTime":np.mean([time_value for _, condition, time_value in gur_solving_set if condition!=gur_Time_capped]),
                "TimeCapped": gur_Time_capped,
                "Horizon": max(df["DueDate"])
            })
            if  or_results[-1]['SolveTime']<=time_limit:
                or_best_job = num_jobs
            if  gur_results[-1]['SolveTime']<=time_limit:
                gur_best_job = num_jobs
            
            or_largest_set_of_jobs[num_machines] = or_best_job
            gur_largest_set_of_jobs[num_machines] = or_best_job
    return pd.DataFrame(or_results),pd.DataFrame(gur_results),pd.DataFrame(list(or_largest_set_of_jobs.items()), columns=["Number of machines", "Number of jobs"]),pd.DataFrame(list(gur_largest_set_of_jobs.items()), columns=["Number of machines", "Number of jobs"])



def create_download_link(val, filename,type):
    if type=="png":
        b64 = base64.b64encode(val).decode()  # Base64 encode the PNG file
        return f'<a href="data:image/png;base64,{b64}" download="{filename}.{type}">Download image</a>'
    else:
        b64 = base64.b64encode(val)
        return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.{type}">Download {filename}</a>'



option = st.selectbox(
    "Choose an option:",  # Label for the dropdown
    ["Extended solver OR Tools", "Compare Gurobi and OR tools"]  # List of options
)
if option == "Extended solver OR Tools":
    st.write("You chose Extended solver OR Tools! 🎉")
     # User input fields (before button)
    num_jobs = st.number_input("Maximum number of jobs", min_value=1, step=1)
    num_machines = st.number_input("Maximum number of machines", min_value=1, step=1)
    time_limit = st.number_input("Search time limit (seconds)", min_value=1, step=1)
    batch = st.number_input("Batch size of trial runs per instance", min_value=1, step=1)

    # Run solver only when button is clicked
    if st.button("Run Solver Performance Tests"):
        with st.spinner("Running tests..."):
            performance_results, largest_set = evaluate_solver(
                int(num_jobs), int(num_machines), int(time_limit), int(batch)
            )
            st.markdown("### Performance Results")
            st.dataframe(performance_results)
            st.markdown("### Largest number of jobs per machine")
            st.dataframe(largest_set)
            st.bar_chart(largest_set, x = 'Number of machines', y = 'Number of jobs', x_label= "number of machines", y_label= "number of jobs")
            graph_data =largest_set.set_index('Number of machines')
            fig, ax = plt.subplots()
            graph_data.plot(kind="bar", ax=ax, legend=False)
            ax.set_title("Maximum number of jobs solvable for number of machines")
            ax.set_xlabel("Number of machines")
            ax.set_ylabel("Number of jobs")

            # Save the chart to a buffer for downloading
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png")
            buffer.seek(0)
            png_data= buffer.getvalue()

            
            # Optionally save results to a CSV
            csv_data = performance_results.to_csv(index=False).encode("utf-8")
            largest_set_data = largest_set.to_csv(index=False).encode("utf-8")
            performance_url = create_download_link(csv_data, 'Performance results',"csv")
            largest_set_url =create_download_link(largest_set_data,"Largest jobset","csv")
            graph_url = create_download_link(png_data,"Bar chart","png")
            st.markdown(performance_url, unsafe_allow_html=True)
            st.markdown(largest_set_url, unsafe_allow_html=True)
            st.markdown(graph_url, unsafe_allow_html=True)

elif option == "Compare Gurobi and OR tools":
    st.write("You chose Compare Gurobi and OR tools! 🚀")
    if st.button("Run Comparison solver Tests"):
        with st.spinner("Running tests..."):
            test_jobs = [2,3]
            test_machines = [2,5]
            or_performance_results,gur_performance_results,or_largest_set,gur_largest_set = compare_solvers(test_jobs,test_machines)
            st.markdown("### Performance Results")
            st.dataframe(or_performance_results)
            st.dataframe(gur_performance_results)
            print(or_largest_set)
            print(gur_largest_set)
            st.markdown("### Largest number of jobs per machine")
            

            combined_data = pd.merge(or_largest_set, gur_largest_set, on="Number of machines")

            # Reshape the DataFrame to a long format
            combined_data_melted = combined_data.melt(id_vars=["Number of machines"], var_name="Solver", value_name="Number of jobs")

            # Display the DataFrame
            st.write(combined_data_melted)  # Optional: View the transformed DataFrame

            # Create a grouped bar chart using Altair
            chart = alt.Chart(combined_data_melted).mark_bar().encode(
                x=alt.X("Number of machines:O", title="Number of Machines"),  # Convert to categorical
                y=alt.Y("Number of jobs:Q", title="Number of Jobs"),
                color="Solver:N",  # Color by solver
                xOffset="Solver:N"  # Offset bars to make them appear side by side
            ).properties(
                width=100  # Adjust width for clarity
            )

            # Display the Altair chart in Streamlit
            st.altair_chart(chart, use_container_width=True)
            fig, ax = plt.subplots()
            # Plot with "hue" by setting `x`, `y`, and `hue`
            sns.barplot(
                data=combined_data_melted, 
                x="Number of machines", 
                y="Number of jobs", 
                hue="Solver",  # Differentiate bars by "variable" column
                ax=ax
                )   

            ax.set_title("Grouped Bar Chart")
            ax.set_xlabel("Number of machines")
            ax.set_ylabel("Number of jobs")
            ax.set_label("Comparison between Gurobi and OR tools per machine")
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

            plt.tight_layout()  # Ensures layout does not cut off elements

            # Save the chart to a buffer for downloading
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png")
            buffer.seek(0)
            png_data= buffer.getvalue()

            
            # Optionally save results to a CSV
            or_csv_data = or_performance_results.to_csv(index=False).encode("utf-8")
            gur_csv_data = gur_performance_results.to_csv(index=False).encode("utf-8")

            largest_set_data = combined_data.to_csv(index=False).encode("utf-8")
            or_performance_url = create_download_link(or_csv_data, 'Performance results OR tools',"csv")
            gur_performance_url = create_download_link(gur_csv_data, 'Performance results Gurobi',"csv")

            largest_set_url =create_download_link(largest_set_data,"Largest jobset","csv")
            graph_url = create_download_link(png_data,"Bar chart","png")
            st.markdown(or_performance_url, unsafe_allow_html=True)
            st.markdown(gur_performance_url, unsafe_allow_html=True)
            st.markdown(largest_set_url, unsafe_allow_html=True)
            st.markdown(graph_url, unsafe_allow_html=True)


# Run and display the evaluation results
