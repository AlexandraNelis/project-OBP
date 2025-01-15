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


def solve_scheduling_problem(df, machine_columns):
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
    model = cp_model.CpModel()

    num_tasks = len(tasks)
    machines = list(range(len(machine_columns)))  # e.g. [0,1,2] if there are 3 machine columns

    # Build a 2D list for processing times: times[task_idx][machine_idx]
    times = []
    for t in tasks:
        row_times = [t[col] for col in machine_columns]
        times.append(row_times)

    # Define horizon as sum of all processing times (a crude upper bound)
    horizon = sum(sum(t_row) for t_row in times)

    # Create variables: start, end, interval
    start_task_per_machine_vars = {}
    end_task_per_machine_vars = {}
    end_task_vars = {}
    intervals = {}

    for t_idx in range(num_tasks):
        for m_idx in machines:
            duration = times[t_idx][m_idx]
            
            start_task_per_machine_var = model.NewIntVar(0, horizon, f'start_{t_idx}_m{m_idx}')
            end_task_per_machine_var = model.NewIntVar(0, horizon, f'end_{t_idx}_m{m_idx}')
            interval_var = model.NewIntervalVar(start_task_per_machine_var, duration, end_task_per_machine_var,
                                                f'interval_{t_idx}_m{m_idx}')

            start_task_per_machine_vars[(t_idx, m_idx)] = start_task_per_machine_var
            end_task_per_machine_vars[(t_idx, m_idx)] = end_task_per_machine_var
            intervals[(t_idx, m_idx)] = interval_var

        end_task_var = model.NewIntVar(0, horizon, f'end_time_task{t_idx}')
        end_task_vars[(t_idx)] = end_task_var
        

    # Constraint 1: Ensure no 2 tasks overlap at the same machine
    for m_idx in machines:
        machine_intervals = [intervals[(t_idx, m_idx)] for t_idx in range(num_tasks)]
        model.AddNoOverlap(machine_intervals)


    # Constraint 2: Ensure the same job is not processed on multiple machines simultaneously
    for t_idx in range(num_tasks):
        task_intervals = [intervals[(t_idx, m_idx)] for m_idx in machines]
        model.AddNoOverlap(task_intervals)


    # Constraint 3: The start time of a task, at each machine, should be later 
    # than its release date
    for t_idx in range(num_tasks):
        for m_idx in machines:
            model.Add(start_task_per_machine_vars[(t_idx, m_idx)] >= tasks[t_idx]['ReleaseDate'])

    # Constraint 4: End time for each task is its maximum end time for all the machines
    for t_idx in range(num_tasks):
        end_times_task = [end_task_per_machine_vars[(t_idx, m_idx)] for m_idx in machines]
        model.AddMaxEquality(end_task_vars[(t_idx)], end_times_task)


    # Tardiness variables
    tardiness_vars = []
    for t_idx in range(num_tasks):
        due_date = tasks[t_idx]['DueDate']
        weight = tasks[t_idx]['Weight']

        # lateness = max(0, finish - due_date)
        # create an IntVar for lateness
        lateness_var = model.NewIntVar(0, horizon, f'lateness_task{t_idx}')
        model.Add(lateness_var >= end_task_vars[(t_idx)] - due_date)
        model.Add(lateness_var >= 0)

        # Weighted tardiness = weight * lateness
        weighted_tardiness_var = model.NewIntVar(0, weight * horizon, f'wtardiness_task{t_idx}')
        model.Add(weighted_tardiness_var == lateness_var * weight)

        tardiness_vars.append(weighted_tardiness_var)

    # Objective: minimize sum of weighted tardiness
    model.Minimize(sum(tardiness_vars))

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    start_time = time.time()
    status = solver.Solve(model)
    end_time = time.time()

    print(end_time - start_time)

    results = {
        'status': status,
        'objective': None,
        'schedule': []
    }

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        results['objective'] = solver.ObjectiveValue()

        # Build output schedule info
        for t_idx in range(num_tasks):
            t_id = tasks[t_idx]['TaskID']
            task_end_time = solver.Value(end_task_vars[(t_idx)])
            tardiness = max(0, task_end_time - tasks[t_idx]['DueDate'])

            machine_times = []
            for m_idx in machines:
                start_time = solver.Value(start_task_per_machine_vars[(t_idx, m_idx)])
                end_time = solver.Value(end_task_per_machine_vars[(t_idx, m_idx)])
                machine_times.append((m_idx, start_time, end_time))

            results['schedule'].append({
                'task_id': t_id,
                'finish_time': task_end_time,
                'tardiness': tardiness,
                'machine_times': machine_times
            })

    return results


def generate_test_case(num_jobs, num_machines):
    """
    Generate a synthetic test case with the specified number of jobs and machines.
    """
    data = {
        "TaskID": list(range(1, num_jobs + 1)),
        "ReleaseDate": np.random.randint(0, 10, size=num_jobs),  # Random release dates
        "DueDate": np.random.randint(20, 50, size=num_jobs),    # Random due dates
        "Weight": np.random.randint(1, 5, size=num_jobs),       # Random weights
    }
    for i in range(1, num_machines + 1):
        data[f"Machine {i}"] = np.random.randint(1, 5, size=num_jobs)  # Random processing times

    return pd.DataFrame(data)

def evaluate_solver():
    """
    Run the solver with increasing job and machine sizes, and record results.
    """
    results = []
    max_jobs = 10  # Maximum number of jobs to test
    max_machines = 2  # Maximum number of machines to test
    time_limit = 60  # Time limit in seconds
    largest_set_of_jobs = {}
    batch = 5
      
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
            results.append({
                "NumJobs": num_jobs,
                "NumMachines": num_machines,
                "SolverStatus": solving_set[0][0]["status"],
                "ObjectiveValue": solving_set[0][0]["objective"],
                "SolveTime": solving_set[0][2],
                "TimeCapped": Time_capped
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


def create_download_link(val, filename,type):
    if type=="png":
        b64 = base64.b64encode(val).decode()  # Base64 encode the PNG file
        return f'<a href="data:image/png;base64,{b64}" download="{filename}.{type}">Download image</a>'
    else:
        b64 = base64.b64encode(val)
        return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.{type}">Download {filename}</a>'

# Run and display the evaluation results
if st.button("Run Solver Performance Tests"):
    with st.spinner("Running tests..."):
        performance_results,largest_set = evaluate_solver()
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

        