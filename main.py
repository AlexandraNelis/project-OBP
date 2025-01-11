import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
import io
import altair as alt
import time


def create_model_variables(model, tasks, machines, machine_columns, horizon):
    """Create and return all model variables."""
    times = [[t[col] for col in machine_columns] for t in tasks]
    
    variables = {
        'start': {},
        'end': {},
        'intervals': {},
        'task_end': {},
    }
    
    for t_idx in range(len(tasks)):
        for m_idx in machines:
            duration = times[t_idx][m_idx]
            
            start_var = model.NewIntVar(0, horizon, f'start_{t_idx}_m{m_idx}')
            end_var = model.NewIntVar(0, horizon, f'end_{t_idx}_m{m_idx}')
            interval_var = model.NewIntervalVar(
                start_var, duration, end_var, f'interval_{t_idx}_m{m_idx}'
            )
            
            variables['start'][(t_idx, m_idx)] = start_var
            variables['end'][(t_idx, m_idx)] = end_var
            variables['intervals'][(t_idx, m_idx)] = interval_var
        
        variables['task_end'][t_idx] = model.NewIntVar(0, horizon, f'end_time_task{t_idx}')
    
    return variables, times

def add_scheduling_constraints(model, tasks, machines, variables):
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
        end_times_task = [variables['end'][(t_idx, m_idx)] 
                         for m_idx in machines]
        model.AddMaxEquality(variables['task_end'][t_idx], end_times_task)

def create_objective_variables(model, tasks, variables, horizon):
    """Create and return tardiness variables for the objective function."""
    tardiness_vars = []
    
    for t_idx, task in enumerate(tasks):
        due_date = task['DueDate']
        weight = task['Weight']
        
        lateness_var = model.NewIntVar(0, horizon, f'lateness_task{t_idx}')
        model.Add(lateness_var >= variables['task_end'][t_idx] - due_date)
        model.Add(lateness_var >= 0)
        
        weighted_tardiness = model.NewIntVar(0, weight * horizon, 
                                           f'wtardiness_task{t_idx}')
        model.Add(weighted_tardiness == lateness_var * weight)
        tardiness_vars.append(weighted_tardiness)
    
    return tardiness_vars

def extract_solution(solver, tasks, machines, variables):
    """Extract the solution from the solver."""
    schedule = []
    
    for t_idx, task in enumerate(tasks):
        task_end_time = solver.Value(variables['task_end'][t_idx])
        tardiness = max(0, task_end_time - task['DueDate'])
        
        machine_times = [
            # Increment machine index by 1 for display purposes
            (m_idx + 1,  # Machine number starts from 1
             solver.Value(variables['start'][(t_idx, m_idx)]),
             solver.Value(variables['end'][(t_idx, m_idx)]))
            for m_idx in machines
        ]
        
        schedule.append({
            'task_id': task['TaskID'],
            'finish_time': task_end_time,
            'tardiness': tardiness,
            'machine_times': machine_times
        })
    
    return schedule

def solve_scheduling_problem(df, machine_columns):
    """
    Optimized version of the scheduling problem solver.
    """
    tasks = df.to_dict('records')
    machines = list(range(len(machine_columns)))
    
    # Calculate horizon
    horizon = max(
        max(t['DueDate'] for t in tasks),
        sum(max(t[col] for col in machine_columns) for t in tasks)
    )
    
    # Initialize model
    model = cp_model.CpModel()
    
    # Create variables
    variables, times = create_model_variables(model, tasks, machines, 
                                           machine_columns, horizon)
    
    # Add constraints
    add_scheduling_constraints(model, tasks, machines, variables)
    
    # Create objective function
    tardiness_vars = create_objective_variables(model, tasks, variables, horizon)
    model.Minimize(sum(tardiness_vars))
    
    # Solve
    solver = cp_model.CpSolver()
    
    # Add time limit and other parameters to improve performance
    solver.parameters.max_time_in_seconds = 300.0  # 5 minute timeout
    solver.parameters.num_search_workers = 8  # Use multiple cores
    # solver.parameters.log_search_progress = True  # Enable logging
    
    start_time = time.time()
    status = solver.Solve(model)
    solve_time = time.time() - start_time
    
    results = {'status': status, 'objective': None, 'schedule': [], 'solve_time': solve_time}
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        results['objective'] = solver.ObjectiveValue()
        results['schedule'] = extract_solution(solver, tasks, machines, variables)
    
    return results


def create_gantt_chart(schedule):
    """
    Create a Gantt chart using Altair to ensure task lines are displayed properly.
    """
    chart_data = []
    for entry in schedule:
        task_id = entry['task_id']
        for (machine_num, start, end) in entry['machine_times']:
            chart_data.append({
                'Task': f"Task {task_id}",
                'Machine': f"M{machine_num}",  # Display machine numbers starting from 1
                'Start': start,
                'Finish': end
            })
    df_gantt = pd.DataFrame(chart_data)

    # Use Altair to ensure task lines are properly displayed
    chart = alt.Chart(df_gantt).mark_bar().encode(
        x=alt.X('Start:Q', title='Start Time'),
        x2=alt.X2('Finish:Q'),
        y=alt.Y('Machine:N', sort='-x', title='Machine'),
        color='Task:N',
        tooltip=['Task', 'Machine', 'Start', 'Finish']
    ).properties(
        title="Schedule Gantt Chart",
        width=800,
        height=400
    )
    return chart

def schedule_to_dataframe(schedule):
    """
    Convert the schedule list of dicts into a row-based DataFrame so it’s easy to export to Excel.
    Each machine-time range becomes a separate row:
      [TaskID, Machine, Start, Finish, Tardiness (only repeated for convenience)]
    """
    rows = []
    for entry in schedule:
        t_id = entry['task_id']
        t_tardiness = entry['tardiness']
        for (m_num, s, f) in entry['machine_times']:
            rows.append({
                'TaskID': t_id,
                'Machine': m_num,  # Display machine numbers starting from 1
                'Start': s,
                'Finish': f,
                'Tardiness': t_tardiness if m_num == entry['machine_times'][-1][0] else 0
            })
    return pd.DataFrame(rows)


def main():
    st.title("Multi-Machine Scheduling (Weighted Tardiness Minimization)")

    st.write(
        """
        **Instructions**:  
        1. Upload an Excel file with columns:
           - **TaskID** (unique integer or label)
           - **ReleaseDate**
           - **DueDate**
           - **Weight**
           - **M1Time**, **M2Time**, **M3Time**, etc. (one column per machine)  
        2. Click **Solve** to compute an optimal or feasible schedule.  
        3. View the Gantt chart, schedule details, and **download** the solution.
        """
    )

    # File uploader for real input data
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write("### Input Data")
        st.dataframe(df)

        # Identify machine columns automatically (columns that start with "M" and end with "Time")
        # Or you can specify them directly if you prefer.
        possible_machine_cols = [c for c in df.columns if c.upper().startswith("M") and c.upper().endswith("TIME")]
        st.write(f"**Detected machine time columns**: {possible_machine_cols}")

        # Let user verify or override which columns are machine times
        machine_columns = st.multiselect(
            "Select Machine Columns (in order):",
            possible_machine_cols,
            default=possible_machine_cols
        )

        if st.button("Solve Scheduling Problem"):
            # Solve
            results = solve_scheduling_problem(df, machine_columns)
            status = results['status']
            objective = results['objective']
            schedule = results['schedule']
            # Removed the print statement for the schedule

            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                st.success(f"Solution found. Total Weighted Tardiness = {objective}")

                # Display a Gantt chart
                fig_gantt = create_gantt_chart(schedule)
                st.altair_chart(fig_gantt, use_container_width=True)

                # Display detailed schedule info
                st.write("### Detailed Schedule")
                results_df = schedule_to_dataframe(schedule)
                st.dataframe(results_df)

                # Download solution as Excel
                output_bytes = io.BytesIO()
                with pd.ExcelWriter(output_bytes, engine="openpyxl") as writer:
                    results_df.to_excel(writer, index=False, sheet_name="Schedule")
                    # Optionally include the input data in the same file:
                    df.to_excel(writer, index=False, sheet_name="InputData")
                output_bytes.seek(0)

                st.download_button(
                    label="Download Schedule as Excel",
                    data=output_bytes,
                    file_name="schedule_solution.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            else:
                st.error("No feasible solution found. Please check your input data.")
    else:
        st.info("Please upload an Excel file to begin.")


if __name__ == "__main__":
    main()