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
    add_scheduling_constraints(model, tasks, machines, variables, times)
    
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
        results['schedule'] = extract_solution(solver, tasks, machines, variables, times)
    
    return results


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


def create_gantt_chart(schedule, input_data):
    """
    Create a Gantt chart using Altair.
    The chart includes an interactive legend to highlight tasks.
    """

    chart_data = []
    for entry in schedule:
        task_id = entry['task_id']
        tardiness = entry['tardiness']
        weight = entry.get('weight', 1)
        release_date = input_data.loc[input_data['TaskID'] == task_id, 'ReleaseDate'].values[0]
        due_date = input_data.loc[input_data['TaskID'] == task_id, 'DueDate'].values[0]
        
        for (machine_num, start, end) in entry['machine_times']:
            chart_data.append({
                'Task': f"Task {task_id}",
                'Machine': f"M{machine_num}",
                'Start': start,
                'Finish': end,
                'Tardiness': tardiness,
                'ReleaseDate': release_date,
                'DueDate': due_date,
                'Weight': weight,
                'TaskID': task_id,
            })
    df_gantt = pd.DataFrame(chart_data)

    selection = alt.selection_multi(fields=['Task'], bind='legend')

    chart = alt.Chart(df_gantt).mark_bar().encode(
        x=alt.X('Start:Q', title='Start Time'),
        x2=alt.X2('Finish:Q'),
        y=alt.Y('Machine:N', sort='-x', title='Machine'),
        color=alt.Color(
            'Task:N',
            title='Task',
            sort=alt.EncodingSortField(
                field='TaskID', 
                order='ascending'
            )
        ),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),  # Dim non-selected tasks
        tooltip=[
            'Task:N',
            'Start:Q',
            'Finish:Q',
            'ReleaseDate:Q',
            'DueDate:Q',
            'Tardiness:Q',
            'Weight:Q'
        ]
    ).properties(
        title="Schedule Gantt Chart",
        width=800,
        height=400
    ).add_selection(
        selection
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
        t_weight = entry['weight']
        for (m_num, s, f) in entry['machine_times']:
            rows.append({
                'TaskID': t_id,
                'Weight': t_weight,
                'Machine': m_num,
                'Start': s,
                'Finish': f,
                'Tardiness': t_tardiness if m_num == entry['machine_times'][-1][0] else 0
            })
    return pd.DataFrame(rows)

def validate_schedule(schedule, input_data, machine_columns,status):
    """
    Validate the given schedule based on the constraints.

    Args:
    - schedule: List of task schedules. Each schedule includes:
        - 'task_id': Task identifier
        - 'machine_times': List of tuples (machine_idx, start_time, end_time)
    - input_data: DataFrame with the input data, including TaskID, ReleaseDate, DueDate, and machine times.
    - machine_columns: List of machine columns representing processing times for each machine.

    Returns:
    - A dictionary with results for each constraint:
        {
            "all_tasks_handled": (is_satisfied, message),
            "no_early_start": (is_satisfied, message),
            "all_machines_visited": (is_satisfied, message),
            "correct_processing_time": (is_satisfied, message)
        }
    """
    task_ids = input_data["TaskID"].tolist()
    results = {}

    # 1. Check that all tasks are handled
    all_tasks_handled = all(task_id in [task["task_id"] for task in schedule] for task_id in task_ids)
    if not all_tasks_handled:
        missing_tasks = set(task_ids) - set(task["task_id"] for task in schedule)
        results["all_tasks_handled"] = (
            False,
            f"The following tasks are missing from the schedule: {list(missing_tasks)}"
        )
    else:
        results["all_tasks_handled"] = (True, "All tasks are handled in the schedule.")

    # 2. Check that tasks do not start before their release dates
    early_start_violations = []
    for task in schedule:
        task_id = task["task_id"]
        release_date = input_data.loc[input_data["TaskID"] == task_id, "ReleaseDate"].values[0]
        for machine_num, start_time, _ in task["machine_times"]:
            if start_time < release_date:
                early_start_violations.append(
                    f"Task {task_id} on Machine {machine_num} starts before its release date ({start_time} < {release_date})."
                )

    if early_start_violations:
        results["no_early_start"] = (False, "\n".join(early_start_violations))
    else:
        results["no_early_start"] = (True, "No tasks start before their release dates.")

    # 3. Check that each task goes through all machines
    machines_visited_violations = []
    for task in schedule:
        machine_times = [machine_num-1 for machine_num, _, _ in task["machine_times"]]
        if sorted(machine_times) != sorted(range(len(machine_columns))):
            machines_visited_violations.append(
                f"Task {task['task_id']} does not go through all machines (expected {len(machine_columns)} machines)."
            )

    if machines_visited_violations:
        results["all_machines_visited"] = (False, "\n".join(machines_visited_violations))
    else:
        results["all_machines_visited"] = (True, "All tasks visit all required machines.")

    # 4. Check that each task spends the correct amount of time on each machine
    processing_time_violations = []
    for task in schedule:
        task_id = task["task_id"]
        for machine_num, start_time, end_time in task["machine_times"]:
            expected_duration = input_data.loc[input_data["TaskID"] == task_id, machine_columns[machine_num-1]].values[0]
            actual_duration = end_time - start_time
            if actual_duration != expected_duration:
                processing_time_violations.append(
                    f"Task {task_id} on Machine {machine_num} has an incorrect duration ({actual_duration} != {expected_duration})."
                )

    if processing_time_violations:
        results["correct_processing_time"] = (False, "\n".join(processing_time_violations))
    else:
        results["correct_processing_time"] = (True, "All tasks have the correct processing time on each machine.")

    #5 check if it is the optimal solution 
    if status == 4:
        results["Optimal solution"] = (True, "This is the optimal solution")
    else:
        results["Optimal solution"] = (False, "This is a feasible solution but not the optimal solution")
    return results


def main():
    # Set a wide layout for better visuals
    st.set_page_config(
        page_title="Multi-Machine Scheduling Optimizer",
        page_icon="🛠️",
        layout="wide"
    )

    # Add a sidebar for navigation and instructions
    with st.sidebar:
        st.title("Upload task data")
        st.markdown(
            """
            1. Upload an Excel file (.xlsx) with the following columns:
               - **TaskID** (unique identifier)
               - **ReleaseDate**
               - **DueDate**
               - **Weight**
               - **Machine1Time**, **Machine2Time**, etc.  \n
                  *(processing times on each machine)*
            2. Configure detected machine columns below.
            3. Click **Solve Scheduling Problem** to optimize the schedule.
            """
        )
        st.info("Ensure your file follows the required format to avoid errors.")
        uploaded_file = st.file_uploader("", type=["xlsx"])

    # Title and description in the main layout
    st.title("Multi-Machine Scheduling Optimizer")
    st.markdown(
        """
        Optimize your multi-machine scheduling tasks to minimize total **weighted tardiness**.  \n
        Use the **sidebar** to upload data and configure settings.
        """
    )

    if uploaded_file is not None:
        try:
            # Read and display input data
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            return

        # Convert column names to strings for validation
        df.columns = df.columns.map(str)

        # Define required and machine columns
        required_columns = {"TaskID", "ReleaseDate", "DueDate", "Weight"}
        possible_machine_columns = [col for col in df.columns if col.upper().startswith("M") and col.upper().endswith("TIME")]

        # Validate required columns
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return

        # Check for unexpected columns
        expected_columns = required_columns.union(possible_machine_columns)
        unexpected_columns = set(df.columns) - expected_columns
        if unexpected_columns:
            st.error(f"Unexpected columns found in the file: {', '.join(unexpected_columns)}")
            return

        # Check for empty cells
        if df.isnull().values.any():
            empty_cells = df[df.isnull().any(axis=1)]
            styled_df = empty_cells.style.applymap(lambda x: 'background-color: rgba(255, 0, 0, 0.6)' if pd.isnull(x) else '')

            st.error("The uploaded file contains empty cells. Please fill in the missing values. Affected rows are displayed below.")
            st.markdown("### Rows with Missing Values")
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            return

        # Detect machine columns automatically
        possible_machines_names = [f"Machine {col[1]}" for col in possible_machine_columns]

        st.markdown("### Input Data Preview")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Add a section to configure machine columns in the sidebar
        with st.sidebar:
            st.markdown("### Configure Machine Columns")
            machine_columns = st.multiselect(
                "Select Machine Columns (in order):",
                possible_machines_names,
                default=possible_machines_names
            )
            st.markdown("---")


        if st.button("Solve Scheduling Problem"):
            # Solve the scheduling problem
            with st.spinner("Solving the scheduling problem..."):
                machine_columns = [c[0] + c[-1] + "Time" for c in machine_columns]
                results = solve_scheduling_problem(df, machine_columns)
                status = results["status"]
                objective = results["objective"]
                schedule = results["schedule"]

            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                st.success(f"Solution found! Total Weighted Tardiness = {objective:.1f}")

                # Display results in an expandable container
                with st.expander("Gantt Chart", expanded=True):
                    fig_gantt = create_gantt_chart(schedule, df)
                    st.altair_chart(fig_gantt, use_container_width=True)

                with st.expander("Detailed Schedule"):
                    results_df = schedule_to_dataframe(schedule)
                    st.dataframe(results_df, use_container_width=True, hide_index=True)

                with st.expander("Validation Results"):
                    validation_results = validate_schedule(schedule, df, machine_columns,status)
                    for constraint, (is_satisfied, message) in validation_results.items():
                        if is_satisfied:
                            st.markdown(f"- **{constraint.replace('_', ' ').capitalize()}**: Satisfied ✅")
                        else:
                            st.markdown(f"- **{constraint.replace('_', ' ').capitalize()}**: Not satisfied ❌")
                            st.text(f"    {message}")

                # Add download button for schedule
                st.markdown("### Download Solution")
                output_bytes = io.BytesIO()
                with pd.ExcelWriter(output_bytes, engine="openpyxl") as writer:
                    results_df.to_excel(writer, index=False, sheet_name="Schedule")
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
        st.info("Upload an Excel file to start.")

if __name__ == "__main__":
    main()