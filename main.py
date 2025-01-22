import streamlit as st
import pandas as pd
import io
import altair as alt
import time
import gurobipy as gp
from gurobipy import GRB, quicksum
from itertools import combinations
from ortools.sat.python import cp_model


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
    solver.parameters.max_time_in_seconds = 300.0  # 5 minute timeout
    solver.parameters.num_search_workers = 8  # Use multiple cores
    solver.parameters.log_search_progress = False  # If True, Enable logging for debugging
    
    start_time = time.time()
    status = solver.Solve(model)
    solve_time = time.time() - start_time
    
    results = {'status': status, 'objective': None, 'schedule': [], 'solve_time': solve_time}
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        results['objective'] = solver.ObjectiveValue()
        results['schedule'] = extract_solution(solver, tasks, machines, variables, times)
    
    return results


def validate_schedule(schedule, input_data, machine_columns, solver_name, status):
    """
    Validate the given schedule based on the constraints and solver status.

    Args:
      - schedule: List of task schedules. Each schedule includes:
          - 'task_id': Task identifier
          - 'machine_times': List of tuples (machine_idx, start_time, end_time)
      - input_data: DataFrame with the input data, including TaskID, ReleaseDate,
        DueDate, and machine times.
      - machine_columns: List of machine columns representing processing times
        for each machine.
      - solver_name: A string indicating which solver is used ("OR-Tools" or "Gurobi").
      - status: The solver's status code (integer).

    Returns:
      - A dictionary with results for each constraint plus an "Optimal solution"
        entry indicating if it's truly optimal or just feasible.
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
                    f"Task {task_id} on Machine {machine_num} starts before its release date "
                    f"({start_time} < {release_date})."
                )

    if early_start_violations:
        results["no_early_start"] = (False, "\n".join(early_start_violations))
    else:
        results["no_early_start"] = (True, "No tasks start before their release dates.")

    # 3. Check that each task goes through all machines
    machines_visited_violations = []
    for task in schedule:
        machine_times = [machine_num - 1 for machine_num, _, _ in task["machine_times"]]
        if sorted(machine_times) != sorted(range(len(machine_columns))):
            machines_visited_violations.append(
                f"Task {task['task_id']} does not go through all machines "
                f"(expected {len(machine_columns)} machines)."
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
            expected_duration = input_data.loc[
                input_data["TaskID"] == task_id,
                machine_columns[machine_num - 1]
            ].values[0]
            actual_duration = end_time - start_time
            if actual_duration != expected_duration:
                processing_time_violations.append(
                    f"Task {task_id} on Machine {machine_num} has an incorrect duration "
                    f"({actual_duration} != {expected_duration})."
                )

    if processing_time_violations:
        results["correct_processing_time"] = (False, "\n".join(processing_time_violations))
    else:
        results["correct_processing_time"] = (True, "All tasks have the correct processing time on each machine.")

    # 5. Check if it is the optimal solution based on the solver's status code
    #    For OR-Tools (CP-SAT): status == 4 means OPTIMAL
    #    For Gurobi:           status == 2 means OPTIMAL
    if solver_name == "OR-Tools":
        if status == 4:  # cp_model.OPTIMAL
            results["Optimal solution"] = (True, "This is the optimal solution (OR-Tools)")
        else:
            results["Optimal solution"] = (False, "Feasible (or no solution), not proven optimal (OR-Tools)")
    elif solver_name == "Gurobi":
        if status == 2:  # GRB.OPTIMAL
            results["Optimal solution"] = (True, "This is the optimal solution (Gurobi)")
        else:
            results["Optimal solution"] = (False, "Feasible (or no solution), not proven optimal (Gurobi)")
    else:
        # If unknown solver_name is given, default to a generic statement:
        results["Optimal solution"] = (False, "Unknown solver status; cannot determine optimality")

    return results


def validate_input_data(df):
    """
    Validate the input DataFrame structure and contents.
    Ensures:
    1. Required columns are present.
    2. No empty cells exist.
    3. ReleaseDate <= DueDate for all tasks.
    4. ReleaseDate is non-negative, and all other columns (DueDate, Weight, machine times) are strictly > 0.
    5. No strings or characters are allowed in any column.
    """
    # Required columns
    required_columns = {"TaskID", "ReleaseDate", "DueDate", "Weight"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check for empty cells
    if df.isnull().values.any():
        return False, "The uploaded file contains empty cells. Please fill in all values."
    
    # Validate numeric columns
    numeric_columns = {"ReleaseDate", "DueDate", "Weight"} | {
        col for col in df.columns if col.upper().startswith("M") and col.upper().endswith("TIME")
    }

    for col in numeric_columns:
        if col in df.columns:
            # Convert column to numeric, coerce invalid values to NaN
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isnull().any():
                return False, f"Column '{col}' contains invalid values (e.g., strings or characters)."
            
            # Non-negative for ReleaseDate, strictly > 1 for others
            if col == "ReleaseDate" and (df[col] < 0).any():
                return False, "ReleaseDate must be non-negative."
            elif col != "ReleaseDate" and (df[col] <= 0).any():
                return False, f"Column '{col}' must have values strictly greater than 0."
    
    # Check for logical consistency: ReleaseDate <= DueDate
    invalid_tasks = df[df["ReleaseDate"] > df["DueDate"]]
    if not invalid_tasks.empty:
        return False, (
            "ReleaseDate cannot be greater than DueDate for the following tasks: "
            f"{', '.join(map(str, invalid_tasks['TaskID'].tolist()))}"
        )
    
    return True, ""


def identify_machine_columns(df):
    """Identify valid machine columns from DataFrame."""
    return [
        col for col in df.columns 
        if col.upper().startswith("M") and 
        col.upper().endswith("TIME") and 
        col[1:-4].isdigit()
    ]

def create_gurobi_variables(model, jobs, machines, horizon):
    """Create and return Gurobi model variables."""
    x = model.addVars(jobs, machines, vtype=GRB.INTEGER, name="x")
    z = model.addVars(combinations(jobs, 2), machines, vtype=GRB.BINARY, name="z_machines")
    z1 = model.addVars(combinations(machines, 2), jobs, vtype=GRB.BINARY, name="z_jobs")
    T = model.addVars(jobs, lb=0, vtype=GRB.INTEGER, name="T")
    
    return x, z, z1, T

def add_gurobi_basic_constraints(model, jobs, machines, x, T, tasks, times, horizon):
    """Add basic constraints to Gurobi model."""
    for i in jobs:
        for k in machines:
            model.addConstr(x[i, k] >= tasks[i]['ReleaseDate'])
            model.addConstr(T[i] >= x[i, k] + times[i][k] - tasks[i]['DueDate'])

def add_gurobi_disjunctive_constraints(model, jobs, machines, x, z, z1, times, horizon):
    """Add disjunctive constraints to Gurobi model."""
    # Machine constraints
    for k in machines:
        for i, j in combinations(jobs, 2):
            model.addConstr(x[i, k] + times[i][k] <= x[j, k] + horizon * (1 - z[i, j, k]))
            model.addConstr(x[j, k] + times[j][k] <= x[i, k] + horizon * z[i, j, k])
    
    # Job constraints
    for k in jobs:
        for i, j in combinations(machines, 2):
            model.addConstr(x[k, i] + times[k][i] <= x[k, j] + horizon * (1 - z1[i, j, k]))
            model.addConstr(x[k, j] + times[k][j] <= x[k, i] + horizon * z1[i, j, k])

def solve_scheduling_problem_gurobi(df, machine_columns):
    """Main Gurobi solver function."""
    tasks = df.to_dict('records')
    model = gp.Model("WeightedTardinessScheduling")
    model.setParam("OutputFlag", 0) # If 1, enable Gurobi output logging for debugging

    # Set Gurobi parameters for tighter control
    model.Params.MIPGap = 0        # Ensure no relative gap
    model.Params.MIPGapAbs = 0     # Ensure no absolute gap

    num_tasks = len(tasks)
    jobs = list(range(num_tasks))
    machines = list(range(len(machine_columns)))
    times = [[t[col] for col in machine_columns] for t in tasks]
    horizon = sum(sum(t_row) for t_row in times)

    # Create variables
    x, z, z1, T = create_gurobi_variables(model, jobs, machines, horizon)
    
    # Set objective
    model.setObjective(quicksum(tasks[j]['Weight'] * T[j] for j in jobs), GRB.MINIMIZE)
    
    # Add constraints
    add_gurobi_basic_constraints(model, jobs, machines, x, T, tasks, times, horizon)
    add_gurobi_disjunctive_constraints(model, jobs, machines, x, z, z1, times, horizon)

    # Solve and extract results
    model.optimize()
    return extract_gurobi_solution(model, tasks, machines, x, T, times)

def extract_gurobi_solution(model, tasks, machines, x, T, times):
    """Extract solution from Gurobi model."""
    status = model.status
    results = {'status': status, 'objective': None, 'schedule': []}

    if status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        results['objective'] = model.ObjVal
        results['schedule'] = build_gurobi_schedule(tasks, machines, x, T, times)
    
    return results

def build_gurobi_schedule(tasks, machines, x, T, times):
    """Build schedule from Gurobi solution."""
    schedule = []
    for t_idx in range(len(tasks)):
        machine_times = [
            (m_idx + 1,
             x[t_idx, m_idx].X,
             x[t_idx, m_idx].X + times[t_idx][m_idx])
            for m_idx in machines
        ]
        
        schedule.append({
            'task_id': tasks[t_idx]['TaskID'],
            'tardiness': T[t_idx].X,
            'machine_times': machine_times,
            'weight': tasks[t_idx]['Weight']
        })
    
    return schedule

def create_ortools_model():
    """Create and return a new OR-Tools CP model."""
    return cp_model.CpModel()

def setup_ortools_solver():
    """Configure and return OR-Tools CP solver."""
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300.0
    solver.parameters.num_search_workers = 8
    return solver

def format_gantt_data(schedule, input_data):
    """Format schedule data for Gantt chart."""
    chart_data = []
    for entry in schedule:
        task_id = entry['task_id']
        task_info = input_data[input_data['TaskID'] == task_id].iloc[0]
        
        for machine_num, start, end in entry['machine_times']:
            chart_data.append({
                'Task': f"{task_id}",
                'Machine': f"M{machine_num}",
                'Start': start,
                'Finish': end,
                'Tardiness': entry['tardiness'],
                'ReleaseDate': task_info['ReleaseDate'],
                'DueDate': task_info['DueDate'],
                'Weight': entry['weight'],
                'TaskID': task_id,
            })
    
    return pd.DataFrame(chart_data)

def create_gantt_chart(schedule, input_data):
    """Create Altair Gantt chart."""
    df_gantt = format_gantt_data(schedule, input_data)
    selection = alt.selection_point(fields=['Task'], bind='legend')
    
    return build_gantt_chart(df_gantt, selection)

def build_gantt_chart(df_gantt, selection):
    """Build and configure Altair chart object."""
    chart = alt.Chart(df_gantt).mark_bar().encode(
        x=alt.X('Start:Q', title='Start Time'),
        x2=alt.X2('Finish:Q'),
        y=alt.Y('Machine:N', sort='-x', title='Machine'),
        color=alt.Color(
            'Task:N',
            title='Task',
            sort=alt.EncodingSortField(field='TaskID', order='ascending'),
            scale=alt.Scale(scheme='turbo'), 
            legend=alt.Legend(
                title="Tasks",
                symbolLimit=100,
                orient="bottom",
                labelFontSize=10,
                titleFontSize=12,
                labelLimit=2000,
                columns=10,
                direction="vertical")),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        tooltip=['Task:N','Start:Q','Finish:Q','ReleaseDate:Q','DueDate:Q','Tardiness:Q','Weight:Q']
        ).properties(title="Schedule Gantt Chart", width=800, height=600
    ).add_selection(selection)

    return chart

def setup_streamlit_ui():
    """Configure Streamlit UI layout."""
    st.set_page_config(
        page_title="TEST Multi-Machine Scheduling Optimizer",
        page_icon="🛠️",
        layout="wide"
    )
    
def setup_sidebar():
    """Configure and return sidebar elements."""
    with st.sidebar:
        st.title("Upload task data")
        show_upload_instructions()
        uploaded_file = st.file_uploader("Upload your data file:", type=["xlsx"])
        st.markdown("---")
        
        st.title("Select solver")
        solver_choice = st.selectbox(
            "Select Solver:",
            ["OR-Tools", "Gurobi"],
            help="Choose the solver to optimize the schedule."
        )
        st.markdown("---")
        
    return uploaded_file, solver_choice

def show_upload_instructions():
    """Display file upload instructions."""
    st.markdown(
        """
        1. Upload an Excel file (.xlsx) with the following columns:
           - **TaskID** (unique identifier)
           - **ReleaseDate**
           - **DueDate**
           - **Weight**
           - **M1Time**, **M2Time**, etc.  \n
              *(processing times on each machine)*
        2. Configure detected machine columns below.
        3. Click **Solve Scheduling Problem** to optimize the schedule.
        """
    )
    st.info("Ensure your file follows the required format to avoid errors.")

def handle_solution_download(results_df, input_df):
    """Handle solution download functionality."""
    output_bytes = io.BytesIO()
    with pd.ExcelWriter(output_bytes, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name="Schedule")
        input_df.to_excel(writer, index=False, sheet_name="InputData")
    output_bytes.seek(0)
    
    st.download_button(
        label="Download Schedule as Excel",
        data=output_bytes,
        file_name="schedule_solution.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        

def display_validation_results(validation_results):
    """Display schedule validation results."""
    for constraint, (is_satisfied, message) in validation_results.items():
        if is_satisfied:
            st.markdown(f"- **{constraint.replace('_', ' ').capitalize()}**: Satisfied ✅")
        else:
            st.markdown(f"- **{constraint.replace('_', ' ').capitalize()}**: Not satisfied ❌")
            st.text(f"    {message}")


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


def main():
    setup_streamlit_ui()
    uploaded_file, solver_choice = setup_sidebar()
    
    st.title("Multi-Machine Scheduling Optimizer")
    st.markdown(
        """
        Optimize your multi-machine scheduling tasks to minimize total **weighted tardiness**.  \n
        Use the **sidebar** to upload data and configure settings.
        """
    )

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            df.columns = df.columns.map(str)
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            return

        # Validate input data
        is_valid, error_message = validate_input_data(df)
        if not is_valid:
            st.error(error_message)
            return

        st.markdown("### Input Data Preview")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Configure machine columns
        with st.sidebar:
            st.title("Configure Machine Columns")
            machine_columns = st.multiselect(
                "Select Machine Columns (in order):",
                identify_machine_columns(df),
                default=identify_machine_columns(df)
            )
            st.markdown("---")

        if st.button("Solve Scheduling Problem"):
            with st.spinner(f"Solving the scheduling problem with {solver_choice}..."):
                if solver_choice == "OR-Tools":
                    results = solve_scheduling_problem(df, machine_columns)
                else:
                    results = solve_scheduling_problem_gurobi(df, machine_columns)

            if results["status"] in [cp_model.OPTIMAL, cp_model.FEASIBLE, GRB.OPTIMAL, GRB.SUBOPTIMAL]:
                st.balloons()
                st.success(f"Solution found! Total Weighted Tardiness = {results['objective']:.1f}")

                with st.expander("Gantt Chart", expanded=True):
                    fig_gantt = create_gantt_chart(results["schedule"], df)
                    st.altair_chart(fig_gantt, use_container_width=True)

                with st.expander("Detailed Schedule"):
                    results_df = schedule_to_dataframe(results["schedule"])
                    st.dataframe(results_df, use_container_width=True, hide_index=True)

                with st.expander("Validation Results"):
                    validation_results = validate_schedule(
                        results["schedule"], df, machine_columns, solver_choice, results["status"]
                    )
                    display_validation_results(validation_results)

                st.markdown("### Download Solution")
                handle_solution_download(results_df, df)
            else:
                st.error("No feasible solution found. Please check your input data.")
    else:
        st.info("Upload an Excel file to start.")

if __name__ == "__main__":
    main()
