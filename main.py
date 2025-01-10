import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
import plotly.express as px
import io
import altair as alt
import time



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


def create_gantt_chart(schedule, input_data):
    """
    Create a Gantt chart using Altair to ensure task lines are displayed properly.
    """
    chart_data = []
    for entry in schedule:
        task_id = entry['task_id']
        tardiness = entry['tardiness']
        release_date = input_data.loc[input_data['TaskID'] == task_id, 'ReleaseDate'].values[0]
        due_date = input_data.loc[input_data['TaskID'] == task_id, 'DueDate'].values[0]
        
        for (machine_num, start, end) in entry['machine_times']:
            chart_data.append({
                'Task': f"Task {task_id}",
                'Machine': f"M{machine_num +1}",
                'Start': start,
                'Finish': end,
                'Tardiness': tardiness,
                'ReleaseDate': release_date,
                'DueDate': due_date,
                'TaskID': task_id,  # Add TaskID for clarity in the tooltip
            })
    df_gantt = pd.DataFrame(chart_data)

    # Use Altair to ensure task lines are properly displayed
    chart = alt.Chart(df_gantt).mark_bar().encode(
        x=alt.X('Start:Q', title='Start Time'),
        x2=alt.X2('Finish:Q'),
        y=alt.Y('Machine:N', sort='-x', title='Machine'),
        color='Task:N',  # Color bars by task
        tooltip=[  # Add all relevant data to the tooltip
            'Task:N',
            'Start:Q',
            'Finish:Q',
            'Tardiness:Q',
            'ReleaseDate:Q',
            'DueDate:Q'
        ]
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
                'Machine': m_num,
                'Start': s,
                'Finish': f,
                'Tardiness': t_tardiness if m_num == entry['machine_times'][-1][0] else 0
            })
    return pd.DataFrame(rows)

def validate_schedule(schedule, input_data, machine_columns):
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
        machine_times = [machine_num for machine_num, _, _ in task["machine_times"]]
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
            expected_duration = input_data.loc[input_data["TaskID"] == task_id, machine_columns[machine_num]].values[0]
            actual_duration = end_time - start_time
            if actual_duration != expected_duration:
                processing_time_violations.append(
                    f"Task {task_id} on Machine {machine_num} has an incorrect duration ({actual_duration} != {expected_duration})."
                )

    if processing_time_violations:
        results["correct_processing_time"] = (False, "\n".join(processing_time_violations))
    else:
        results["correct_processing_time"] = (True, "All tasks have the correct processing time on each machine.")

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
        st.title("⚙️ Settings")
        st.markdown(
            """
            **Instructions**:
            1. Upload an Excel file with the following columns:
               - **TaskID** (unique identifier)
               - **ReleaseDate**
               - **DueDate**
               - **Weight**
               - **Machine 1**, **Machine 2**, etc. (processing times on each machine)
            2. Configure detected machine columns in the sidebar.
            3. Click **Solve** to optimize the schedule.
            4. Review results, validation, and download the schedule.
            """
        )
        st.markdown("---")
        st.info("Ensure your file follows the required format to avoid errors.")

    # Title and description in the main layout
    st.title("📅 Multi-Machine Scheduling Optimizer")
    st.markdown(
        """
        Optimize your multi-machine scheduling tasks to minimize total **weighted tardiness**.
        Use the **sidebar** to upload data and configure settings.
        """
    )

    # File uploader
    st.markdown("### Upload Your Excel File")
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

    if uploaded_file is not None:
        # Read and display input data
        df = pd.read_excel(uploaded_file)
        st.markdown("### 📋 Input Data Preview")
        st.dataframe(df, use_container_width=True)

        # Detect machine columns automatically
        possible_machine_cols = [c for c in df.columns if c.upper().startswith("M") and c.upper().endswith("TIME")]
        possible_machines_names = ["Machine " + c[1]  for c in df.columns if c.upper().startswith("M") and c.upper().endswith("TIME")]
        
        # Add a section to configure machine columns in the sidebar
        with st.sidebar:
            st.markdown("### 🏗️ Configure Machine Columns")
            machine_columns = st.multiselect(
                "Select Machine Columns (in order):",
                possible_machines_names,
                default=possible_machines_names
            )
            st.markdown("---")

        if st.button("🔍 Solve Scheduling Problem"):
            # Solve the scheduling problem
            with st.spinner("Solving the scheduling problem..."):
                machine_columns = [c[0] + c[-1] + "Time" for c in machine_columns]
                results = solve_scheduling_problem(df, machine_columns)
                status = results["status"]
                objective = results["objective"]
                schedule = results["schedule"]

            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                st.success(f"Solution found! Total Weighted Tardiness = {objective:.2f}")

                # Display results in an expandable container
                with st.expander("📊 Gantt Chart", expanded=True):
                    fig_gantt = create_gantt_chart(schedule, df)
                    st.altair_chart(fig_gantt, use_container_width=True)

                with st.expander("📝 Detailed Schedule"):
                    results_df = schedule_to_dataframe(schedule)
                    st.dataframe(results_df, use_container_width=True)

                with st.expander("✅ Validation Results"):
                    validation_results = validate_schedule(schedule, df, machine_columns)
                    for constraint, (is_satisfied, message) in validation_results.items():
                        if is_satisfied:
                            st.markdown(f"- **{constraint.replace('_', ' ').capitalize()}**: Satisfied ✅")
                        else:
                            st.markdown(f"- **{constraint.replace('_', ' ').capitalize()}**: Not satisfied ❌")
                            st.text(f"    {message}")

                # Add download button for schedule
                st.markdown("### 📥 Download Solution")
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