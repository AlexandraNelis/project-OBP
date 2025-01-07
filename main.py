import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
import plotly.express as px
import io


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
    start_vars = {}
    end_vars = {}
    intervals = {}

    for t_idx in range(num_tasks):
        for m_idx in machines:
            duration = times[t_idx][m_idx]
            start_var = model.NewIntVar(0, horizon, f'start_{t_idx}_m{m_idx}')
            end_var = model.NewIntVar(0, horizon, f'end_{t_idx}_m{m_idx}')
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                f'interval_{t_idx}_m{m_idx}')
            start_vars[(t_idx, m_idx)] = start_var
            end_vars[(t_idx, m_idx)] = end_var
            intervals[(t_idx, m_idx)] = interval_var

    # Constraint 1: No overlap on each machine
    for m_idx in machines:
        model.AddNoOverlap([intervals[(t_idx, m_idx)] for t_idx in range(num_tasks)])

    # Constraint 2: Flow constraint from machine m to m+1 for each task
    # and release date constraints for machine 1
    for t_idx in range(num_tasks):
        # For each consecutive pair of machines
        for m_idx in range(len(machines) - 1):
            model.Add(start_vars[(t_idx, m_idx + 1)] >= end_vars[(t_idx, m_idx)])

        # Release date on the first machine
        release_date = tasks[t_idx]['ReleaseDate']
        model.Add(start_vars[(t_idx, 0)] >= release_date)

    # Tardiness variables
    tardiness_vars = []
    for t_idx in range(num_tasks):
        due_date = tasks[t_idx]['DueDate']
        weight = tasks[t_idx]['Weight']

        # Finish time is the end on the last machine
        last_machine = machines[-1]
        finish_time_var = end_vars[(t_idx, last_machine)]

        # lateness = max(0, finish - due_date)
        # create an IntVar for lateness
        lateness_var = model.NewIntVar(0, horizon, f'lateness_{t_idx}')
        model.Add(lateness_var >= finish_time_var - due_date)
        model.Add(lateness_var >= 0)

        # Weighted tardiness = weight * lateness
        weighted_tardiness_var = model.NewIntVar(0, weight * horizon, f'wtardiness_{t_idx}')
        model.Add(weighted_tardiness_var == lateness_var * weight)

        tardiness_vars.append(weighted_tardiness_var)

    # Objective: minimize sum of weighted tardiness
    model.Minimize(sum(tardiness_vars))

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

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
            finish_time = solver.Value(end_vars[(t_idx, machines[-1])])
            tardiness = max(0, finish_time - tasks[t_idx]['DueDate'])

            # Save the (machine, start, end) for Gantt
            machine_times = []
            for m_idx in machines:
                s = solver.Value(start_vars[(t_idx, m_idx)])
                e = solver.Value(end_vars[(t_idx, m_idx)])
                machine_times.append((m_idx + 1, s, e))

            results['schedule'].append({
                'task_id': t_id,
                'finish_time': finish_time,
                'tardiness': tardiness,
                'machine_times': machine_times
            })

    return results


def create_gantt_chart(schedule):
    chart_data = []
    for entry in schedule:
        task_id = entry['task_id']
        for (machine_num, start, end) in entry['machine_times']:
            chart_data.append({
                'Task': f"Task {task_id}",
                'Machine': f"M{machine_num}",
                'Start': start,
                'Finish': end
            })
    df_gantt = pd.DataFrame(chart_data)
    fig = px.timeline(
        df_gantt,
        x_start="Start",
        x_end="Finish",
        y="Machine",
        color="Task",
        title="Schedule Gantt Chart"
    )
    # Force numeric axis
    fig.update_xaxes(type="linear")
    fig.update_yaxes(autorange="reversed")
    return fig


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
            print("schedule:", schedule)

            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                st.success(f"Solution found. Total Weighted Tardiness = {objective}")

                # Display a Gantt chart
                fig_gantt = create_gantt_chart(schedule)
                st.plotly_chart(fig_gantt, use_container_width=True)

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
