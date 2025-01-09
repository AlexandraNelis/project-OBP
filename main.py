import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
import plotly.express as px
import io
import altair as alt


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
