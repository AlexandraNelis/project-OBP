import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
import plotly.express as px
import io
import altair as alt
import time

import gurobipy as gp
from gurobipy import Model, GRB, quicksum


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
    model = gp.Model("WeightedTardinessScheduling")

    num_tasks = len(tasks)
    jobs = list(range(num_tasks))
    machines = list(range(len(machine_columns)))  # e.g. [0,1,2] if there are 3 machine columns

    # Build a 2D list for processing times: times[task_idx][machine_idx]
    times = []
    for t in tasks:
        row_times = [t[col] for col in machine_columns]
        times.append(row_times)

    # Define horizon as sum of all processing times (a crude upper bound)
    horizon = sum(sum(t_row) for t_row in times)

    x = model.addVars(jobs, machines, vtype=GRB.INTEGER, name="x")  # Start times
    z1 = model.addVars(machines, machines, jobs, vtype=GRB.BINARY, name="z")  # Binary variables
    z = model.addVars(jobs, jobs, machines, vtype=GRB.BINARY, name="z")  # Binary variables
    T = model.addVars(jobs, vtype=GRB.INTEGER, name="T")  # Tardiness for each job

    model.setObjective(quicksum(tasks[j]['Weight'] * T[j] for j in jobs), GRB.MINIMIZE)
    
    # Constraints
    for i in jobs:
        for k in machines:
            # Start times must be later than the release date
            model.addConstr(x[i, k] >= tasks[i]['ReleaseDate'], name=f"start_time_nonneg_{i}_{k}")
            # Precedence constraint
            model.addConstr(x[i, k] >= 0, name=f"start_time_nonneg_{i}_{k}")
            # Tardiness
            model.addConstr(T[i] >= x[i, k] + times[i][k] - tasks[i]['DueDate'], name=f"tardiness_{i}_{k}")
            model.addConstr(T[i] >= 0, name=f"non_negative_tardiness_{i}")
    
    for k in machines:
        for i in jobs:
            for j in jobs:
                if i < j:  # Avoid duplicate pairs
                    # Disjunctive constraints
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
                    # Disjunctive constraints
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
            #print("schedule:", schedule)

            if status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
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
