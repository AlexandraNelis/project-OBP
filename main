import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
import plotly.express as px


def solve_scheduling_problem(df):
    """
    Given a pandas DataFrame with columns:
      'TaskID', 'ReleaseDate', 'DueDate', 'Weight', 'M1Time', 'M2Time', 'M3Time'
    returns a dictionary containing scheduling results and objective value.
    """
    tasks = df.to_dict('records')
    model = cp_model.CpModel()

    # Identify how many tasks and machines
    all_tasks = range(len(tasks))
    # Extend if you have more than 3 machines. For this example:
    machines = [0, 1, 2]

    # Collect processing times in a 2D list
    times = []
    for t in tasks:
        times.append([t['M1Time'], t['M2Time'], t['M3Time']])

    # Create variables
    horizon = sum(sum(times[t_idx]) for t_idx in all_tasks)
    start_vars = {}
    end_vars = {}
    intervals = {}

    for t_idx in all_tasks:
        for m_idx in machines:
            duration = times[t_idx][m_idx]
            start_var = model.NewIntVar(0, horizon, f'start_{t_idx}_m{m_idx}')
            end_var = model.NewIntVar(0, horizon, f'end_{t_idx}_m{m_idx}')
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                f'interval_{t_idx}_m{m_idx}')
            start_vars[(t_idx, m_idx)] = start_var
            end_vars[(t_idx, m_idx)] = end_var
            intervals[(t_idx, m_idx)] = interval_var

    # Constraints
    # 1) No overlap on same machine
    for m_idx in machines:
        machine_intervals = []
        for t_idx in all_tasks:
            machine_intervals.append(intervals[(t_idx, m_idx)])
        model.AddNoOverlap(machine_intervals)

    # 2) Flow constraint: M(m+1) cannot start before M(m) finishes for each task
    for t_idx in all_tasks:
        # M2 can't start before M1 is done
        model.Add(start_vars[(t_idx, 1)] >= end_vars[(t_idx, 0)])
        # M3 can't start before M2 is done
        model.Add(start_vars[(t_idx, 2)] >= end_vars[(t_idx, 1)])
        # Respect release date on machine 1
        release_date = tasks[t_idx]['ReleaseDate']
        model.Add(start_vars[(t_idx, 0)] >= release_date)

    # Tardiness
    tardiness_vars = []
    for t_idx in all_tasks:
        due_date = tasks[t_idx]['DueDate']
        weight = tasks[t_idx]['Weight']
        finish_time = end_vars[(t_idx, 2)]  # last machine end time
        lateness_var = model.NewIntVar(0, horizon, f'lateness_{t_idx}')
        model.Add(lateness_var >= finish_time - due_date)
        model.Add(lateness_var >= 0)

        weighted_tardiness_var = model.NewIntVar(0, weight * horizon, f'wtardiness_{t_idx}')
        model.Add(weighted_tardiness_var == lateness_var * weight)
        tardiness_vars.append(weighted_tardiness_var)

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
        # Gather schedule info
        for t_idx in all_tasks:
            t_id = tasks[t_idx]['TaskID']
            finish = solver.Value(end_vars[(t_idx, 2)])
            tardiness = max(0, finish - tasks[t_idx]['DueDate'])
            # Save per-machine info
            machine_times = []
            for m_idx in machines:
                s = solver.Value(start_vars[(t_idx, m_idx)])
                e = solver.Value(end_vars[(t_idx, m_idx)])
                machine_times.append((m_idx + 1, s, e))
            results['schedule'].append({
                'task_id': t_id,
                'finish_time': finish,
                'tardiness': tardiness,
                'machine_times': machine_times
            })

    return results


def create_gantt_chart(schedule):
    """
    Build a DataFrame suitable for a Plotly Gantt chart from the schedule results.
    schedule is a list of dicts with keys: 'task_id', 'machine_times'.
    Each 'machine_times' is a list of (machine_number, start, end).
    """
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
    # Create the Gantt chart with Plotly
    fig = px.timeline(
        df_gantt,
        x_start="Start",
        x_end="Finish",
        y="Machine",
        color="Task",
        title="Schedule Gantt Chart"
    )
    fig.update_yaxes(autorange="reversed")  # So M1 is at the top
    return fig


def main():
    st.title("Machine Scheduling with Weighted Tardiness Minimization")

    st.write("Upload an Excel file with columns: TaskID, ReleaseDate, DueDate, Weight, M1Time, M2Time, M3Time")

    # File uploader
    uploaded_file = st.file_uploader("Upload .xlsx file", type="xlsx")

    if uploaded_file is not None:
        # Read the Excel data into a pandas DataFrame
        df = pd.read_excel(uploaded_file)
        st.write("### Input Data")
        st.dataframe(df)

        if st.button("Solve Scheduling Problem"):
            results = solve_scheduling_problem(df)
            status = results['status']
            objective = results['objective']
            schedule = results['schedule']

            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                st.success(f"Solution found. Objective (Total Weighted Tardiness) = {objective}")

                # Display schedule results
                st.write("### Schedule Results")
                for item in schedule:
                    st.write(f"**Task {item['task_id']}**: "
                             f"Finish time = {item['finish_time']}, "
                             f"Tardiness = {item['tardiness']}")

                # (Optional) Show Gantt chart
                fig = create_gantt_chart(schedule)
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.error("No feasible solution found.")


if __name__ == "__main__":
    main()
