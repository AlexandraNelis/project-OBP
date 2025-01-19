import pandas as pd
from ortools.sat.python import cp_model
from typing import Dict, List, Tuple
import time


def create_model_variables(model: cp_model.CpModel, 
                         tasks: List[Dict], 
                         machines: List[int], 
                         machine_columns: List[str], 
                         horizon: int) -> Tuple[Dict, List[List[int]]]:
    """Create and return all model variables: the start and end times (and the corresponding interval duration) of tasks on machines,
    with the list of processing times for each task on each machine given by the input"""
    times = [[t[col] for col in machine_columns] for t in tasks]
    variables = {
        'start': {},
        'intervals': {},
        'task_end': {},
    }    
    for task_idx, _ in enumerate(tasks):
        for machine_idx in machines:
            duration = times[task_idx][machine_idx]
            start_var = model.NewIntVar(0, horizon, f'start_{task_idx}_m{machine_idx}')
            interval_var = model.NewIntervalVar(
                start_var, duration, start_var + duration, 
                f'interval_{task_idx}_m{machine_idx}'
            )
            variables['start'][(task_idx, machine_idx)] = start_var
            variables['intervals'][(task_idx, machine_idx)] = interval_var
        variables['task_end'][task_idx] = model.NewIntVar(
            0, horizon, f'end_time_task{task_idx}'
        )
    
    return variables, times


def add_scheduling_constraints(model: cp_model.CpModel, 
                             tasks: List[Dict], 
                             machines: List[int], 
                             variables: Dict, 
                             times: List[List[int]]) -> None:
    """Add all scheduling constraints to the model."""
    #1. No Overlap on Machines
    for machine_idx in machines:
        machine_intervals = [
            variables['intervals'][(task_idx, machine_idx)] 
            for task_idx, _ in enumerate(tasks)
        ]
        model.AddNoOverlap(machine_intervals)
    
    #2. No task overlaps on the same machine
    for task_idx, _ in enumerate(tasks):
        task_intervals = [
            variables['intervals'][(task_idx, machine_idx)] 
            for machine_idx in machines
        ]
        model.AddNoOverlap(task_intervals)
        
        #3. Respect Release Dates
        for machine_idx in machines:
            model.Add(variables['start'][(task_idx, machine_idx)] >= 
                     tasks[task_idx]['ReleaseDate'])
        
        #4. Correct End Times: end time for each task is its maximum end time for all the machines
        end_times_task = [
            variables['start'][(task_idx, machine_idx)] + times[task_idx][machine_idx]
            for machine_idx in machines
        ]
        model.AddMaxEquality(variables['task_end'][task_idx], end_times_task)


def create_objective_variables(model: cp_model.CpModel, 
                             tasks: List[Dict], 
                             variables: Dict, 
                             horizon: int) -> List:
    """Create and return tardiness variables for the objective function."""
    weighted_tardiness = []
    
    for idx, task in enumerate(tasks):
        tardiness_var = model.NewIntVar(0, horizon, f'lateness_task{idx}')
        model.Add(tardiness_var >= variables['task_end'][idx] - task['DueDate'])
        model.Add(tardiness_var >= 0)
        weighted_tardiness.append(tardiness_var * task['Weight']) #to prioritize important tasks
    
    return weighted_tardiness


def solve_scheduling_problem(df: pd.DataFrame, 
                           machine_columns: List[str]) -> Dict:
    """Solve the scheduling problem and return results."""
    tasks = df.to_dict('records')
    machines = list(range(len(machine_columns)))
    horizon = max(
        max(t['DueDate'] for t in tasks),
        sum(max(t[col] for col in machine_columns) for t in tasks)
    ) #defines the maximum possible end time for any task in the optimization problem
    model = cp_model.CpModel()
    variables, times = create_model_variables(
        model, tasks, machines, machine_columns, horizon
    )
    add_scheduling_constraints(model, tasks, machines, variables, times)
    tardiness_vars = create_objective_variables(model, tasks, variables, horizon)
    model.Minimize(sum(tardiness_vars))
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300.0 #5 minutes
    solver.parameters.num_search_workers = 8 #the solver can use up to 8 CPU cores to explore multiple parts of the solution space simultaneously (parallel search)
    
    start_time = time.time()
    status = solver.Solve(model)
    solve_time = time.time() - start_time
    
    results = {
        'status': status, 
        'objective': None, 
        'schedule': [], 
        'solve_time': solve_time
    }

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        results['objective'] = solver.ObjectiveValue()
        results['schedule'] = extract_solution(solver, tasks, machines, variables, times)
    
    return results

def validate_schedule(schedule: List[Dict], 
                     input_data: pd.DataFrame, 
                     machine_columns: List[str],
                     status: int) -> Dict:
    """Validate the schedule against constraints."""
    task_ids = set(input_data["TaskID"].tolist())
    
    results = {}
    
    # 1. Check that all tasks are handled
    all_tasks_handled = all(task_id in [task["task_id"] for task in schedule] for task_id in task_ids)
    if not all_tasks_handled:
        missing_tasks = task_ids - {task["task_id"] for task in schedule}
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
    
    # 5. Check that the objective is optimized
    results["Optimal solution"] = (
        status == cp_model.OPTIMAL,
        "Optimal solution found" if status == cp_model.OPTIMAL 
        else "Feasible but not optimal solution found"
    )
    
    return results


def extract_solution(solver: cp_model.CpSolver, 
                    tasks: List[Dict], 
                    machines: List[int], 
                    variables: Dict, 
                    times: List[List[int]]) -> List[Dict]:
    """Extract the solution from the solver."""
    schedule = []
    
    for task_idx, task in enumerate(tasks):
        task_end_time = solver.Value(variables['task_end'][task_idx])
        tardiness = max(0, task_end_time - task['DueDate'])
        
        machine_times = [
            (machine_idx + 1,
             solver.Value(variables['start'][(task_idx, machine_idx)]),
             solver.Value(variables['start'][(task_idx, machine_idx)]) + 
             times[task_idx][machine_idx])
            for machine_idx in machines
        ]

        schedule.append({
            'task_id': task['TaskID'],
            'finish_time': task_end_time,
            'tardiness': tardiness,
            'machine_times': machine_times,
            'weight': task['Weight']
        })
    
    return schedule


def schedule_to_dataframe(schedule: List[Dict]) -> pd.DataFrame:
    """Convert schedule to DataFrame format for export."""
    return pd.DataFrame([
        {
            'TaskID': entry['task_id'],
            'Weight': entry['weight'],
            'Machine': machine_num,
            'Start': start,
            'Finish': end,
            'Tardiness': entry['tardiness'] if machine_num == entry['machine_times'][-1][0] else 0
        }
        for entry in schedule
        for machine_num, start, end in entry['machine_times']
    ])