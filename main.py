import streamlit as st
import pandas as pd
from ortools.sat.python.cp_model import OPTIMAL, FEASIBLE
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode
from gurobipy import GRB

# Import from backend
from backend import (
    validate_input_data,
    validate_schedule,
    identify_machine_columns,
    solve_scheduling_problem,
    solve_scheduling_problem_gurobi,
)

# Import from frontend
from frontend import (
    setup_streamlit_ui,
    setup_sidebar,
    create_gantt_chart,
    handle_solution_download,
    display_validation_results,
    schedule_to_dataframe
)


def show_current_results(df, solver_choice, machine_columns):
    """
    Display the stored results (if any) from st.session_state.
    Runs on every script execution, so results remain
    visible after "Continue Solving" clicks.
    """
    results = st.session_state.get("results", None)
    if not results:
        return  # Nothing to show

    status = results["status"]
    # Define feasible statuses based on solver
    feasible_statuses = []
    if solver_choice == "OR-Tools":
        feasible_statuses = [FEASIBLE, OPTIMAL]
    elif solver_choice == "Gurobi":
        feasible_statuses = [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]

    # If a solution is feasible/optimal/suboptimal/time-limit, display
    if status in feasible_statuses:
        if results["objective"] is not None:
            st.success(f"Solution found! Total Weighted Tardiness = {results['objective']:.1f}")
        else:
            st.warning("Solution found, but the objective value is not available.")

        # Gantt Chart
        with st.expander("Gantt Chart", expanded=True):
            fig_gantt = create_gantt_chart(results["schedule"], df)
            st.altair_chart(fig_gantt, use_container_width=True)

        # Detailed Schedule
        with st.expander("Detailed Schedule"):
            results_df = schedule_to_dataframe(results["schedule"])
            st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Validation Results
        with st.expander("Validation Results"):
            validation_results = validate_schedule(
                results["schedule"], df, machine_columns, solver_choice, status
            )
            display_validation_results(validation_results)

        # Download Button
        st.markdown("### Download Solution")
        handle_solution_download(results_df, df)
    else:
        st.error("No feasible solution found. Please check your input data.")


def main():
    setup_streamlit_ui()
    uploaded_file, solver_choice = setup_sidebar()

    # Initialize 'is_optimizing' to avoid multiple solves at once
    if "is_optimizing" not in st.session_state:
        st.session_state["is_optimizing"] = False

    st.title("Multi-Machine Scheduling Optimizer")
    st.markdown("""
        Optimize your multi-machine scheduling tasks to minimize total **weighted tardiness**.  
        Use the **sidebar** to upload data **or** click below to **manually input data**.
    """)

    # Toggle between Manual Input vs. File Upload
    if "manual_mode" not in st.session_state:
        st.session_state["manual_mode"] = False  # default to file upload

    # Create two buttons, one for each mode
    col1, col2 = st.columns([2, 2])

    switch1 = col1.button("Manual Data Input")
    switch2 = col2.button("Upload Data Input")

    with col1:
        if switch1:
            st.session_state["manual_mode"] = True
            # Clear old results and model so they don't show up
            st.session_state["results"] = None
            st.session_state["gurobi_model"] = None
            st.session_state["is_optimizing"] = False

    with col2:
        if switch2:
            st.session_state["manual_mode"] = False
            # Clear old results and model so they don't show up
            st.session_state["results"] = None
            st.session_state["gurobi_model"] = None
            st.session_state["is_optimizing"] = False

    # If user changes solver from a previous run, reset model and results
    if "last_solver" in st.session_state and st.session_state["last_solver"] != solver_choice:
        st.session_state["gurobi_model"] = None
        st.session_state["results"] = None
    st.session_state["last_solver"] = solver_choice

    # -------------------- MANUAL MODE --------------------
    if st.session_state["manual_mode"]:
        st.subheader("Manual Data Input")

        # Initialize a DataFrame in session_state if not present
        if "manual_df" not in st.session_state:
            # Start with default columns and one row
            df_init = pd.DataFrame({
                "TaskID": [1],
                "ReleaseDate": [0],
                "DueDate": [0],
                "Weight": [0],
                "M1Time": [0],
                "M2Time": [0],
                "M3Time": [0]
            })
            st.session_state["manual_df"] = df_init

        # Buttons to add/delete rows or machine columns
        #Don’t allow deleting if there is only 1 row or only 1 machine column remaining.
		#Use your existing identify_machine_columns function to find the last machine column name.
		#Drop that column from st.session_state["manual_df"] only if at least 2 machine columns remain.
		#Similarly, only drop the last row if there are at least 2 rows.

        row_col1, row_col2 = st.columns(2)

        with row_col1:
            if st.button("Add 1 More Row"):
                current_df = st.session_state["manual_df"]
                if "TaskID" in current_df.columns:
                    max_id = current_df["TaskID"].max()
                    if pd.isna(max_id):
                        max_id = 0
                    new_id = max_id + 1
                else:
                    new_id = 1  # fallback

                new_row = {
                    "TaskID": new_id,
                    "ReleaseDate": 0,
                    "DueDate": 0,
                    "Weight": 0,
                    "M1Time": 0,
                    "M2Time": 0,
                    "M3Time": 0
                }
                st.session_state["manual_df"] = pd.concat(
                    [current_df, pd.DataFrame([new_row])],
                    ignore_index=True
                )

            # Delete last row only if there's more than 1 row
            if st.button("Delete Last Row"):
                if len(st.session_state["manual_df"]) > 1:
                    st.session_state["manual_df"].drop(
                        st.session_state["manual_df"].index[-1],
                        inplace=True
                    )
                else:
                    st.warning("Cannot delete the last row (at least 1 row is required).")

        with row_col2:
            if st.button("Add 1 More Machine Column"):
                # Count how many columns already start with 'M' and end with 'Time'
                machine_cols = identify_machine_columns(st.session_state["manual_df"])
                next_machine_index = len(machine_cols) + 1
                new_col = f"M{next_machine_index}Time"
                st.session_state["manual_df"][new_col] = 0  # Initialize with default value

            # Delete last machine column only if there's more than 1 machine column
            if st.button("Delete Last Machine Column"):
                machine_cols = identify_machine_columns(st.session_state["manual_df"])
                if len(machine_cols) > 1:
                    last_machine_col = machine_cols[-1]  # e.g. 'M3Time'
                    st.session_state["manual_df"].drop(columns=[last_machine_col], inplace=True)
                else:
                    st.warning("Cannot delete the last machine (at least 1 machine column is required).")

        gb = GridOptionsBuilder.from_dataframe(st.session_state["manual_df"])
        gb.configure_default_column(editable=True, groupable=True)
        gb.configure_grid_options(stopEditingWhenCellsLoseFocus=True)
        gb.configure_column("TaskID", editable=False)
        gb_options = gb.build()

        st.info("Edit your data below.")

        # Create a dynamic key to force AgGrid to refresh properly
        grid_key = f"manual_aggrid_{len(st.session_state['manual_df'])}_{len(st.session_state['manual_df'].columns)}"

        # Display data in AgGrid
        aggrid_return = AgGrid(
            st.session_state["manual_df"],
            key=grid_key,
            gridOptions=gb_options,
            data_return_mode=DataReturnMode.AS_INPUT,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            fit_columns_on_grid_load=True,
            theme="balham",
            enable_enterprise_modules=False
        )

        # Update session_state with edited data
        updated_df = pd.DataFrame(aggrid_return["data"])
        st.session_state["manual_df"] = updated_df

        # Solve scheduling problem
        if not st.session_state["is_optimizing"]:
            if st.button("Solve Scheduling Problem (Manual)"):
                st.session_state["is_optimizing"] = True

                # 1) Copy manual data into main 'df'
                st.session_state["df"] = st.session_state["manual_df"].copy()
                df = st.session_state["df"]

                # Validate input data
                is_valid, error_message = validate_input_data(df)
                if not is_valid:
                    st.error(error_message)
                    st.session_state["is_optimizing"] = False
                    return

                # Show a preview
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

                with st.spinner(f"Solving the scheduling problem with {solver_choice}..."):
                    if solver_choice == "OR-Tools":
                        results = solve_scheduling_problem(df, machine_columns)
                    else:
                        results, model, tasks, machines, x, T, times = solve_scheduling_problem_gurobi(df, machine_columns)

                st.session_state["results"] = results
                st.session_state["is_optimizing"] = False

        else:
            st.info("Optimization process interrupted. Please restart the whole program.")

    # -------------------- FILE UPLOAD MODE --------------------
    else:
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                df.columns = df.columns.map(str)  # Ensure string columns
                # Store in session_state as well
                st.session_state["df"] = df
            except Exception as e:
                st.error(f"Error reading the file: {e}")
                return

            # Validate input data
            is_valid, error_message = validate_input_data(st.session_state["df"])
            if not is_valid:
                st.error(error_message)
                return

            st.markdown("### Input Data Preview")
            st.dataframe(st.session_state["df"], use_container_width=True, hide_index=True)

            # Configure machine columns
            with st.sidebar:
                st.title("Configure Machine Columns")
                machine_columns = st.multiselect(
                    "Select Machine Columns (in order):",
                    identify_machine_columns(st.session_state["df"]),
                    default=identify_machine_columns(st.session_state["df"])
                )
                st.markdown("---")

            if not st.session_state["is_optimizing"]:
                if st.button("Solve Scheduling Problem (File)"):
                    st.session_state["is_optimizing"] = True
                    with st.spinner(f"Solving the scheduling problem with {solver_choice}..."):
                        if solver_choice == "OR-Tools":
                            results = solve_scheduling_problem(st.session_state["df"], machine_columns)
                        else:
                            results, model, tasks, machines, x, T, times = solve_scheduling_problem_gurobi(df, machine_columns)
                            st.session_state["results"] = results  # Only store the `results` dictionary
                            st.session_state["gurobi_model"] = model
                            st.session_state["gurobi_data"] = (tasks, machines, x, T, times)

                    st.session_state["results"] = results
                    st.session_state["is_optimizing"] = False

        else:
            st.info("Upload an Excel file in the sidebar or switch to Manual Data Input.")

    # ---------- If Gurobi solution is feasible but not proven optimal, allow continuing ----------
    if solver_choice == "Gurobi" and st.session_state.get("gurobi_model") is not None:
        current_results = st.session_state.get("results", {})
        status = current_results.get("status")
        # We allow continuing if status is SUBOPTIMAL, TIME_LIMIT, or FEASIBLE
        if status in [GRB.SUBOPTIMAL, GRB.TIME_LIMIT, FEASIBLE] and status not in [OPTIMAL, GRB.OPTIMAL]:
            if not st.session_state["is_optimizing"]:
                if st.button("Continue Solving for 5 More Minutes"):
                    st.session_state["is_optimizing"] = True
                    gurobi_model = st.session_state["gurobi_model"]
                    tasks, machines, x, T, times = st.session_state["gurobi_data"]

                    # machine_cols might come from either manual or file
                    # If we are in manual mode but used Gurobi, we still have st.session_state["df"]
                    # so let's re-identify columns:
                    machine_columns = identify_machine_columns(st.session_state["df"])

                    with st.spinner("Continuing optimization..."):
                        results, model, tasks, machines, x, T, times = solve_scheduling_problem_gurobi(
                            df=st.session_state["df"],
                            machine_columns=machine_columns,
                            initial_model=gurobi_model,   # <--- pass existing model
                            tasks=tasks,
                            machines=machines,
                            x=x,
                            T=T,
                            times=times
                        )

                    st.session_state["is_optimizing"] = False
                    st.session_state["results"] = results
                    st.session_state["gurobi_model"] = model
                    st.session_state["gurobi_data"] = (tasks, machines, x, T, times)

                    if results["status"] == GRB.OPTIMAL:
                        st.balloons()
                        st.success(f"Optimal solution found! Weighted Tardiness = {results['objective']:.1f}")
                    else:
                        st.success("Continued solving. Check the updated results.")
            else:
                st.info("Optimization process interrupted. Please restart the whole program.")

    # Finally, display results if any
    if "df" in st.session_state and "results" in st.session_state:
        show_current_results(
            st.session_state["df"],
            solver_choice,
            identify_machine_columns(st.session_state["df"])
        )


# Run the app
if __name__ == "__main__":
    main()