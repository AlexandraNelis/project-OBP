import streamlit as st
import pandas as pd
from ortools.sat.python.cp_model import OPTIMAL, FEASIBLE
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
    if status in [OPTIMAL, FEASIBLE, GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
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

        # Validation
        with st.expander("Validation Results"):
            validation_results = validate_schedule(
                results["schedule"], df, machine_columns, solver_choice, status
            )
            display_validation_results(validation_results)

        # Download
        st.markdown("### Download Solution")
        handle_solution_download(results_df, df)
    else:
        st.error("No feasible solution found. Please check your input data.")


def main():
    setup_streamlit_ui()
    uploaded_file, solver_choice = setup_sidebar()

    # Initialize the 'is_optimizing' flag if not present
    if "is_optimizing" not in st.session_state:
        st.session_state["is_optimizing"] = False

    st.title("Multi-Machine Scheduling Optimizer")
    st.markdown(
        """
        Optimize your multi-machine scheduling tasks to minimize total **weighted tardiness**.
        Use the **sidebar** to upload data and configure settings.
        """
    )

    # 1) Load or restore DataFrame
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            df.columns = df.columns.map(str)
            # If the user uploads a new dataset, invalidate old Gurobi model or results
            if "df" in st.session_state and not df.equals(st.session_state["df"]):
                st.session_state["gurobi_model"] = None
                st.session_state["results"] = None

            st.session_state["df"] = df
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            return
    else:
        df = st.session_state.get("df", None)

    if df is None:
        st.info("Upload an Excel file to start.")
        return

    # 2) If solver changed from a previous run, also reset the old model
    if "last_solver" in st.session_state and st.session_state["last_solver"] != solver_choice:
        st.session_state["gurobi_model"] = None
        st.session_state["results"] = None
    st.session_state["last_solver"] = solver_choice

    # 3) Validate input data
    is_valid, error_message = validate_input_data(df)
    if not is_valid:
        st.error(error_message)
        return

    st.markdown("### Input Data Preview")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # 4) Identify machine columns
    with st.sidebar:
        st.title("Configure Machine Columns")
        machine_columns = st.multiselect(
            "Select Machine Columns (in order):",
            identify_machine_columns(df),
            default=identify_machine_columns(df)
        )
        st.markdown("---")

    # 5) Solve button (only enabled if not currently optimizing)
    if not st.session_state["is_optimizing"]:
        if st.button("Solve Scheduling Problem"):
            # Start optimizing
            st.session_state["is_optimizing"] = True
            with st.spinner(f"Solving the scheduling problem with {solver_choice}..."):
                if solver_choice == "Gurobi":
                    (
                        results,
                        gurobi_model,
                        tasks,
                        machines,
                        x,
                        T,
                        times
                    ) = solve_scheduling_problem_gurobi(df, machine_columns)

                    st.session_state["results"] = results
                    st.session_state["gurobi_model"] = gurobi_model
                    st.session_state["gurobi_data"] = (tasks, machines, x, T, times)

                else:
                    results = solve_scheduling_problem(df, machine_columns)
                    st.session_state["results"] = results
                    st.session_state["gurobi_model"] = None

            # Finished optimizing
            st.session_state["is_optimizing"] = False
    else:
        st.info("Optimization process interrupted. Please restart the whole program.")

    # 6) If Gurobi is used and we have a partial model, allow "Continue" if feasible but not proven optimal
    if solver_choice == "Gurobi" and st.session_state.get("gurobi_model") is not None:
        current_results = st.session_state.get("results", {})
        status = current_results.get("status")
        # Show "Continue" only if the solver is not optimizing *and*
        # the status is SUBOPTIMAL/TIME_LIMIT/FEASIBLE (i.e., not proven optimal yet)
        if status in [GRB.SUBOPTIMAL, GRB.TIME_LIMIT, FEASIBLE] and status not in [GRB.OPTIMAL] and not st.session_state["is_optimizing"]:
            if st.button("Continue Solving for 5 More Minutes"):
                if st.session_state["is_optimizing"]:
                    st.error("Optimization process interrupted. Please restart the whole program.")
                else:
                    st.session_state["is_optimizing"] = True
                    gurobi_model = st.session_state.get("gurobi_model", None)
                    if gurobi_model is not None:
                        tasks, machines, x, T, times = st.session_state["gurobi_data"]
                        with st.spinner("Continuing optimization... please wait"):
                            (
                                new_results,
                                updated_model,
                                tasks,
                                machines,
                                x,
                                T,
                                times
                            ) = solve_scheduling_problem_gurobi(
                                df,
                                machine_columns,
                                initial_model=gurobi_model,
                                tasks=tasks,
                                machines=machines,
                                x=x,
                                T=T,
                                times=times
                            )
                        st.session_state["results"] = new_results
                        st.session_state["gurobi_model"] = updated_model

                        if new_results["status"] == GRB.OPTIMAL:
                            st.success(f"Optimal solution found! Weighted Tardiness = {new_results['objective']:.1f}")
                            st.session_state["gurobi_model"] = None
                        else:
                            st.success("Continued solving.")

                st.session_state["is_optimizing"] = False
        elif status in [GRB.OPTIMAL]:
            st.info("Already at optimal solution.")
        # If we are optimizing, or the solver is at an infeasible state, no button is shown

    # 7) Display whatever results we have
    show_current_results(df, solver_choice, machine_columns)


if __name__ == "__main__":
    main()
