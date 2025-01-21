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
    solve_scheduling_problem_gurobi
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


def main():
    setup_streamlit_ui()
    uploaded_file, solver_choice = setup_sidebar()

    st.title("Multi-Machine Scheduling Optimizer")
    st.markdown(
        """
        Optimize your multi-machine scheduling tasks to minimize total **weighted tardiness**.  
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

        # Identify machine columns
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

            # Check solver status
            if results["status"] in [OPTIMAL, FEASIBLE, GRB.OPTIMAL, GRB.SUBOPTIMAL]:
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
