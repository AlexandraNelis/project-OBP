import altair as alt
import pandas as pd
import io
import streamlit as st
from typing import List, Dict
from ortools.sat.python import cp_model
from backend import solve_scheduling_problem, schedule_to_dataframe, validate_schedule


def create_gantt_chart(schedule: List[Dict], input_data: pd.DataFrame) -> alt.Chart:
    """Create an interactive Gantt chart visualization."""
    chart_data = []
    
    for entry in schedule:
        task_id = entry['task_id']
        task_data = input_data.loc[input_data['TaskID'] == task_id].iloc[0]
        
        for machine_num, start, end in entry['machine_times']:
            chart_data.append({
                'Task': f"Task {task_id}",
                'Machine': f"M{machine_num}",
                'Start': start,
                'Finish': end,
                'Tardiness': entry['tardiness'],
                'ReleaseDate': task_data['ReleaseDate'],
                'DueDate': task_data['DueDate'],
                'Weight': entry['weight'],
                'TaskID': task_id,
            })
    
    df_gantt = pd.DataFrame(chart_data)
    selection = alt.selection_point(fields=['Task'], bind='legend')

    return alt.Chart(df_gantt).mark_bar().encode(
        x=alt.X('Start:Q', title='Start Time'),
        x2=alt.X2('Finish:Q'),
        y=alt.Y('Machine:N', sort='-x', title='Machine'),
        color=alt.Color(
            'Task:N',
            title='Task',
            sort=alt.EncodingSortField(field='TaskID', order='ascending')
        ),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        tooltip=[
            'Task:N',
            'Start:Q',
            'Finish:Q',
            'ReleaseDate:Q',
            'DueDate:Q',
            'Tardiness:Q',
            'Weight:Q'
        ]
    ).properties(
        title="Schedule Gantt Chart",
        width=800,
        height=400
    ).add_params(selection)


def display_empty_cells(df: pd.DataFrame) -> None:
    """Display rows with empty cells."""
    empty_cells = df[df.isnull().any(axis=1)]
    styled_df = empty_cells.style.applymap(
        lambda x: 'background-color: rgba(255, 0, 0, 0.6)' if pd.isnull(x) else ''
    )
    
    st.error("File contains empty cells. Please fill in missing values.")
    st.markdown("### Rows with Missing Values")
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


def setup_machine_columns(df: pd.DataFrame) -> List[str]:
    """Setup and configure machine columns."""
    possible_machine_columns = [
        col for col in df.columns 
        if col.upper().startswith("M") and col.upper().endswith("TIME")
    ]
    machine_names = [f"Machine {col[1]}" for col in possible_machine_columns]
    
    with st.sidebar:
        st.markdown("### Configure Machine Columns")
        selected_machines = st.multiselect(
            "Select Machine Columns (in order):",
            machine_names,
            default=machine_names
        )
        st.markdown("---")
    
    return [f"M{name[-1]}Time" for name in selected_machines]


def process_scheduling_solution(df: pd.DataFrame, machine_columns: List[str]) -> None:
    """Process and display the scheduling solution."""
    with st.spinner("Solving the scheduling problem..."):
        results = solve_scheduling_problem(df, machine_columns)
        
    if results['status'] in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        st.success(f"Solution found! Total Weighted Tardiness = {results['objective']:.1f}")
        
        with st.expander("Gantt Chart", expanded=True):
            fig_gantt = create_gantt_chart(results['schedule'], df)
            st.altair_chart(fig_gantt, use_container_width=True)
        
        with st.expander("Detailed Schedule"):
            results_df = schedule_to_dataframe(results['schedule'])
            st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        with st.expander("Validation Results"):
            validation_results = validate_schedule(
                results['schedule'], df, machine_columns, results['status']
            )
            display_validation_results(validation_results)
        
        create_download_button(results_df, df)
    else:
        st.error("No feasible solution found. Please check your input data.")


def display_validation_results(validation_results: Dict) -> None:
    """Display validation results with appropriate formatting."""
    for constraint, (is_satisfied, message) in validation_results.items():
        constraint_name = constraint.replace('_', ' ').capitalize()
        if is_satisfied:
            st.markdown(f"- **{constraint_name}**: Satisfied ✅")
        else:
            st.markdown(f"- **{constraint_name}**: Not satisfied ❌")
            st.text(f"    {message}")


def create_download_button(results_df: pd.DataFrame, input_df: pd.DataFrame) -> None:
    """Create and display the download button for the solution."""
    st.markdown("### Download Solution")
    output_bytes = io.BytesIO()
    
    with pd.ExcelWriter(output_bytes, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name="Schedule")
        input_df.to_excel(writer, index=False, sheet_name="InputData")
    
    output_bytes.seek(0)
    st.download_button(
        label="Download Schedule as Excel",
        data=output_bytes,
        file_name="schedule_solution.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


def setup_sidebar() -> None:
    """Setup the sidebar with instructions."""
    with st.sidebar:
        st.title("Upload task data")
        st.markdown(
            """
            1. Upload an Excel file (.xlsx) with:
               - **TaskID** (unique identifier)
               - **ReleaseDate**
               - **DueDate**
               - **Weight**
               - **Machine1Time**, **Machine2Time**, etc.
            2. Configure detected machine columns below.
            3. Click **Solve Scheduling Problem** to optimize.
            """
        )
        st.info("Ensure correct file format to avoid errors.")


def setup_main_page() -> None:
    """Setup the main page with title and description."""
    st.title("Multi-Machine Scheduling Optimizer")
    st.markdown(
        """
        Optimize multi-machine scheduling tasks to minimize total 
        **weighted tardiness**.  
        Use the **sidebar** to upload data and configure settings.
        """
    )

def validate_columns(df: pd.DataFrame) -> bool:
    """Validate required columns in the input data."""
    required_columns = {"TaskID", "ReleaseDate", "DueDate", "Weight"}
    machine_columns = {col for col in df.columns 
                      if col.upper().startswith("M") and col.upper().endswith("TIME")}
    
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return False
    
    unexpected_columns = set(df.columns) - required_columns - machine_columns
    if unexpected_columns:
        st.error(f"Unexpected columns found: {', '.join(unexpected_columns)}")
        return False
    
    return True

def load_and_validate_data(uploaded_file) -> pd.DataFrame:
    """Load and validate the uploaded data."""
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.map(str)
        
        if not validate_columns(df):
            return None
            
        if df.isnull().values.any():
            display_empty_cells(df)
            return None
            
        st.markdown("### Input Data Preview")
        st.dataframe(df, use_container_width=True, hide_index=True)
        return df
        
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None