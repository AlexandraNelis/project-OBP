import streamlit as st
import altair as alt
import pandas as pd
import io

def setup_streamlit_ui():
    """Configure Streamlit UI layout."""
    st.set_page_config(
        page_title="Multi-Machine Scheduling Optimizer",
        page_icon="🛠️",
        layout="wide"
    )


def show_upload_instructions():
    """Display file upload instructions."""
    st.markdown(
        """
        1. Upload an Excel file (.xlsx) with the following columns:
           - **TaskID** (unique identifier)
           - **ReleaseDate**
           - **DueDate**
           - **Weight**
           - **M1Time**, **M2Time**, etc.  \n
              *(processing times on each machine)*
        2. Configure detected machine columns below.
        3. Click **Solve Scheduling Problem** to optimize the schedule.
        """
    )
    st.info("Ensure your file follows the required format to avoid errors.")


def build_gantt_chart(df_gantt, selection):
    """Build and configure Altair chart object."""
    return alt.Chart(df_gantt).mark_bar().encode(
        x=alt.X('Start:Q', title='Start Time'),
        x2=alt.X2('Finish:Q'),
        y=alt.Y('Machine:N', sort='-x', title='Machine'),
        color=alt.Color(
            'Task:N',
            title='Task',
            sort=alt.EncodingSortField(field='TaskID', order='ascending'),
            scale=alt.Scale(scheme='turbo')
        ),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        tooltip=[
            'Task:N', 'Start:Q', 'Finish:Q', 'ReleaseDate:Q',
            'DueDate:Q', 'Tardiness:Q', 'Weight:Q'
        ]
    ).properties(
        title="Schedule Gantt Chart",
        width=800,
        height=600
    ).add_params(selection)


def create_gantt_chart(schedule, input_data):
    """Create Altair Gantt chart."""
    df_gantt = format_gantt_data(schedule, input_data)
    selection = alt.selection_point(fields=['Task'], bind='legend')
    
    return build_gantt_chart(df_gantt, selection)


def format_gantt_data(schedule, input_data):
    """Format schedule data for Gantt chart."""
    chart_data = []
    for entry in schedule:
        task_id = entry['task_id']
        task_info = input_data[input_data['TaskID'] == task_id].iloc[0]
        
        for machine_num, start, end in entry['machine_times']:
            chart_data.append({
                'Task': f"Task {task_id}",
                'Machine': f"M{machine_num}",
                'Start': start,
                'Finish': end,
                'Tardiness': entry['tardiness'],
                'ReleaseDate': task_info['ReleaseDate'],
                'DueDate': task_info['DueDate'],
                'Weight': entry['weight'],
                'TaskID': task_id,
            })
    
    return pd.DataFrame(chart_data)


def handle_solution_download(results_df, input_df):
    """Handle solution download functionality."""
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


def display_validation_results(validation_results):
    """Display schedule validation results."""
    for constraint, (is_satisfied, message) in validation_results.items():
        if is_satisfied:
            st.markdown(f"- **{constraint.replace('_', ' ').capitalize()}**: Satisfied ✅")
        else:
            st.markdown(f"- **{constraint.replace('_', ' ').capitalize()}**: Not satisfied ❌")
            st.text(f"    {message}")


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
        t_weight = entry['weight']
        for (m_num, s, f) in entry['machine_times']:
            rows.append({
                'TaskID': t_id,
                'Weight': t_weight,
                'Machine': m_num,
                'Start': s,
                'Finish': f,
                'Tardiness': t_tardiness if m_num == entry['machine_times'][-1][0] else 0
            })
    return pd.DataFrame(rows)


def setup_sidebar():
    """Configure and return sidebar elements."""
    with st.sidebar:
        st.title("Upload task data")
        show_upload_instructions()
        uploaded_file = st.file_uploader("Upload your data file:", type=["xlsx"])
        st.markdown("---")
        
        st.title("Select solver")
        solver_choice = st.selectbox(
            "Select Solver:",
            ["OR-Tools", "Gurobi"],
            help="Choose the solver to optimize the schedule."
        )
        st.markdown("---")
        
    return uploaded_file, solver_choice
