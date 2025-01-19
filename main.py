import streamlit as st
from frontend import setup_sidebar, setup_main_page, load_and_validate_data, setup_machine_columns, process_scheduling_solution


def main() -> None:
    """Main application function."""
    st.set_page_config(
        page_title="Multi-Machine Scheduling Optimizer",
        page_icon="🛠️",
        layout="wide"
    )
    
    setup_sidebar()
    setup_main_page()
    
    uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])
    if not uploaded_file:
        st.info("Upload an Excel file to start.")
        return
        
    df = load_and_validate_data(uploaded_file)
    if df is None:
        return
        
    machine_columns = setup_machine_columns(df)
    
    if st.button("Solve Scheduling Problem"):
        process_scheduling_solution(df, machine_columns)

if __name__ == "__main__":
    main()
