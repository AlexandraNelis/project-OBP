# Multi-Machine Scheduling Optimizer

## Overview

The Multi-Machine Scheduling Optimizer is a Python-based tool designed to solve an instance of the job-shop scheduling problem, for tasks that require processing on multiple machines, without taking into account their order. It minimizes total weighted tardiness while respecting constraints such as release dates and machine processing times. The tool is implemented using Streamlit for the user interface. Moreover, for the optimization two different approaches have been used, namely Google OR-Tools and Gurobi. When running the application one will have the option to choose for either Google OR-Tools or Gurobi as the solver.

## Features

* Upload task scheduling data (Excel format) by choosing one of the two options:
    - Manual Data Input
    - Upload Data Input
 
* Automatically detect machine-related columns.

* Configurable machine column selection for flexibility.

* Validate schedules against defined constraints.

* Minimize total weighted tardiness using constraint programming.

* Solve the scheduling problem and display:
    - Solution status
    - Gantt Chart
    - Detailed schedule
    - Validation results

* Choose the solver that you prefer:
    - Google OR-Tools
    - Gurobi
 
* Download the solution

## Installation

### Prerequisites

* Python 3.8+

* Recommended environment: Virtual environment or Conda

### Required Libraries

* Install the required libraries using the following command:

`pip install -r requirements.txt`

### Running the Application

* To run the Streamlit app, use the following command:

`streamlit run main.py`

## Testing performance

In addition to offering the option to choose between two solvers for schedule optimization, a series of tests have been conducted to evaluate the running time and difficulty of the solvers. These tests can be accessed by following the MasterTester branch, which contains the code for performance testing. And on top of that a user interface has been created for easy understanding and access.
