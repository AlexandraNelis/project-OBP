# Multi-Machine Scheduling Optimizer

## Overview

The Multi-Machine Scheduling Optimizer is a Python-based tool designed to solve an instance of the job-shop scheduling problem, for tasks that require processing on multiple machines, without taking into account their order. It minimizes total weighted tardiness while respecting constraints such as release dates and machine processing times. The tool is implemented using Streamlit for the user interface and Google OR-Tools for optimization.

## Features

* Upload task scheduling data via an Excel file.

* Automatically detect machine-related columns.

* Configurable machine column selection for flexibility.

* Minimize total weighted tardiness using constraint programming.

* Generate and visualize Gantt charts for optimized schedules.

* Validate schedules against defined constraints:


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
