# Multi-Period Portfolio Optimization

A sophisticated portfolio optimization system that combines machine learning predictions with modern portfolio theory to create and maintain optimal investment portfolios.

## Features

- **Data Collection**: Automated fetching of historical stock data using Yahoo Finance API
- **Data Preprocessing**: Comprehensive technical indicator generation and data cleaning
- **Machine Learning**: Random Forest-based return prediction with time-series cross-validation
- **Portfolio Optimization**: Mean-variance optimization with sector constraints
- **Backtesting**: Realistic backtesting with transaction costs and periodic rebalancing
- **Performance Analysis**: Detailed performance metrics and visualizations

## Key Performance Metrics

- CAGR: ~20.57%
- Sharpe Ratio: 1.05
- Maximum Drawdown: -36.03%
- Calmar Ratio: 0.57

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/shaikhmubin02/multi-period-portfolio-optimization.git
    cd multi-period-portfolio-optimization
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows:
    ```