import sys
import duckdb
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy import stats
import io
import base64
import numpy as np

def solve():
    """
    This function connects to a remote Parquet dataset using DuckDB,
    performs data analysis to answer three specific questions,
    and prints the answers in a JSON format.
    """
    try:
        # Initialize DuckDB connection with necessary extensions
        con = duckdb.connect(database=':memory:', read_only=False)
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("INSTALL parquet; LOAD parquet;")

        # --- Question 1: Which high court disposed the most cases from 2019 - 2022? ---
        
        # This query counts cases per court for the specified years,
        # orders them, and picks the top one. The court identifier from the
        # S3 path partitioning is used as the high court name.
        query1 = """
        SELECT
            court,
            COUNT(*) as case_count
        FROM
            read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1', hive_partitioning=1)
        WHERE
            year BETWEEN 2019 AND 2022
        GROUP BY
            court
        ORDER BY
            case_count DESC
        LIMIT 1;
        """
        result1 = con.execute(query1).fetchone()
        most_active_court = result1[0] if result1 else "Not found"

        # --- Question 2 & 3: Regression analysis and plot for court='33_10' ---
        
        # This query fetches data for the specific court ('33_10'), calculates the delay
        # in days between registration and decision. It filters for records with valid
        # dates and where the decision date is on or after the registration date.
        query2 = """
        WITH date_data AS (
            SELECT
                year,
                CAST(decision_date AS DATE) AS decision_dt,
                TRY_CAST(STRPTIME(date_of_registration, '%d-%m-%Y') AS DATE) AS registration_dt
            FROM
                read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=33_10/bench=*/metadata.parquet?s3_region=ap-south-1', hive_partitioning=1)
        )
        SELECT
            year,
            (decision_dt - registration_dt) AS delay_days
        FROM
            date_data
        WHERE
            registration_dt IS NOT NULL
            AND decision_dt IS NOT NULL
            AND (decision_dt - registration_dt) >= 0;
        """
        df = con.execute(query2).fetchdf()
        
        # Aggregate data: calculate mean delay per year
        if 'delay_days' in df.columns:
            df['delay_days'] = df['delay_days'].dt.days
        
        yearly_delay = df.groupby('year')['delay_days'].mean().reset_index()

        # Initialize variables for slope and plot
        regression_slope_answer = 0.0
        plot_uri = ""

        # Ensure we have at least two data points for a meaningful regression
        if len(yearly_delay) >= 2:
            x = yearly_delay['year']
            y = yearly_delay['delay_days']
            
            # --- Question 2: Calculate regression slope ---
            # Perform linear regression on (year, average_delay).
            # The average delay is calculated as (decision_date - registration_date).
            slope_positive_delay, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # The question specifically asks for the slope of (date_of_registration - decision_date),
            # which is the negative of the delay we calculated. Therefore, the slope will be inverted.
            regression_slope_answer = -slope_positive_delay

            # --- Question 3: Generate scatter plot with regression line ---
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.figure(figsize=(8, 5), dpi=80)
            
            # Scatter plot of the actual average delay data points
            plt.scatter(x, y, alpha=0.7, label='Avg Yearly Delay (Days)')
            
            # Regression line for the positive delay
            regression_line = slope_positive_delay * x + intercept
            plt.plot(x, regression_line, 'r--', linewidth=2, label='Regression Line')
            
            plt.title('Average Case Delay by Year for Court 33_10')
            plt.xlabel('Year')
            plt.ylabel('Average Delay (days)')
            plt.legend()
            plt.tight_layout()

            # Save plot to a memory buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            
            # Encode image to base64 and create data URI
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plot_uri = f"data:image/png;base64,{img_b64}"

        # --- Compile final answers into a dictionary ---
        answers = {
            "Which high court disposed the most cases from 2019 - 2022?": most_active_court,
            "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": regression_slope_answer,
            "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": plot_uri
        }

    except Exception as e:
        # Handle potential errors during execution and format as JSON
        answers = {
            "error": f"An error occurred: {str(e)}",
            "Which high court disposed the most cases from 2019 - 2022?": "Error processing data",
            "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": "Error processing data",
            "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": "Error processing data"
        }
    finally:
        # Ensure the database connection is closed
        if 'con' in locals() and con:
            con.close()

    # Print the final JSON output to stdout
    print(json.dumps(answers))

if __name__ == '__main__':
    solve()