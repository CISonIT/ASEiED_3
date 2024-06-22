# Sentiment Analysis on Social Media

## Objective
The objective of this project is to perform sentiment analysis on selected social media platforms, with a focus on Twitter. This analysis aims to understand public sentiment regarding specific events and observe changes in these sentiments over time.

## Data
The data used in this project is sourced from Twitter. Gathering this data requires the use of the Twitter API, which allows for the collection of tweets related to specific events.

## Technologies
- AWS EMR (Elastic MapReduce)
- Apache Spark
- Python

## Requirements
1. **Data Collection**: Gather data from Twitter for the period of the selected event using the Twitter API.
2. **Sentiment Analysis**: Analyze changes in public sentiment over time.
3. **Visualization**: Present the results in a graphical format.

## Project Steps

### 1. Setting Up the Environment
- Configure AWS EMR for scalable data processing.
- Set up Apache Spark on AWS EMR for distributed data processing.
- Prepare a Python environment with necessary libraries for data collection and sentiment analysis.

### 2. Data Collection
- Use the Twitter API to collect tweets related to the chosen event.
- Store the collected data in a suitable format for processing (e.g., JSON, CSV).

### 3. Data Processing and Sentiment Analysis
- Load the collected data into Apache Spark.
- Clean and preprocess the data (remove duplicates, handle missing values, etc.).
- Perform sentiment analysis using a suitable Python library (e.g., TextBlob, VADER, or a pre-trained model from Hugging Face's Transformers).
- Analyze changes in sentiment over time.

### 4. Visualization
- Use a Python visualization library (e.g., Matplotlib, Seaborn, Plotly) to create graphical representations of the sentiment analysis results.
- Ensure that the visualizations effectively communicate the changes in public sentiment over the selected period.
