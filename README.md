# Social Media Sentiment Analysis Project

## Project Overview
This project focuses on sentiment analysis, specifically analyzing emotions or opinions from social media platforms related to specific events. The goal is to understand public sentiment changes over time using data gathered from Twitter.

## Technologies Used
- AWS EMR (Elastic MapReduce)
- Apache Spark
- Python

## Project Details
Initially, the project aimed to collect real-time Twitter data using the Twitter API. However, due to recent changes in Twitter's developer API (now referred to as X), free tier access no longer supports data pulling, making it a paid service. Consequently, an alternative approach was adopted.

### Alternative Approach
Instead of real-time data, a dataset from the 2016 US presidential election tweets was used. The dataset was in CSV format and contained tweets from that period. Prior to analysis, a preprocessing step was implemented to filter out non-textual elements like emoticons from the text data.

For sentiment analysis, a pre-trained model called EmoRoBERTa from Hugging Face was employed. This model was fine-tuned to detect emotions based on text input. The dataset was formatted to fit the input requirements of the EmoRoBERTa model.

### Data Processing and Analysis
AWS EMR with Apache Spark was utilized to create a data pipeline for processing the large dataset. Apache Spark provided the framework to distribute computation across multiple nodes, enabling efficient data processing.

### Results Visualization
The results of sentiment analysis were visualized using Matplotlib, a Python plotting library. Graphical representations were generated to illustrate changes in sentiment over the course of the analyzed dataset.

## Contributors and Roles

| Name               | Tasks                                                                                               |
|--------------------|-----------------------------------------------------------------------------------------------------|
| Jakub Cisoń        | Group management, documentation, data filtering from CSV tweets dataset, SparkSession setup         |
| Michał Zarzycki    | Found pre-trained language model for emotion detection, basic data visualization, EMR configuration |
| Jakub Ebertowski   | Created AWS pipeline for data processing                                                            |
| Maria Szcześniak   | Assisted in pipeline creation, improved visualization techniques (attempted)                        |
| Łukasz Czarzasty   |                                                                                                     |
