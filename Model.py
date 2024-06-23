import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class TwitterSentyment:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, from_tf=True)
        self.emotion = pipeline('sentiment-analysis', model=self.model, tokenizer=self.tokenizer)

    def analyze_sentiment(self, text):
        return self.emotion(text)

class DataManager:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data["sentences"]

    def process_data(self, sentences, sentiment_analyzer):
        results = []
        for sentence in sentences:
            try:
                analysis = sentiment_analyzer.analyze_sentiment(sentence)
                results.append((sentence, analysis))
            except UnicodeDecodeError as e:
                print(f"Skipping sentence due to encoding error: {e}")
        return results

    def print_results(self, results):
        for sentence, analysis in results:
            print(f"Sentence: {sentence}")
            for sentiment in analysis:
                label = sentiment['label']
                score = sentiment['score']
                print(f"Sentiment Analysis: Label - {label}, Score - {score:.4f}")
            print("-" * 80)

    def plot_results(self, results):
        labels = []

        for sentence, analysis in results:
            for sentiment in analysis:
                labels.append(sentiment['label'])

        label_counts = Counter(labels)

        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")
        bar_plot = sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()), palette="viridis")
        bar_plot.set_title('Frequency of Sentiment Labels')
        bar_plot.set_xlabel('Sentiment Label')
        bar_plot.set_ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.show()
