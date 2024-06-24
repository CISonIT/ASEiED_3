#import modules
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover
import csv
import matplotlib.pyplot as plt

# Mapowane słowa kluczowe na odpowiednie wartości numeryczne
emotion_mapping = {
    'adore': 0, 'great': 0, 'brilliant': 0, 'best': 0, 'fantastic': 0,
    'dislike': 1, 'hate': 1, 'rubbish': 1, 'terrible': 1,
    'bad': 2, 'worst': 2,
    'like': 3, 'good': 3,
    'love': 4, 'awesome': 4,
    'loathe': 5, 'awful': 5,
    'pity': 6, 'regret': 6,
    'proud': 7,
    'sad': 8, 'upset': 8
}


# Funkcja która zwraca odpowiednią cyfrę przypisaną do emocji
def determine_emotion(text):
    words = text.lower().split()  # Podział tekstu na słowa i zmień na małe litery

    for word in words:
        if word in emotion_mapping:
            return emotion_mapping[word]

    return 9  # Jeśli żadne z kluczowych słów nie zostało znalezione, zwróć 9 (neutralna emocja)


# Wczytanie pliku CSV, aktualizacja wartości kolumny 'Sentiment' i wypisanie wartości
input_file = 'dataset/tweets.csv'

with open(input_file, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    fieldnames = reader.fieldnames  # Oryginalne nagłówki

    rows = []
    for row in reader:
        sentiment_text = row['SentimentText']
        emotion_value = determine_emotion(sentiment_text)

        # Aktualizacja wartości kolumny 'Sentiment' na podstawie emocji
        row['Sentiment'] = str(emotion_value)  # Konwersja na string, jeśli to potrzebne
        rows.append(row)

        # Wypisanie wartości tekstu oraz numeru emocji
        print(f"SentimentText: {sentiment_text}")
        print(f"Sentiment: {emotion_value}")
        print("-" * 50)  # Linia oddzielająca wypisywane dane


# Zapisanie zaktualizowanych danych do tego samego pliku CSV (nadpisuje oryginalny plik)
with open(input_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Pomyślnie zaktualizowano kolumnę 'Sentiment' i zapisano do pliku {input_file}.")
##########################################################################################
# Tworzenie Spark session
appName = "Sentiment Analysis in Spark"
spark = SparkSession \
    .builder \
    .appName(appName) \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# Wczytanie plik csv do dataFrame z automatycznie wywnioskowanym schematem
tweets_csv = spark.read.csv('dataset/tweets.csv', inferSchema=True, header=True)
tweets_csv.show(truncate=False, n=3)

# Wybieram kolumny "SentimentText" i "Sentiment",przekształcam dane "Sentiment" na liczbę całkowitą
data = tweets_csv.select("SentimentText", col("Sentiment").cast("Int").alias("label"))
data.show(truncate=False, n=5)

# Dzielenie danych na 70% danych treningowych, 30% testowych
dividedData = data.randomSplit([0.7, 0.3])
trainingData = dividedData[0] #index 0 = data training
testingData = dividedData[1] #index 1 = data testing
train_rows = trainingData.count()
test_rows = testingData.count()
print("Training data rows:", train_rows, "; Testing data rows:", test_rows)

# Tokenization
tokenizer = Tokenizer(inputCol="SentimentText", outputCol="SentimentWords")
tokenizedTrain = tokenizer.transform(trainingData)
tokenizedTrain.show(truncate=False, n=5)

# Remove stopwords
swr = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="MeaningfulWords")
SwRemovedTrain = swr.transform(tokenizedTrain)
SwRemovedTrain.show(truncate=False, n=5)

# Hashing TF
hashTF = HashingTF(inputCol=swr.getOutputCol(), outputCol="features")
numericTrainData = hashTF.transform(SwRemovedTrain).select('label', 'MeaningfulWords', 'features')
numericTrainData.show(truncate=False, n=3)

# Train logistic regression model
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10, regParam=0.01)
model = lr.fit(numericTrainData)
print("Training is done!")

# Process test data
tokenizedTest = tokenizer.transform(testingData)
SwRemovedTest = swr.transform(tokenizedTest)
numericTest = hashTF.transform(SwRemovedTest).select('label', 'MeaningfulWords', 'features')
numericTest.show(truncate=False, n=2)

# Predict and evaluate
prediction = model.transform(numericTest)
predictionFinal = prediction.select("MeaningfulWords", "prediction", "label")

# Assigning emotion labels
predictionFinal = predictionFinal.withColumn("predicted_emotion_label",
                                             when(col("prediction") == 0, "admiration") \
                                             .when(col("prediction") == 1, "anger") \
                                             .when(col("prediction") == 2, "negative") \
                                             .when(col("prediction") == 3, "positive") \
                                             .when(col("prediction") == 4, "love") \
                                             .when(col("prediction") == 5, "disgust") \
                                             .when(col("prediction") == 6, "disappointment") \
                                             .when(col("prediction") == 7, "pride") \
                                             .when(col("prediction") == 8, "sadness") \
                                             .when(col("prediction") == 9, "neutral"))

predictionFinal.show(n=400, truncate=False)

emotion_counts = predictionFinal.groupBy("predicted_emotion_label").count().collect()
emotions = [row["predicted_emotion_label"] for row in emotion_counts]
counts = [row["count"] for row in emotion_counts]

# Rysowanie wykresu słupkowego
plt.figure(figsize=(10, 6))
plt.bar(emotions, counts, color='blue')
plt.xlabel('Emotion')
plt.ylabel('Counts')
plt.title('Counts of Predicted Emotions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


correctPrediction = predictionFinal.filter(predictionFinal['prediction'] == predictionFinal['label']).count()
totalData = predictionFinal.count()
accuracy = correctPrediction / totalData
print("correct prediction:", correctPrediction, ", total data:", totalData, ", accuracy:", accuracy)


