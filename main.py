from Model import TwitterSentyment, DataManager
from DataMapper import process_tweets  # Importuj funkcję do filtracji danych
from pyspark.sql import SparkSession

def main():
    # Konfiguracja Apache Spark na AWS EMR

    # Utwórz sesję Spark
    spark = SparkSession.builder \
        .appName("Twitter Sentiment Analysis") \
        .getOrCreate()

    # Ścieżka do pliku CSV z tweetami
    file_path = '2016_US_election_tweets_100k.csv'

    # Inicjalizacja pustej listy na zdania
    sentences = []

    # Wczytujemy dane z pliku CSV i filtrujemy używając funkcji process_tweets
    chunk_size = 100
    num_rows_read = 0

    for chunk in spark.read.csv(file_path, header=True, inferSchema=True, chunksize=chunk_size):
        # Zliczamy wczytane wiersze
        num_rows_read += chunk.count()

        # Wywołujemy funkcję do filtracji danych
        filtered_tweets = process_tweets(chunk)

        # Dodajemy przefiltrowane tweety do listy sentences
        sentences.extend(filtered_tweets)

    # Zapisujemy przetworzone tweety do pliku JSON
    data = {
        "sentences": sentences
    }
    data_file_path = "data.json"
    with open(data_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Zapisano dane do {data_file_path}")

    # Inicjalizacja klasy TwitterSentyment
    model_path = "EmoRoBERTa"
    sentiment_analyzer = TwitterSentyment(model_path)

    # Inicjalizacja klasy DataManager
    data_manager = DataManager(data_file_path)

    # Pobranie danych z pliku data.json
    sentences = data_manager.load_data()

    # Przetwarzanie danych i analiza sentymentu
    results = data_manager.process_data(sentences, sentiment_analyzer)

    # Wyświetlenie wyników
    data_manager.print_results(results)

    # Prezentacja wyników w formie graficznej
    data_manager.plot_results(results)

    # Zakończ sesję Spark
    spark.stop()


if __name__ == "__main__":
    main()
