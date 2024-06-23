from Model import TwitterSentyment, DataManager

def main():
    model_path = "EmoRoBERTa"
    data_file_path = "data.json"

    # Inicjalizacja klasy TwitterSentyment
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

if __name__ == "__main__":
    main()
