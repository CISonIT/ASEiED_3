import pandas as pd
import json
import re

# Ścieżka do pliku CSV
file_path = '2016_US_election_tweets_100k.csv'

# Inicjalizujemy pustą listę na zdania
sentences = []

# Wczytujemy pierwsze 100 wierszy danych
chunk_size = 100
num_rows_read = 0

# Definicja regex do wykrywania emotikonów
emoji_pattern = re.compile(r"["
                           u"\U0001F600-\U0001F64F"  # emotikony emotki
                           u"\U0001F300-\U0001F5FF"  # symbole i piktogramy
                           u"\U0001F680-\U0001F6FF"  # symbole transportu i techniki
                           u"\U0001F1E0-\U0001F1FF"  # flagi (iOS)
                           u"\U00002500-\U00002BEF"  # chińskie znaki ideograficzne
                           u"\U00002702-\U000027B0"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"  # symboly uzupełniające
                           u"\u3030"
                           "]+", flags=re.UNICODE)

# Definicja regex do wykrywania słów rozpoczynających się od '@' lub '#'
mention_hashtag_pattern = re.compile(r'[@#]\w+')

# Definicja regex do wykrywania słów rozpoczynających się od 'http'
http_pattern = re.compile(r'\bhttp\S+', flags=re.IGNORECASE)

for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    # Zliczamy wczytane wiersze
    num_rows_read += len(chunk)

    # Filtrujemy dane w bieżącej partii
    filtered_chunk = chunk[(chunk['lang'] == 'en') &
                           (chunk['tweet_text'].notna()) &
                           (chunk['tweet_text'] != '')]

    # Przetwarzamy każdy tweet w bieżącej partii
    for tweet in filtered_chunk['tweet_text']:
        # Usuwamy emotikony
        cleaned_tweet = re.sub(emoji_pattern, '', tweet)

        # Usuwamy słowa zaczynające się od '@' lub '#'
        cleaned_tweet = re.sub(mention_hashtag_pattern, '', cleaned_tweet)

        # Usuwamy całe słowa zaczynające się od 'http'
        cleaned_tweet = re.sub(http_pattern, '', cleaned_tweet)

        # Usuwamy nadmiarowe białe znaki
        cleaned_tweet = cleaned_tweet.strip()

        # Dodajemy oczyszczony tweet do listy sentences
        if cleaned_tweet:
            sentences.append(cleaned_tweet)


# Tworzymy słownik do zapisu w formacie JSON
data = {
    "sentences": sentences
}

# Zapisujemy słownik do pliku JSON
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("Zapisano dane do data.json.")
