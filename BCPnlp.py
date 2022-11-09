import requests
import json
import os
import pathlib

import pandas as pd
import re, string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# nltk.download()  # get select corpuses on first run

lemmatizer = WordNetLemmatizer()
sw = stopwords.words('english')


def fetch_data(page_num):
    endpoint = rf"https://haveyoursay.bcpcouncil.gov.uk/lcwip/maps/interactive-cycling-network-map/markers?page={page_num}"

    res = requests.get(endpoint).json()
    return res


def get_data_from_BCP():
    page_num = 1
    data = []

    response = fetch_data(page_num=page_num)
    data += response['markers']
    while len(response['markers']) != 0:
        print(f"Fetching page {page_num}")
        page_num+=1
        response = fetch_data(page_num=page_num)
        data += response['markers']

    with open(os.path.join(pathlib.Path(__file__).parent.resolve(), "output.json"), "w") as f:
        json.dump(data, f)


def clean_text(text):

    text = text.lower()
    text = re.sub('@', '', text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r"[^a-zA-Z ]+", "", text)

    text = nltk.word_tokenize(text)
    text = [w for w in text if w not in sw]
    return text


def lem(text):
    text = [lemmatizer.lemmatize(t) for t in text]
    text = [lemmatizer.lemmatize(t, 'v') for t in text]
    return text


if __name__ == "__main__":
    # get_data_from_BCP()

    # get data and format as dataframe
    with open(os.path.join(pathlib.Path(__file__).parent.resolve(), "output.json"), "r") as f:
        data = json.loads(f.read())
    comments = [d['survey']['marker_response'][0]['answer'] for d in data]
    survey_df = pd.DataFrame(columns=["comment"], data=comments)

    # clean and lemmatize
    survey_df['comment'] = survey_df['comment'].apply(lambda x: clean_text(x))
    survey_df['comment'] = survey_df['comment'].apply(lambda x: lem(x))

    # get all the words
    all_words = []
    for i in range(len(survey_df)):
        all_words = all_words + survey_df['comment'][i]

    # word frequency
    nlp_words = nltk.FreqDist(all_words)
    plot1 = nlp_words.plot(20, color='salmon', title='Word Frequency')

    # Bigrams
    bigrm = list(nltk.bigrams(all_words))
    words_2 = nltk.FreqDist(bigrm)
    words_2.plot(20, color='salmon', title='Bigram Frequency')

    # trigram
    trigram = list(nltk.trigrams(all_words))
    words_3 = nltk.FreqDist(trigram)
    words_3.plot(20, color='salmon', title='Trigram Frequency')

    # Get sentiment from comments
    survey_df['comment'] = [str(thing) for thing in survey_df['comment']]
    sentiment = []
    for i in range(len(survey_df)):
        blob = TextBlob(survey_df['comment'][i])
        for sentence in blob.sentences:
            sentiment.append(sentence.sentiment.polarity)
    survey_df['sentiment'] = sentiment
    survey_df['sentiment'].plot.hist(color='salmon', title='Comments Polarity')

    input("Enter to exit")
