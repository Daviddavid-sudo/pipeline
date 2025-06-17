import nltk
import string
import re
import inflect
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize


p = inflect.engine()
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def text_lowercase(text):
    return text.lower()


def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result


def convert_number(text):
    temp_str = text.split()
    new_string = []

    for word in temp_str:
        if word.isdigit():
            temp = p.number_to_words(word)
            new_string.append(temp)

        else:
            new_string.append(word)

    temp_str = ' '.join(new_string)
    return temp_str


def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def remove_whitespace(text):
    return  " ".join(text.split())


def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return filtered_text


def stem_words(text):
    word_tokens = word_tokenize(text)
    stems = [stemmer.stem(word) for word in word_tokens]
    return stems


def lemma_words(text):
    word_tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(word) for word in word_tokens]
    return lemmas


def read_csv(file):
    with open('IMDBDataset.csv') as f:
        review = csv.reader(f)
        data = [row for row in review]
    return data


def write_csv(readfile,stem):
    with open('Cleaned_lemma.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(pipeline(readfile,stem))


def pipeline(readfile,stem):
    data = read_csv(readfile)
    cleaned_data = []
    for review in data[1:]:
        review[0] = text_lowercase(review[0])
        review[0] = remove_numbers(review[0])
        review[0] = convert_number(review[0])
        review[0] = remove_punctuation(review[0])
        review[0] = remove_whitespace(review[0])
        review[0] = remove_stopwords(review[0])
        if stem:
            for i in range(len(review[0])):
                review[0][i] = stem_words(review[0][i])[0]
            cleaned_data.append([review[0],review[1]])
        else:
            for i in range(len(review[0])):
                review[0][i] = lemma_words(review[0][i])[0]
            cleaned_data.append([review[0],review[1]])
    return cleaned_data

write_csv('IMDBDataset.csv', False)