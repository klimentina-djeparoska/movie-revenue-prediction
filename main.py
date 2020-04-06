import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import string
import json
from sklearn.model_selection import train_test_split


def read_corpus(data_path):
    data_frame = pd.read_csv(data_path, names=column_names, skiprows=1, na_values="?", sep=",", skipinitialspace=True)
    data = data_frame.fillna(0).to_numpy()
    labels = np.array(pd.read_csv(data_path, names=column_names, skiprows=1, na_values="?", sep=",", skipinitialspace=True)["revenue"].get_values())

    return data, labels


def remove_unnecessary_columns(data_list, column_list, delete_indexes):

    dropped_columns = []

    # remove dropping columns
    for row in data_list:
        new_row = []
        for i in range(len(row)):
            if i not in delete_indexes:
                new_row.append(row[i])
        dropped_columns.append(new_row)

    cleaned_data = dropped_columns
    cleaned_columns = []
    for i in range(column_list.__len__()):
        if i not in delete_indexes:
            cleaned_columns.append(column_list[i])

    return cleaned_data, cleaned_columns


def remove_punctuation(list):
    return ''.join([x for x in list if x not in string.punctuation])


def preprocessing_data(data, columns, map_columns):
    # IMDB_ID - check for duplicates in data
    result = []
    for row in data:
        for i in range(len(row)):
            if i == map_columns['imdb_id']:
                result.append(row[i])

    r = set(result)
    if r.__len__() == data.__len__():
        print("NO DUPLICATES")
    else:
        print("DUPLICATES")

    # BELONGS_TO_COLLECTION R = {0, 1} belongs or not
    for row in data:
        for i in range(len(row)):
            if i == map_columns['belongs_to_collection']:
                if not row[i] == 0:
                    row[i] = 1

    # HOMEPAGE - mapping values to numbers 1 if it has otherwise 0
    for row in data:
        for i in range(len(row)):
            if i == map_columns['homepage']:
                if isinstance(row[i], str):
                    row[i] = 1
                else:
                    row[i] = 0

    # STATUS - mapping values to numbers
    for row in data:
        for i in range(len(row)):
            if i == map_columns['status']:
                if row[i] == "Released":
                    row[i] = 1
                else:
                    row[i] = 0

    # ORIGINAL_LANGUAGE - map to values 1 to n in binary
    original_languages = []
    for row in data:
        for i in range(len(row)):
            if i == map_columns['original_language']:
                original_languages.append(row[i])

    original_languages = list(set(original_languages))

    for row in data:
        for i in range(len(row)):
            if i == map_columns['original_language']:
                row[i] = int(bin(original_languages.index(row[i]))[2:])

    # PRODUCTION_COMPANIES - 5 intervals based  on company rank (range from 1 to 9996), 0 - NAN
    intervals = [[1, 51], [51, 101], [101, 1001], [1001, 10001], [10001, 100000]]
    binary_values = [0, 1, 10, 11, 100]
    for row in data:
        for i in range(len(row)):
            if i == map_columns['production_companies']:
                if not row[i] == 0:
                    temp = row[i].replace("\\", "")
                    temp = temp.replace("s\' ", "s")
                    temp = temp.replace("O\'C", "oC")
                    temp = temp.replace("l\'", "l")
                    temp = temp.replace("l, \'", "l\", \"")
                    temp = temp.replace("d\'A", "dA")
                    temp = temp.replace("w\'s", "ws")
                    temp = temp.replace("y \"Tor\"", "y Tor")
                    temp = temp.replace("L\'i", "L i")
                    temp = temp.replace("L\'A", "L A")
                    temp = temp.replace("I\'m", "I m")
                    temp = temp.replace("d\'I", "d I")
                    temp = temp.replace("d\'O", "d O")
                    temp = temp.replace("\"DIA\"", "DIA")
                    temp = temp.replace("t\'s", "t s")
                    temp = temp.replace("n\'t", "n t")
                    temp = temp.replace("\"Tsar\"", "Tsar")
                    temp = temp.replace("D\'A", "D A")
                    temp = temp.replace("n\' ", "n ")
                    temp = temp.replace("r\'s", "r s")
                    temp = temp.replace("n\'s", "n s")
                    temp = temp.replace("g\'s", "g s")
                    temp = temp.replace("e\'s", "e s")
                    temp = temp.replace("e\'r", "e r")
                    temp = temp.replace("O\' S", "O S")
                    temp = temp.replace("o\' B", "o B")
                    temp = temp.replace("N\' C", "N C")
                    temp = temp.replace("y\'s", "y s")
                    temp = temp.replace("d\'E", "d E")
                    temp = temp.replace("L\'I", "L I")
                    temp = temp.replace("t \'9", "t 9")
                    temp = temp.replace("c\'s", "c s")
                    temp = temp.replace("k\'s", "k s")
                    temp = temp.replace("\"Kvadrat\"", "Kvadrat")
                    temp = temp.replace("\'", "\"")
                    json_format = json.loads(temp)
                    value = json_format[0]['id']
                    index = 0
                    for inter in intervals:
                        if value in range(inter[0], inter[1]):
                            row[i] = binary_values[index]
                        index += 1

    # PRODUCTION_COUNTRIES - one hot on range values
    range_list = []
    for row in data:
        for i in range(len(row)):
            if i == map_columns['production_countries']:
                if not row[i] == 0:
                    temp = row[i].replace("D'I", "D I")
                    temp = temp.replace("\'", "\"")
                    json_format = json.loads(temp)
                    country = json_format[0]['iso_3166_1']
                    range_list.append(country)

    range_list = list(set(range_list))

    for row in data:
        for i in range(len(row)):
            if i == map_columns['production_countries']:
                if not row[i] == 0:
                    temp = row[i].replace("D'I", "D I")
                    temp = temp.replace("\'", "\"")
                    json_format = json.loads(temp)
                    country = json_format[0]['iso_3166_1']
                    row[i] = int(bin(range_list.index(country))[2:])

    # RELEASE_DATE - divide in intervals based on year range-years =(1921, 2017)
    intervals = [[1910, 1941], [1941, 1961], [1961, 1981], [1981, 2001], [2000, 2020]]
    binary_map = [0, 1, 10, 11, 100]

    for row in data:
        for i in range(len(row)):
            if i == map_columns['release_date']:
                if not row[i] == 0:
                    part_year = int(row[i].split("/").pop())
                    year = 0
                    if part_year > 17:
                        year = 1900 + part_year
                    else:
                        year = 2000 + part_year
                    index = 0
                    for inter in intervals:
                        if year in range(inter[0], inter[1]):
                            row[i] = binary_map[index]
                        index += 1

    # CREW - counting the number - the bigger the more money movie will make
    count = 0
    for row in data:
        count += 1
        for i in range(len(row)):
            if i == map_columns['crew']:
                if not row[i] == 0:
                    res = len(row[i].split("}")) - 1
                    row[i] = res

    # RUNTIME - length in time - divided in two intervals 90-120 and otherwise will be mapped to 1 and 0 accordingly
    for row in data:
        for i in range(len(row)):
            if i == map_columns['runtime']:
                if not row[i] == 0:
                    if int(row[i]) in range(90, 121):
                        row[i] = 1
                    else:
                        row[i] = 0

    delete_indexes = [5, 7, 8, 10, 15, 17, 18, 19, 20]
    data, columns = remove_unnecessary_columns(data, columns, delete_indexes)

    # GENRES 1- 20
    genres = ['Comedy', 'Drama', 'Family', 'Romance', 'Thriller', 'Action', 'Animation', 'Adventure', 'Horror',
              'Documentary', 'Music', 'Crime', 'Science Fiction', 'Mystery', 'Foreign', 'Fantasy', 'War', 'Western',
              'History', 'TV Movie']
    values = []
    value = []
    for row in data:
        for i in range(len(row)):
            if i == map_columns['genres']:
                if not row[i] == 0:
                    value = []
                    temp = row[i].replace("\'", "\"")
                    json_format = json.loads(temp)
                    for el in json_format:
                        value.append(el["name"])

        values.append(value)

    columns = np.concatenate((columns, genres))
    for i in range(len(data)):
        for j in range(0, 20):
            if genres[j] in values[i]:
                data[i].append(1)
            else:
                data[i].append(0)

    delete_indexes = [3]  # genres
    data, columns = remove_unnecessary_columns(data, columns, delete_indexes)

    print(data[0])

    return data, columns


if __name__ == "__main__":
    print("Hello there!")
    column_names = ['id', 'belongs_to_collection', 'budget', 'genres', 'homepage',
                    'imdb_id', 'original_language', 'original_title', 'overview',
                    'popularity', 'poster_path', 'production_companies', 'production_countries',
                    'release_date', 'runtime', 'spoken_languages', 'status', 'tagline',
                    'title', 'Keywords', 'cast', 'crew', 'revenue']
    map_columns = {}
    for i in range(column_names.__len__()):
        map_columns[column_names[i]] = i

    train_path = "./data/train.csv"
    train_data, train_labels = read_corpus(train_path)
    train_data, column_names = preprocessing_data(train_data, column_names, map_columns)

    # convert np-array to pd dataframe
    train_data = pd.DataFrame(data=train_data, columns=column_names, index=None)

    sns.pairplot(train_data[["release_date", "popularity", "status"]], diag_kind="kde")

    # MODEL
    train_data = train_data.to_numpy()

    train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.3, random_state=42)


    def build_model():
        model = keras.Sequential([
            layers.Dense(8, activation='relu', input_shape=[len(train_data[0])]),
            layers.Dense(8, activation='relu'),
            layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.RMSprop(0.00001)

        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
        return model


    model = build_model()
    model.summary()

    train_data.astype(float)
    test_data.astype(float)

    EPOCHS = 2000

    model = build_model()

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    early_history = model.fit(train_data, train_labels,
                              epochs=EPOCHS, validation_split=0.2, verbose=1)

    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Basic': early_history}, metric="mae")
    plt.ylim([0, 2000])
    plt.ylabel('MAE [revenue]')

    plotter.plot({'Basic': early_history}, metric="mse")
    plt.ylim([0, 2000])
    plt.ylabel('MSE [revenue^2]')

    loss, mae, mse = model.evaluate(test_data, test_labels, verbose=1)

    print("Testing set Mean Abs Error: {:5.2f} revenue".format(mae))

    test_predictions = model.predict(test_data).flatten()

    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [revenue]')
    plt.ylabel('Predictions [revenue]')
    lims = [0, 100000]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)

    error = test_predictions - test_labels
    plt.hist(error, bins = 100000)
    plt.xlabel("Prediction Error [revenue]")
    _ = plt.ylabel("Count")