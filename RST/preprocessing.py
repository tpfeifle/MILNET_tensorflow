import pandas as pd
import json

# NUM_SAMPLES = 1000
ds_path = "../data/gourmet.json"
# number of files to split data set into --> one file per process to run (has to be parallelized)
nb_files = 40
# output path
rev_texts_path = '../data/'


def preprocess():
    """Preprocess Amazon Product data before running the RST parser on it

    This filters out review texts that have no content and maps overall rating to sentiment values.
    It splits the output into multiple txt files (nb_files) so that they can be fed into RST parsers running in
    parallel
    """

    # Load Data from json-file to list
    raw_data = []
    with open(ds_path) as f:
        for line in f:
            raw_data.append(json.loads(line))
    print(len(raw_data))

    # convert data from list to pandas dataframe
    df = pd.DataFrame(raw_data)

    # filter all review texts that have more then 30 characters
    df = df[df["reviewText"].str.len() >= 30]

    # convert overall rating to sentiment
    df.insert(3, "sentiment", df["overall"].replace({5.0: 1, 4.0: 1, 3.0: 0, 2.0: -1, 1.0: -1}), allow_duplicates=True)

    # compute minimum number of occurences of all sentiments
    sent_count_min = df["sentiment"].value_counts().min()
    df = df.groupby("sentiment").head(sent_count_min)

    # shuffle data (random_state for reproducibility)
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)

    print("Total reviews: {}".format(len(df)))
    print(df["overall"].value_counts())

    df.head()

    print("Creating .txt file that contains {} reviews: {}".format(rev_texts_path, len(df)))
    with open("../data/processed/gourmet.txt", "w") as f:
        for i, row in df.iterrows():
            f.write("###{}\n".format(row["overall"]))
            f.write(row["reviewText"] + "\n\n")

    print("Creating {} documents that contains {} reviews each: {}".format(nb_files, int(len(df) / nb_files),
                                                                           rev_texts_path))

    reviews_per_file = int(len(df) / nb_files)
    file_counter = 0
    reviews = ""
    review_counter = 0

    for i, row in df.iterrows():

        reviews += "###{}\n{}\n\n".format(row["overall"], row["reviewText"])
        review_counter += 1

        if review_counter == reviews_per_file:
            with open(rev_texts_path + str(file_counter + 1) + ".txt", "w") as f:
                f.write(reviews)

            reviews = ""
            file_counter += 1
            review_counter = 0

    with open(rev_texts_path + str(file_counter) + ".txt", "a") as f:
        f.write(reviews)


preprocess()
