import pandas as pd

ds_path = "/Users/tim/Downloads/processed/train_test_validation V0.2/train/dataframe.csv"
nb_files = 1
rev_texts_path = '../data/organic/processed/'


def preprocess_organic():
    """Preprocess Organic data before running the RST parser on it

    This filters out review texts that have no content and maps overall rating to sentiment values.
    It splits the output into multiple txt files (nb_files) so that they can be fed into RST parsers running in
    parallel
    """

    # convert data from list to pandas dataframe
    header_columns = 'Author_ID|Author_name|Comment_number|Sentence_number|Domain_Relevance|Sentiment|Entity|Attribute|Sentence|Source_file|Annotator|Aspect' \
        .split('|')
    df = pd.read_csv(ds_path, sep='|', names=header_columns, header=0)
    print(df["Sentence"].sample(10))

    # filter out review texts with less than 5 characters
    df = df[df["Sentence"].str.len() >= 5]

    # filter out all reviews without sentiment
    df = df[df["Domain_Relevance"] == 9]

    # convert overall rating to sentiment
    df.insert(6, "sentiment_number", df["Sentiment"].replace({"p": 1, "0": 0, "n": -1}), allow_duplicates=True)

    # compute minimum number of occurences of all sentiments
    sent_count_min = df["sentiment_number"].value_counts().min()
    df = df.groupby("sentiment_number").head(sent_count_min)

    # shuffle data (random_state for reproducibility)
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)

    print("Total reviews: {}".format(len(df)))
    print(df["Sentiment"].value_counts())

    df.head()

    print("Creating .txt file that contains {} reviews: {}".format(rev_texts_path, len(df)))
    with open("../data/organic/processed/organic.txt", "w") as f:
        for i, row in df.iterrows():
            # f.write("review_{}\n".format(i))
            f.write("###{}\n".format(row["sentiment_number"]))
            f.write(row["Sentence"] + "\n\n")


preprocess_organic()
