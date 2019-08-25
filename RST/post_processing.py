import re
import csv
nb_files = 1


def post_processing():
    """Create txt-file and csv-file containing the processed reviews of all sub-files

    The from the RST parser generated files are merged back into files containing all reviews
    now containing the EDU_BREAK keyword.
    """
    reviews = ""

    for i in range(nb_files):
        with open("/Users/tim/GoogleDrive/TU/NLP_Lab/MILNET-pytorch/tim/data/organic/processed/edus/{}.txt.edus".format(i + 1), "r") as f:
            reviews += f.read()

    reviews = re.sub("(-LRB-|-RRB-|-LSB-|-RSB-)", "", reviews)

    for i in range(5):
        reviews = re.sub('### {}.0 (EDU_BREAK )?'.format(i + 1), '\n###{}.0\n'.format(i + 1), reviews)
    reviews = reviews[1:]

    # txt
    with open("/Users/tim/GoogleDrive/TU/NLP_Lab/MILNET-pytorch/tim/data/organic/processed/final/organic_edus.txt", "w") as f:
        f.write(reviews)

    reviews = reviews.split("###")[1:]

    # csv
    with open("/Users/tim/GoogleDrive/TU/NLP_Lab/MILNET-pytorch/tim/data/organic/processed/final/organic_edus.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["overall", "reviewText"])

        for review in reviews:
            rating = review[0:3]
            review_text = review[3:]
            writer.writerow([rating, review_text])

post_processing()