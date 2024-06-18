from transformers import pipeline

BATCH_COUNT = 0
BATCH_SIZE = 10
sentiment_calculated = []


def run_model(twitter, column_name = "text"):
    pipe = pipeline("text-classification", model="finiteautomata/bertweet-base-sentiment-analysis")

    def label_to_number(label):
      if label == 'NEG':
        return 0
      if label == 'POS':
        return 4
      if label == 'NEU':
        return 2

    def create_tweet_list():
      for tweet in twitter[column_name]:
        if len(tweet) > 128:
          tweet = tweet[:128]
        tweet_list.append(tweet)

    def calculate_sentiment_with_outside_model():
      global BATCH_COUNT
      while BATCH_COUNT < (len(twitter) / BATCH_SIZE):
        print("Starting batch " + str(BATCH_COUNT+1))
        local_sentiment_calculated = pipe(tweet_list[(BATCH_SIZE * BATCH_COUNT):(BATCH_SIZE * (BATCH_COUNT + 1))])
        print("Completed batch " + str(BATCH_COUNT+1) + " processed " + str((BATCH_COUNT + 1) * BATCH_SIZE) + "/" + str(len(twitter)) + "tweets." )
        for result in local_sentiment_calculated:
          sentiment_calculated.append(label_to_number(result['label']))
        BATCH_COUNT = BATCH_COUNT + 1

    tweet_list = []

    create_tweet_list()
    calculate_sentiment_with_outside_model()
    return sentiment_calculated

def combine(original_data, sentiment):
  original_data["outer_sentiment"] = sentiment
  return original_data