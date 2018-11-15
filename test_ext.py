import csv
import fileinput

#Variables that contains the user credentials to access Twitter API
access_token = "1059659088630472705-YgDGgPwcx4DxCkIksDCRJ35hREbKNp"
access_token_secret = "PHqFvPv8nvy5bM9gzXCRf2cVu1DwQgRzLO9FOYzSEVKLk"
consumer_key = "xbNCpYwzHJ8vQ7FTlnceHOPS4"
consumer_secret = "Qr6euKSDCmZ0svNOKPAdtEF3l9u8tsyyb0eOtAl9JceKkxJsHf"


import twitter
api = twitter.Api(
    consumer_key=consumer_key,
    consumer_secret=consumer_secret,
    access_token_key=access_token,
    access_token_secret=access_token_secret)

hashtags_to_track = [
    "#mood",
]

LANGUAGES = ['en']

stream = api.GetStreamFilter(track=hashtags_to_track, languages=LANGUAGES)


with open('test_tweets.csv', 'w') as csv_file:
    csv_writer = csv.writer(csv_file)
    for line in stream:
        # Signal that the line represents a tweet
        if 'in_reply_to_status_id' in line:
            tweet = twitter.Status.NewFromJsonDict(line)
            print(tweet.id)
            row = [tweet.id, tweet.user.screen_name, tweet.text]
            csv_writer.writerow(row)
            csv_file.flush()

