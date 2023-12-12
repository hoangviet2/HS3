#import chatbot
import asyncio, json
from EdgeGPT.EdgeGPT import Chatbot, ConversationStyle
import re
# Array and Dataframe
import numpy as np
import pandas as pd
from os import getcwd
import nltk
# Training
from sklearn.model_selection import train_test_split
nltk.download('twitter_samples')
nltk.download('stopwords')
import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples
import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """

    stemmer = PorterStemmer()

    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)

    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    #tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'https?:\/\/\S*', '', tweet, flags=re.MULTILINE)

    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # remove @
    tweet = re.sub('@[^\s]+','',tweet)
    #print(tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()
    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        #print("THIS IS ANS" + str(y))
        for word in process_tweet(tweet):
            #print("word: "+ word)
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs
# select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')# split the data into two pieces, one for training and one for testing (validation set)
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# combine positive and negative labels
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)
freqs = build_freqs(train_x, train_y)

def sigmoid(z):
    '''
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    '''
    # calculate the sigmoid of z
    h = 1/(1+np.exp(-1*z))

    return h

def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
       '''

    # get 'm', the number of rows in matrix x
    m = len(x)

    for i in range(0, num_iters):

        # get z, the dot product of x and theta
        z = np.dot(x,theta)

        # get the sigmoid of z
        h = sigmoid(z)
        # calculate the cost function
        try:
          J = ((np.dot(np.transpose(y),np.log(h)))+(np.dot(np.transpose(1-y),np.log(1-h))))*(-1/m)

          # update the weights theta
          theta = theta - (alpha/m)*(np.dot(np.transpose(x),(h-y)))
        except:
          print("val h" + h)
    J = float(J)
    return J, theta

def extract_features(tweet, freqs):
    '''
    Input:
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output:
        x: a feature vector of dimension (1,3)
    '''
    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_tweet(tweet)

    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3))

    #bias term is set to 1
    x[0,0] = 1


    # loop through each word in the list of words
    for word in word_l:

        # increment the word count for the positive label 1
        x[0,1] += freqs.get((word,1.0),0)

        # increment the word count for the negative label 0
        x[0,2] += freqs.get((word,0.0),0)

    assert(x.shape == (1, 3))
    return x

# collect the features 'x' and stack them into a matrix 'X'
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freqs)

# training labels corresponding to X
Y = train_y

# Apply gradient descent
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 2000)
print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")

def predict_tweet(tweet, freqs, theta):
    '''
    Input:
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output:
        y_pred: the probability of a tweet being positive or negative
    '''

    # extract the features of the tweet and store it into x
    x = extract_features(tweet,freqs)

    # make the prediction using x and theta
    y_pred = sigmoid(np.dot(x,theta))

    return y_pred



# flask
from database import Users, Chat_sessions, Chat_messages, Q_values, S_values

# Import streamlit components
import streamlit as st
from st_pages import Page, show_pages, add_page_title
import time

# import modules
from chat_setup_configuration import page_configure
from css_customization import customization
from page_management import management

# Data Processing
import pandas as pd

# Load cookies
cookies = json.loads(open("assets/cookies.json", encoding="utf-8").read()) 

# set up page configuration
page_configure()

# Pages management
management()

# css configuration
customization()




# Function to get response from chatbot
async def main(res, input_text):
    #bot = await Chatbot.create(cookies=cookies)
    #response = await bot.ask(prompt=input_text, conversation_style=ConversationStyle.creative, simplify_response=True)
    response = "";
    y_hat = predict_tweet(input_text, freqs, theta)
    if y_hat > 0.5:
        response = f'Look like you had go thourgh a very great experience! \n With possibility of {y_hat[0][0] * 100}%'
    else:
        response = f'Look like you had go thourgh a very bad experience! \n With possibility of {y_hat[0][0] * 100}%'
    #print(predict_tweet(input_text, freqs, theta))
    #bot_response = response["text"]
    #output_response = re.sub('\[\^\d+\^\]', '', bot_response)
    # use regex to get the output in correct format
    res = response
    return res


def chat_function():
    st.title("Phần mềm giúp xác định cảm xúc dựa trên đoạn văn bản")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Tell us your story"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        print(prompt)
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            res = ""
            res = asyncio.run(main(res, prompt))
            assistant_response = res
            
            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    
    

chat_function()