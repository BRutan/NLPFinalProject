#################################################
# Classifiers.py
#################################################
# Objects used to classify sentiment of tweets
# as well as perform utility classifications
# (ex: spam).

from abc import ABC, abstractmethod, abstractproperty
import matplotlib.pyplot as plt
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.linear_model as slm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from time import sleep
from wordcloud import ImageColorGenerator, WordCloud, STOPWORDS 

class FrequencyCloud:
    """
    * Compute word frequency of generic text
    and display in a WordCloud.
    """
    def __init__(self, text, topN):
        """
        * Generate wordcloud and display based on
        topN words in text.
        Inputs:
        * text: string or iterable of string sentences.
        * topN: Integer or Tuple of integers if want to display multiple
        topN wordclouds.
        """
        self.__Generate(text, topN)

    ################
    # Properties:
    ################
    @property
    def Clouds(self):
        return self.__clouds
    @property
    def Sentiment(self):
        return self.__sentiment
    ################
    # Interface Functions:
    ################
    def DisplayAll(self, waitSeconds):
        """
        * Display all generated word clouds for
        waitSeconds then close.
        """
        for n in self.__clouds:
            plt.axis("off")
            plt.imshow(self.__clouds[n], interpolation='bilinear')
            sleep(waitSeconds)
            plt.close()
            
    def SavePlots(self, folder):
        """
        * Output all wordclouds to target folder.
        """
        for n in self.__clouds:
            plt.imshow(self.__clouds[n], interpolation='bilinear')
            plt.axis("off")
            plt.savefig("%s/%s_top_%s.png" % (folder, self.__sentiment, n))
    ################
    # Private Helpers:
    ################
    def __Generate(self, sentiment, text, topN):
        """
        * Generate one or more wordclouds using all
        tokens listed in text.
        """
        self.__clouds = {}
        self.__sentiment = sentiment
        # Append all texts together to generate wordcloud:
        texts = []
        for uid in text:
            reviewTexts.extend(reviews[uid].ReviewText)
        combinedText = ''.join(reviewTexts)
        # Generate and store wordcloud for each topN requested:
        topN = (topN) if not isinstance(topN, tuple) else topN
        for n in topN:
            self.__clouds[n] = WordCloud(max_words=n).generate(combinedText)

class DataSplit:
    """
    * Split reviews into training and testing.
    """
    def __init__(self, dataObjs, testRatio = .5):
        """
        * Split reviews into training and testing.
        Inputs: 
        * dataObjs: Dictionary mapping { sentiment -> Tweet }.
        Unlabeled data will be placed into Unlabeled property.
        """
        self.__Initialize()
        self.__Split(dataObjs, testRatio)
    ############
    # Properties:
    ############
    @property
    def Testing(self):
        return self.__testing
    @property
    def Training(self):
        return self.__training
    @property
    def Unlabeled(self):
        return self.__unlabeled
    ############
    # Private Helpers:
    ############
    def __Initialize(self):
        """
        * Initialize the object.
        """
        self.__testing = None
        self.__training = None
        self.__unlabeled = None
        
    def __Split(self, tweets, testRatio = .5):
        """
        * Split dataset into training and 
        testing.
        Inputs:
        * dataObjs: Dictionary mapping { sentiment -> Tweets }.
        * testRatio: Proportion of reviews for given sentiment to
        use as testing set.
        """
        options = { 'test_size' : testRatio, 'shuffle' : True }
        # Divide into training and testing sets:
        data = []
        for tweet in tweets:
            if sentiment != 'unlabeled':
                sentNumerical = 1 if sentiment == 'positive' else 0
                # Keep track of the original sentiment:
                reviews = tweets[sentiment].Reviews
                for uid in reviews:
                    reviews[uid].sentiment = sentNumerical
                data.extend(reviews.values())
            else:
                self.__unlabeled = tweets[sentiment].Reviews
        trainTest = train_test_split(data, **options)
        # Map back to { uid -> Review }:
        self.__testing = { review.UniqueId : review for review in trainTest[1] } 
        self.__training = { review.UniqueId : review for review in trainTest[0] } 

#################
# Classifiers:
#################
class SentimentPerformance:
    """
    * Compute and display performance of model.
    """
    def __init__(self, model, name, training, testing, labels):
        """
        * Compute and store performance of model
        on training set.
        Inputs:
        * model: Classification model (must have .predict() function).
        * name: String name of model.
        * training: Sentences used in training the model.
        * testing: Sentences used in testing the model.
        * labels: Dictionary mapping { Training/Testing -> Labels }.
        """
        self.__Initialize(name)
        self.__Calculate(model, training, testing, labels)
    
    def __str__(self):
        """
        * Display as string.
        """
        props = [attr for attr in dir(self) if not attr.startswith('_') and attr != 'ModelName']
        iters = ['Training', 'Testing']
        str_ = [self.__modelname, ' Performance:\n']
        for iter in iters:
            str_.append('%s:\n' % iter)
            for prop in props:
                variable = getattr(self, prop)[iter]
                values = '(%.2f,%.2f)' % (variable[0], variable[1])
                str_.append('%s:%s\t' % (prop, values))
            str_.append('\n')
        return ' '.join(str_)

    #############
    # Properties:
    #############
    @property
    def FScore(self):
        return self.__fscore
    @property
    def ModelName(self):
        return self.__modelname
    @property
    def Precision(self):
        return self.__precision
    @property
    def Recall(self):
        return self.__recall
    #############
    # Private Helpers:
    #############
    def __Calculate(self, model, training, testing, labels):
        """
        * Compute model performance statistics for
        both in-sample (Training) and out-of-sample sets (Testing).
        """
        iters = ['Training', 'Testing']
        encoder = LabelEncoder()
        for num, sentences in enumerate([training, testing]):
            iter_name = iters[num]
            predictions = model.predict(sentences)
            encoded_labels = encoder.fit_transform(labels[iter_name])
            performance = precision_recall_fscore_support(predictions, encoded_labels)
            self.__fscore[iter_name] = performance[2]
            self.__precision[iter_name] = performance[0]
            self.__recall[iter_name] = performance[1]

    def __Initialize(self, name):
        """
        * Initialize empty object.
        """
        self.__modelname = name
        self.__fscore = {}
        self.__precision = {}
        self.__recall = {}

class SentimentAnalyzer(ABC):
    """
    * Abstract base class used to classify whether
    a sentence has positive or negative
    sentiment.
    """
    def __init__(self, model, split, name):
        """
        * Store and train the model.
        """
        self.__model = model
        self.__name = name
        self.__Train(split)
    ###########
    # Properties:
    ###########
    @property
    def Model(self):
        return self.__model
    @property
    def Name(self):
        return self.__name
    @property
    def Performance(self):
        """
        * SentimentPerformance object generated by
        testing dataset.
        """
        return self.__performance
    ###########
    # Interface Methods:
    ###########
    def Classify(self, sentence):
        """
        * Predict sentiment of sentence(s)
        using model.
        Inputs:
        * sentence: string sentence or list of string sentences.
        """
        if not isinstance(sentence, (str, list, dict)):
            raise ValueError('sentence must be a string or list of strings.')
        elif isinstance(sentence, list) and not all([isinstance(rev, str) for rev in sentence]):
            raise ValueError('sentence must only contain strings if a list.')
        sentence = [sentence] if not isinstance(sentence, (dict, list)) else sentence
        if isinstance(sentence, dict):
            # Maintain the original mapping:
            keys = sentence.keys()
            predictions = self.__model.predict_proba(sentence.values())
            out = { key : predictions[num] for num, key in enumerate(keys) }
        else:
            # Map sentence to numeric classification under model:
            keys = sentence
            predictions = self.__model.predict_proba(sentence)
            out = { key : predictions[num] for num, key in enumerate(keys) }
        return out
        
    ###########
    # Private Helpers:
    ###########
    def __Train(self, split):
        """
        * Train the model and evaluate performance 
        using passed training and testing set.
        """
        encoder = LabelEncoder()
        training_reviews = split.Training.values()
        testing_reviews = split.Testing.values()
        training_sentences = [review.ReviewText for review in training_reviews]
        testing_sentences = [review.ReviewText for review in testing_reviews]
        # Encode the binary sentiment:
        training_labels = [review.sentiment for review in training_reviews]
        testing_labels = [review.sentiment for review in testing_reviews]
        labels = {'Training' : training_labels, 'Testing' : testing_labels}
        encoded_training_labels = encoder.fit_transform(training_labels)
        # Fit the model to data:
        self.__model.fit(training_sentences, encoded_training_labels)
        # Evaluate out-of-sample performance:
        self.__performance = SentimentPerformance(self.__model, self.__name, training_sentences, testing_sentences, labels)


class SpamClassifier(SentimentAnalyzer):
    """
    * N-Gram model to classify text as 
    spam/not-spam.
    """
    def __init__(self, split):
        """
        * Initialize model using data.
        """
        model = ""
        super().__init__(model, split, 'SpamClassifier')
        



    
class LogisticRegression(SentimentAnalyzer):
    """
    * Perform logistic regression to 
    classify sentiment based upon tokens.
    """
    def __init__(self, split):
        """
        * Train logistic regression model.
        Inputs:
        * split: DataSplit object.
        """
        model = Pipeline([('vec', CountVectorizer()), ('Model' ,slm.LogisticRegression())])
        super().__init__(model, split, 'LogisticRegression')

class NaiveBayes(SentimentAnalyzer):
    """
    * Use Naive Bayes to classify text.
    """
    def __init__(self, split):
        """
        * Train Naive Bayes classifier using training 
        set.
        Inputs:
        * split: DataSplit object.
        """
        model = Pipeline([('vec', CountVectorizer()), ('Model' ,BernoulliNB())])
        super().__init__(model, split, 'NaiveBayes')

