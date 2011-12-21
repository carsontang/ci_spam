import re
import math
from pysqlite2 import dbapi2 as sqlite
import redis

def get_words(doc):
    '''Parses a text file into words of length greater than 2 and less than 20'''
    
    # Split the words by non-alpha characters
    splitter = re.compile('\\W*')
    words = [s.lower() for s in splitter.split(doc)
             if len(s) > 2 and len(s) < 20]
    
    # Return the unique set of words only
    return dict([ (w,1) for w in words ])

class Classifier(object):
    '''
    A generic classifier class. An instance of Classifier will be able
    to differentiate text of different categories.
    '''
    
    def __init__(self, get_features, file_name=None):
        '''Sets the classifier's get_features method to one passed in by the user.'''
        self.get_features = get_features
    
    def setdb(self, dbfile):
        '''
        Sets up a connection with the Redis database.
        '''
        
        '''
        self.fc contains the number of times a feature has appeared in a category.
        self.cc contains a category's item count.
        self.total_count contains the total number of items used for classification.
        '''
        
        self.redis_con      = redis.StrictRedis(host="localhost", port=6379, db=0)
        self.fc             = "fc"
        self.cc             = "cc"
        self.cat_list       = "cat_list"
        self.total_count    = "total_count"
        self.redis_con.set(self.total_count, 0)
    
    def inc_feat_count(self, feature, category):
        '''
        Increment the feature's count within a category
        '''
        count = self.feat_count(feature, category)
        if count == 0:
            self.redis_con.set("%s:%s:%s" % (self.fc, feature, category), 1)
        else:
            self.redis_con.incr("%s:%s:%s" % (self.fc, feature, category))

    
    def inc_cat_count(self, category):
        '''
        Increment the count of a category.
        '''
        count = self.cat_count(category)
        if count == 0:
            self.redis_con.set("%s:%s" % (self.cc, category), 1)
            self.redis_con.rpush(self.cat_list, category)
        else:
            self.redis_con.incr("%s:%s" % (self.cc, category))
        self.redis_con.incr(self.total_count)
    
    def feat_count(self, feature, category):
        '''
        Return the number of times a feature has appeared in a category.
        '''
        result = self.redis_con.get("%s:%s:%s" % (self.fc, feature, category))
        if result == None:
            return 0
        else:
            return int(result)
    
    def categories(self):
        '''
        Return a list of all the categories.
        '''
        result = self.redis_con.lrange(self.cat_list, 0, -1)
        return result
    
    
    def cat_count(self, category):
        '''
        Return the number of items in a category
        '''
        # TO-DO: figure out how to sum up counts in cat_count without having to keep track of it separately
        result = self.redis_con.get("%s:%s" % (self.cc, category))
        if result == None:
            return 0
        else:
            return float(result)
    
    def get_total_count(self):
        '''
        Return the total number of items used in training.
        '''
        result = int(self.redis_con.get(self.total_count))
        return result
    
    def feat_prob(self, feature, category):
        '''
        Return P(Feature | Category), probability that feature is in category.
        '''
        if self.cat_count(category) == 0: return 0
        return self.feat_count(feature, category) / self.cat_count(category)
    
    def weighted_prob(self, feature, category, prob_f, weight = 1.0, assumed_prob = 0.5):
        '''
        Return the weighted probability of a feature being in a category.
        '''
        
        basic_prob = prob_f(feature, category)
        
        # Count the number of times this feature has appeared in all categories
        totals = sum([self.feat_count(feature, category) for category in self.categories()])
        
        # Calculate the weighted average
        weighted_avg = ((weight*assumed_prob) + (totals*basic_prob))/(weight+totals)
        
        return weighted_avg
    
    def train(self, item, category):
        '''
        Add item in category to classifier.
        '''
        features = self.get_features(item)
        
        for feature in features:
            self.inc_feat_count(feature, category)
        
        self.inc_cat_count(category)
    
        # Commit the training data to the database
        self.redis_con.save()

class NaiveBayes(Classifier):
    '''
    A naive bayes classifier based on Toby Segaran's "Programming Collective Intelligence:
    Building Smart Web 2.0 Applications."
    '''
    
    def __init__(self, get_features):
        Classifier.__init__(self, get_features)
        self.thresholds = {}
    
    def set_threshold(self, category, threshold):
        '''
        Set the threshold of a item being classified as the specified category.
        For example, if the threshold of "spam" is 3, for an item to be classified as
        "spam", P("spam" | Document) >= 3 * P(Category | Document) for ALL categories.
        '''
        self.thresholds[category] = threshold
    
    def get_threshold(self, category):
        '''
        Return the threshold of classification for the specified category.
        If no threshold was specified, the default threshold value is 1.0.
        '''
        if category not in self.thresholds: return 1.0
        return self.thresholds[category]
    
    def doc_prob(self, item, category):
        '''
        Return P(Document | Category) = P(Feature_1 | Category) * P(Feature_2 | Category) * ...
        * P(Feature_N | Category)
        '''
        features = self.get_features(item)
        
        # Multiply the probabilities of all the features together.
        # Assumes features are independent
        prob = 1
        for feature in features: prob *= self.weighted_prob(feature, category, self.feat_prob)
        return prob
    
    def prob(self, item, category):
        '''
        Return a number proportional to P(Category | Document).
        There is no need to calculate P(Document) because it is the same for all categories 
        '''
        cat_prob = self.cat_count(category)
        cat_prob /= self.get_total_count()
        doc_prob = self.doc_prob(item, category)
        
        # P(Category | Document) * P(Document) = P(Document | Category) * P(Category)
        return doc_prob * cat_prob
    
    
    def classify(self, item, default = None):
        '''
        Return the classification of the document based on the classifier's training data.
        '''
        probs = {}
        
        max = 0.0
        for category in self.categories():
            probs[category] = self.prob(item, category)
            if probs[category] > max:
                max = probs[category]
                best = category
    
        # Ensure that P(Category | Document) exceeds the specified threshold
        for category in probs:
            if category == best: continue
            if probs[category] * self.get_threshold(best) > probs[best]: return default
        return best

def sample_train(cl):
    '''
    Train the classifier with this set of "good" and "bad" data.
    '''
    cl.train("Visit our casino along highway 5. We guarantee you will enjoy your experience.", "bad")
    cl.train("Buy now. Sale ends soon. Do not miss out on this great opportunity.", "bad")
    cl.train("We like playing board games Tuesdays and Thursdays.", "good")
    cl.train("Looking for a job. Come to our institute and earn a certificate in less than 9 months.", "bad")
    cl.train("Please attend class. It is crucial for you to understand the subjects you are being taught.", "good")
