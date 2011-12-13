import re
import math
from pysqlite2 import dbapi2 as sqlite

def get_words(doc):
    splitter = re.compile('\\W*')
    # Split the words by non-alpha characters
    words = [s.lower() for s in splitter.split(doc)
             if len(s) > 2 and len(s) < 20]
    
    # Return the unique set of words only
    return dict([ (w,1) for w in words ])

class classifier:
    def __init__(self, get_features, file_name=None):
        """
            Mapping of feature to the number of times it has appeared in each category
            Example:
            feature is word
            {'money':{'spam':10, 'okay':3}, 'viagra':{'spam':20, 'okay':1}}
            """
        self.feature_count = {}
        
        # Counts of documents in each category
        self.category_count = {}
        self.get_features = get_features
    
    """
        Database methods
        """
    def setdb(self, dbfile):
        self.con = sqlite.connect(dbfile)
        self.con.execute("create table if not exists feat_count(feature, category, count)")
        self.con.execute("create table if not exists cat_count(category, count)")
    
    """
        Classifier methods
        """
    
    
    def inc_feat_count(self, feature, category):
        """
            Increment the feature's count within a category
            """
        count = self.feat_count(feature, category)
        if count == 0:
            self.con.execute("insert into feat_count values ('%s', '%s', 1)"
                             % (feature, category))
        else:
            self.con.execute("update feat_count set count=%d where feature='%s' and category='%s'"
                             % (count+1, feature, category))
    
    def inc_cat_count(self, category):
        """
            Increment the count of a category
            """
        count = self.cat_count(category)
        if count == 0:
            self.con.execute("insert into cat_count values ('%s', 1)" % (category))
        else:
            self.con.execute("update cat_count set count = %d where category = '%s'"
                             % (count+1, category))
    
    def feat_count(self, feature, category):
        """
            Return the number of times a feature has appeared in a category
            """
        result = self.con.execute(
                                  "select count from feat_count where feature = '%s' and category = '%s'"
                                  % (feature, category)).fetchone()
        
        if result == None: return 0
        else: return float(result[0])
    
    def categories(self):
        """
            Return the categories
            """
        result = self.con.execute("select category from cat_count")
        return [tuple[0] for tuple in result]
    
    def cat_count(self, category):
        """
            Return the number of items in a category
            """
        result = self.con.execute("select count from cat_count where category = '%s'"
                                  % (category)).fetchone()
        if result == None: return 0
        else: return float(result[0])
    
    def get_total_count(self):
        """
            Return the total number of items used in training
            """
        result = self.con.execute("select count from cat_count")
        if result == None: return 0
        return sum([float(tuple[0]) for tuple in result])
    
    def get_cats(self):
        """
            Return a list of all categories    
            """
        result = self.con.execute("select category from cat_count")
        return [tuple[0] for tuple in result]
    
    def feat_prob(self, feature, category):
        """
            Return probability that feature is in category
            Example: |{'money'}| / |{'money', 'money', 'casino'}| = 1/3
            """
        if self.cat_count(category) == 0: return 0
        return self.feat_count(feature, category) / self.cat_count(category)
    
    def weighted_prob(self, feature, category, prob_f, weight = 1.0, assumed_prob = 0.5):
        basic_prob = prob_f(feature, category)
        
        # Count the number of times this feature has appeared in all categories
        totals = sum([self.feat_count(feature, category) for category in self.categories()])
        
        # Calculate the weighted average
        weighted_avg = ((weight*assumed_prob) + (totals*basic_prob))/(weight+totals)
        
        return weighted_avg
    
    def train(self, item, category):
        """
            Add item in category to classifier.
            Commit the addition of this item to the database
            """
        features = self.get_features(item)
        
        for feature in features:
            self.inc_feat_count(feature, category)
        
        self.inc_cat_count(category)
        self.con.commit()

class naivebayes(classifier):
    
    def __init__(self, get_features):
        classifier.__init__(self, get_features)
        self.thresholds = {}
    
    def set_threshold(self, category, threshold):
        self.threshold[category] = threshold
    
    def get_threshold(self, category):
        if category not in self.thresholds: return 1.0
        return self.thresholds[category]
    
    def doc_prob(self, item, category):
        features = self.get_features(item)
        
        # Multiply the probabilities of all the features together; assumes features are independent
        prob = 1
        for feature in features: prob *= self.weighted_prob(feature, category, self.feat_prob)
        return prob
    
    def prob(self, item, category):
        """
            Return a number proportional to P(Category | Item)
            There is no need to calculate P(Item) because it is the same for all categories 
            """
        cat_prob = self.cat_count(category)
        cat_prob /= self.get_total_count()
        doc_prob = self.doc_prob(item, category)
        
        # P(Item | Category) * P(Category)
        return doc_prob * cat_prob
    
    
    def classify(self, item, default = None):
        probs = {}
        
        max = 0.0
        for category in self.categories():
            probs[category] = self.prob(item, category)
            if probs[category] > max:
                max = probs[category]
                best = category
        
        for category in probs:
            if category == best: continue
            if probs[category] * self.get_threshold(best) > probs[best]: return default
        return best

def sample_train(cl):
    cl.train("We like playing board games Tuesdays and Thursdays.", "good")
    cl.train("Visit our casino along highway 5. We guarantee you will enjoy your experience.", "bad")
    cl.train("Buy now. Sale ends soon. Do not miss out on this great opportunity.", "bad")
    cl.train("Looking for a job. Come to our institute and earn a certificate in less than 9 months.", "bad")
    cl.train("Please attend class. It is crucial for you to understand the subjects you are being taught.", "good")
