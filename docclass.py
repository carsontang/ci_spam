import re
import math

def get_words(doc):
    splitter = re.compile('\\W*')
    # Split the words by non-alpha characters
    words = [s.lower() for s in splitter.split(doc)
         if len(s) > 2 and len(s) < 20]

    # Return the unique set of words only
    return dict([ (w,1) for w in words ])

class classifier:
    def __init__(self, get_features, file_name=None):
        # Mapping of feature to the number of times it has appeared in each category
        # Example:
        # feature is word
        # {'money':{'spam':10, 'okay':3}, 'viagra':{'spam':20, 'okay':1}}
        self.feature_count = {}

        # Counts of documents in each category
        self.category_count = {}
        self.get_features = get_features

    # Increment the feature's count within a category
    def inc_feat_count(self, feature, category):
        self.feature_count.setdefault(feature, {})
        self.feature_count[feature].setdefault(category, 0)
        self.feature_count[feature][category] += 1

    # Increment the count of a category
    def inc_cat_count(self, category):
        self.category_count.setdefault(category, 0)
        self.category_count[category] += 1

    # Return the number of times a feature has appeared in a category
    def feat_count(self, feature, category):
        if feature in self.feature_count and category in self.feature_count[feature]:
            return float(self.feature_count[feature][category])
        return 0.0

    # Return the categories
    def categories(self):
        return [category for category in self.category_count.keys()]
    
	# Return the number of items in a category
    def cat_count(self, category):
        if category in self.category_count:
            return float(self.category_count[category])
        return 0

        # Return the total number of items used in training
    def get_total_count(self):
        return sum(self.category_count.values())

    # Return a list of all categories
    def get_cats(self):
        return self.category_count.keys()

    # Return probability that feature is in category
    # Example: |{'money'}| / |{'money', 'money', 'casino'}| = 1/3
    def feat_prob(self, feature, category):
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
        features = self.get_features(item)

        for feature in features:
            self.inc_feat_count(feature, category)

        self.inc_cat_count(category)

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
    
    # Return the P(Category | Item) * P(Item)
    # There is no need to calculate P(Item) because it is the same for all categories 
    def prob(self, item, category):
        cat_prob = self.cat_count(category)
        cat_prob /= self.get_total_count()
        doc_prob = self.doc_prob(item, category)
        return doc_prob * cat_prob

    def classify(self, item, default = None):
        probs = {}
        
        # Find the category with the highest probability
        max = 0.0
        for category in self.categories():
            probs[category] = self.prob(item, category)
            if probs[category] > max:
                max = probs[category]
                best = category
        
        for category in probs:
            if category == best: break
            if probs[category] * self.get_threshold(best) > probs[best]: return default
        return best

def sample_train(cl):
    cl.train("We like playing board games Tuesdays and Thursdays.", "good")
    cl.train("Visit our casino along highway 5. We guarantee you will enjoy your experience.", "bad")
    cl.train("Buy now. Sale ends soon. Do not miss out on this great opportunity.", "bad")
    cl.train("Looking for a job. Come to our institute and earn a certificate in less than 9 months.", "bad")
    cl.train("Please attend class. It is crucial for you to understand the subjects you are being taught.", "good")
