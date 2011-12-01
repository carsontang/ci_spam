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

	# Return the number of items in a category
	def cat_count(self, category):
		if category in self.category_count:
			return float(self.category_count[category])
		return 0.0

	# Return the total number of items used in training
	def get_total_count(self):
		return sum(self.category_count.values())
	
	# Return a list of all categories
	def get_cats(self):
		return self.category_count.keys()
	
	# Return probability that feature is in category
	# Example: |{'money'}| / |{'money', 'money', 'casino'}| = 1/3
	def feat_prob(self, feature, category):
		if self.cat_count(category)	== 0: return 0
		return self.feat_count(feature, category) / self.cat_count(category)
	
	def train(self, item, category):
		features = self.get_features(item)
		
		for feature in features:
			self.inc_feat_count(feature, category)

		self.inc_cat_count(category)

def sample_train(cl):
	cl.train("We like playing board games Tuesdays and Thursdays.", "good")
	cl.train("Visit our casino along highway 5. We guarantee you will enjoy your experience.", "bad")
	cl.train("Buy now. Sale ends soon. Do not miss out on this great opportunity.", "bad")
	cl.train("Looking for a job. Come to our institute and earn a certificate in less than 9 months.", "bad")
	cl.train("Please attend class. It is crucial for you to understand the subjects you are being taught.", "good")
