import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


TRAIN = 'trainNotRT.csv'
# TWEET = 'tweet'
USER = 'user'
EMOJIS = 'emojilists'
SEP = ', '
FILE_NAME = 'user_emojis.csv'


def get_emojis():
	df = pd.read_csv(TRAIN)
	return df[[USER, EMOJIS]]	

def process(emoji_str):
	# the format of the emoji_str is [*emoji1_encoding*, ..., *emojin_encoding*]
	return emoji_str[1:-1].split(SEP)

def get_counts(df):
	'''
	Counts for each user how many times each emoji appears
	'''
	counts = dict()
	for index, row in df.iterrows():
		emojis = process(row[EMOJIS])
		if '' not in emojis:  
			try:
				counts[row[USER]] += Counter(emojis)
			except KeyError:
				counts[row[USER]] = Counter(emojis)
	return counts

def get_top_counts(counts, n=5):
	'''
	Retruns a copy of a dictionary with only emojis that appeared the most
	'''
	result = Counter()
	for user, emojis in counts.items():
		result[user] = Counter(dict(sorted(emojis.items(), key=lambda x: x[1], reverse=True)[:n]))
	return result

def remove_low_counts(counts, threshold=40):
	'''
	Returns a copy of a dictionary with only emojis that appeared more than a threshold times
	'''
	result = Counter()
	for user, emojis in counts.items():
		result[user] = Counter(dict([entry for entry in emojis.items() if entry[1] >= threshold]))		
	return result
			
def remove_common_counts(counts):
	'''
	Returns a copy of a dictionary with only emojis that are unique to each user
	'''
	# fidning what emojis to remove for each user
	to_remove = dict()
	for user_i in counts.keys():
		for user_j, emojis in counts.items():
			if user_i != user_j:
				try:
					to_remove[user_i].update(emojis.keys())
				except KeyError:
					to_remove[user_i] = set(emojis.keys())
	
	# creating a dict with only unique emojis
	result = Counter()
	for user, emojis in counts.items():
		result[user] = Counter(dict([entry for entry in emojis.items() if entry[0] not in to_remove[user]]))
	return result

			
def plot_counts(counts):
	for user, emojis in counts.items():
		# decoded = [unicode(emoji, 'unicode-escape') for emoji in emojis.keys()]
		decoded = emojis.keys()
		plt.bar(decoded, emojis.values())
		plt.title('User #{}'.format(user))
		plt.xticks(rotation='vertical')
		plt.show()

def write_counts(counts):
	with open(FILE_NAME, 'w') as f:
		f.write(SEP.join([USER, EMOJIS]))
		f.write('\n')
		for user, emojis in counts.items():
			f.write(SEP.join([str(user), str(list(emojis.keys()))]))	
			f.write('\n')

def print_counts(coutns):
	for user, emojis in counts.items():
		print('User #{}: {}'.format(user, emojis))

if __name__ == '__main__':
	counts = remove_common_counts(remove_low_counts(get_counts(get_emojis())))
	# print_counts(counts)	
	# plot_counts(counts)
	write_counts(counts)
