from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nerual import pre_pro


TEST_PATH = 'test.csv'
PATH = 'train.csv'

if __name__ == '__main__':
	cv = CountVectorizer(max_features=25000, stop_words='english')
	data = pd.read_csv(PATH)
	data.tweet = data.tweet.apply(pre_pro)	
	f = cv.fit_transform(data.tweet)
	test_data = pd.read_csv(TEST_PATH)
	test_data.tweet = test_data.tweet.apply(pre_pro)
	test_x = test_data.tweet
	t = cv.transform(test_x)
	y = data['user']

	clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(f, y)
	clf.predict(t)
	print(clf.score(t, y))	
