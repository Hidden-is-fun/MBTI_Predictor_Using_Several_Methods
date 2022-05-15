# Data Analysis
import pandas as pd
import numpy as np

from pandas import DataFrame

# Text Processing
import re
from sklearn.preprocessing import LabelEncoder

# Machine Learning packages
from sklearn.feature_extraction.text import CountVectorizer

# Model training and evaluation
from sklearn.model_selection import train_test_split

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import accuracy_score

# Ignore noise warning
import warnings
warnings.filterwarnings("ignore")

# Show entire data instead of '...' by changing the value of 'max_colwidth'
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)
pd.set_option('max_colwidth', 50)

'''
Save the model using pickle or joblib (if you want)
想要尝试的话可以使用pickle或者joblib对模型进行持久化
'''


def output_csv(data, output_name, index=True, header=True):
    pd.DataFrame.to_csv(data, f'{output_name}.csv', index=index, header=header)


data_set = pd.read_csv("data/mbti.csv")

''' Output >> data_set.head(), data_set.tail()
   type                                              posts
0  INFJ  'http://www.youtube.com/watch?v=qsXHcwe3krw|||...
1  ENTP  'I'm finding the lack of me in these posts ver...
2  INTP  'Good one  _____   https://www.youtube.com/wat...
3  INTJ  'Dear INTP,   I enjoyed our conversation the o...
4  ENTJ  'You're fired.|||That's another silly misconce...
      type                                              posts
8670  ISFP  'https://www.youtube.com/watch?v=t8edHB_h908||...
8671  ENFP  'So...if this thread already exists someplace ...
8672  INTP  'So many questions when i do these things.  I ...
8673  INFP  'I am very conflicted right now when it comes ...
8674  INFP  'It has been too long since I have been on per...
'''

nRow, nCol = data_set.shape

''' Output >> nRow, nCol
There are 8675 rows and 2 columns
'''

types = np.unique(np.array(data_set['type']))

''' Output >> types
['ENFJ' 'ENFP' 'ENTJ' 'ENTP' 'ESFJ' 'ESFP' 'ESTJ' 'ESTP' 
 'INFJ' 'INFP' 'INTJ' 'INTP' 'ISFJ' 'ISFP' 'ISTJ' 'ISTP']
'''

total = data_set.groupby(['type']).count()
# output_csv(total, 'personality_type') -> See personality_type.csv

''' Output >> total
      posts
type       
ENFJ    190
ENFP    675
ENTJ    231
ENTP    685
ESFJ     42
ESFP     48
ESTJ     39
ESTP     89
INFJ   1470
INFP   1832
INTJ   1091
INTP   1304
ISFJ    166
ISFP    271
ISTJ    205
ISTP    337
'''

''' --- Now let we see how many words they type in each posts ---'''
# We just simply assume '[space]<Content>[space] = 1 word'
single_person = [0] * 60
single_posts = [0] * 60
lb_person = []
lb_posts = []
for i in data_set['posts']:
    single_person[int(len(i.split(' ')) / 50)] += 1
    posts = i.split('|||')
    for _ in posts:
        single_posts[int(len(_.split(' ')) / 5)] += 1
for _ in range(60):
    lb_person.append(f'{_ * 50}-{_ * 50 + 50}')
    lb_posts.append(f'{_ * 5}-{_ * 5 + 5}')
'''output_csv(DataFrame({'post_length': lb_person,
                      'count': single_person}), 'word_count_1', False)
output_csv(DataFrame({'post_length': lb_posts,
                      'count': single_posts}), 'word_count_2', False)'''


# After checking the Datasets, we found that there are many posts can't reflect user's personality, like:
# 1) Url links
# 2) Posts that pile with numbers, interjections
# 3) Extremely short posts
# ...
# So we should do some process to our dataset to make it 'cleaner'
# 在查看数据集之后，我们发现其中有许多数据并不能体现个人信息，例如超链接，大量的数字堆叠等，因此下一步我们需要进行数据的清洗


def preprocess_text(df, remove_special=True):
    # Remove links
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'https?://.*?[\s+]', '', x.replace("|", " ") + " "))

    # Keep the End Of Sentence characters
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'\.', ' EOSTokenDot ', x + " "))
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'\?', ' EOSTokenQuest ', x + " "))
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'!', ' EOSTokenExs ', x + " "))

    # Strip Punctuation
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'[\\.+]', ".", x))

    # Remove multiple fullstops
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'[^\w\s]', '', x))

    # Remove Non-words
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

    # Convert posts to lowercase
    df["posts"] = df["posts"].apply(lambda x: x.lower())

    # Remove multiple letter repeating words
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'([a-z])\1{2,}[\s|\w]*', '', x))

    # Remove very short or long words
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'(\b\w{0,3})?\b', '', x))
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'(\b\w{30,1000})?\b', '', x))

    # Remove multi spaces
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'(\s)\1+', ' ', x))

    # Remove MBTI Personality Words - crutial in order to get valid model accuracy estimation for unseen data.
    if remove_special:
        pers_types = ['INFP', 'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP', 'ISFP', 'ENTJ', 'ISTJ', 'ENFJ', 'ISFJ',
                      'ESTP', 'ESFP', 'ESFJ', 'ESTJ']
        pers_types = [p.lower() for p in pers_types]
        p = re.compile("(" + "|".join(pers_types) + ")")

    return df


# Preprocessing of entered Text
new_df = preprocess_text(data_set)

''' Output >> Process result (new_df)

# Before
0	'http://www.youtube.com/watch?v=qsXHcwe3krw|||http://41.media.tumblr.com/tumblr_lfouy03PMA1qa1rooo1_500.jpg|||enfp and intj moments  https://www.youtube.com/watch?v=iz7lE1g4XM4  sportscenter not top t...
1	'I'm finding the lack of me in these posts very alarming.|||Sex can be boring if it's in the same position often. For example me and my girlfriend are currently in an environment where we have to crea...
2	'Good one  _____   https://www.youtube.com/watch?v=fHiGbolFFGw|||Of course, to which I say I know; that's my blessing and my curse.|||Does being absolutely positive that you and your best friend could...
3	'Dear INTP,   I enjoyed our conversation the other day.  Esoteric gabbing about the nature of the universe and the idea that every rule and social code being arbitrary constructs created...|||Dear ENT...
4	'You're fired.|||That's another silly misconception. That approaching is logically is going to be the key to unlocking whatever it is you think you are entitled to.   Nobody wants to be approached wit...
5	'18/37 @.@|||Science  is not perfect. No scientist claims that it is, or that scientific  information will not be revised as we discover new things.  Rational  thinking has been very useful to our soc...
6	'No, I can't draw on my own nails (haha). Those were done by professionals on my nails. And yes, those are all gel.  You mean those you posted were done by yourself on your own nails? Awesome!|||Proba...
7	'I tend to build up a collection of things on my desktop that i use frequently and then move them into a folder called  'Everything' from there it get sorted into type and sub type|||i ike to collect ...
8	I'm not sure, that's a good question. The distinction between the two is so dependant on perception. To quote Robb Flynn, ''The hate you feel is nothing more, than love you feel to win this war.''|||G...
9	'https://www.youtube.com/watch?v=w8-egj0y8Qs|||I'm in this position where I have to actually let go of the person, due to a various reasons. Unfortunately I'm having trouble mustering enough strength ...

# After
0	 enfp intj moments sportscenter plays pranks what been most lifechanging experience your life eostokenquest repeat most today eostokendot perc experience immerse eostokendot last thing infj friend posted facebook before committing suicide next eostokendot rest peace hello enfj eostokendot sorry hear your distress eostokendot only natural relationship perfection time every moment existence eostokendot figure hard times times growth eostokendot eostokendot eostokendot eostokendot eostokendot eosto...
1	 finding lack these posts very alarming eostokendot boring same position often eostokendot example girlfriend currently environment where have creatively cowgirl missionary eostokendot there isnt enough eostokendot eostokendot eostokendot giving meaning game theory eostokendot hello entp grin thats takes eostokendot than converse they most flirting while acknowledge their presence return their words with smooth wordplay more cheeky grins eostokendot this lack balance hand coordination eostokendo...
2	good course which know thats blessing curse eostokendot does being absolutely positive that your best friend could amazing couple count eostokenquest than eostokendot more could madly love case reconciled feelings which eostokendot eostokendot eostokendot didnt thank link eostokenexs socalled tisi loop stem from current topicobsession deadly eostokendot like when youre stuck your thoughts your mind just wanders circles eostokendot feels truly terrible eostokendot eostokendot eostokendot eostoken...
3	dear intp enjoyed conversation other eostokendot esoteric gabbing about nature universe idea that every rule social code being arbitrary constructs created eostokendot eostokendot eostokendot dear entj long time eostokendot sincerely alpha none them eostokendot other types hurt deep existential ways that want part eostokendot probably sliding scale that depends individual preferences like everything humanity eostokendot draco malfoy also eostokendot either eostokendot either though which stackin...
4	youre fired eostokendot thats another silly misconception eostokendot that approaching logically going unlocking whatever think entitled eostokendot nobody wants approached with eostokendot eostokendot eostokendot guys eostokendot eostokendot eostokendot really wants superduperlongass vacation eostokendot cmon guys eostokendot boss just doesnt listen eostokendot even approached logically everything eostokendot never mind eostokendot just permanent vacation eostokendot months eostokenquest wouldn...
5	 eostokendot science perfect eostokendot scientist claims that that scientific information will revised discover things eostokendot rational thinking been very useful society eostokendot eostokendot eostokendot eostokendot infp edgar allen infp your siggy eostokendot people obvious quick infp eostokendot agree that eostokendot isfp eostokendot compare haku definite infp eostokendot flat through most naruto eostokendot eostokendot dont eostokendot eostokendot eostokendot lets this party started d...
6	 cant draw nails haha eostokendot those were done professionals nails eostokendot those eostokendot mean those posted were done yourself your nails eostokenquest awesome eostokenexs probably electronic screen syndrome eostokendot with advent technology social media suffer from overstimulation daily basis eostokendot guilty well eostokendot past happy just eostokendot eostokendot eostokendot love nail arts eostokenexs these some mine this first time hearing this about menstruation church eostoken...
7	 tend build collection things desktop that frequently then move them into folder called everything from there sorted into type type collect objects even work eostokendot eostokendot eostokendot people would call junk like collect eostokendot unused software eostokenquest take that your hands have bunch adobe eostokendot eostokendot eostokendot think quite normal tend only friends real life every couple months said earlier some people just dont good ones edit mostly mean tolerate eostokendot eost...
8	 sure thats good question eostokendot distinction between dependant perception eostokendot quote robb flynn hate feel nothing more than love feel this eostokendot good question eostokenexs tough sure loved winona ryder lydia beetlejuice eostokendot eostokendot eostokendot been lonely much time eostokendot while been working changing think trying find positive everything matter bleak might seem eostokendot eostokendot eostokendot eostokendot hope look back this current stretch time think thank th...
9	 this position where have actually person various reasons eostokendot unfortunately having trouble mustering enough strength actually pull through eostokendot sometimes eostokendot eostokendot eostokendot what year eostokendot what year eostokendot just utterly bewildered with this point eostokendot laundry eostokendot long clothes left wear fine eostokendot then time comes that left contend with mountain that laundry pile eostokendot sent from apollo eostokendot eostokendot eostokendot going ba...

We can see that the dataset is much clear than before.
经过数据处理，可以看到，数据集变得更加整齐，利于下一步的操作。
'''

#  We should to delete the posts which are too short after processing
#  将过短的post删除

min_words = 20
new_df["no. of. words"] = new_df["posts"].apply(lambda x: len(re.findall(r'\w+', x)))
new_df = new_df[new_df["no. of. words"] >= min_words]

post_length = [0] * 60
lb_length = []
for i in data_set['posts']:
    post_length[int(len(i.split(' ')) / 50)] += 1
'''output_csv(DataFrame({'post_length': lb_person,
                      'before': single_person,
                      'after': post_length}), 'post_length_after_processing', False)'''
# See src/4.png for details
# 处理前后结果见src/4.png

# Use Encoder to make labels to int type
# 将label转换为int类型
# Eg: ENFJ -> 0
enc = LabelEncoder()
new_df['type of encoding'] = enc.fit_transform(new_df['type'])
target = new_df['type of encoding']

''' Output >> new_df.head(10) 
   type                                              posts  no. of. words  type of encoding
0  INFJ   enfp intj moments sportscenter plays pranks w...            430                 8
1  ENTP   finding lack these posts very alarming eostok...            803                 3
2  INTP  good course which know thats blessing curse eo...            253                11
3  INTJ  dear intp enjoyed conversation other eostokend...            777                10
4  ENTJ  youre fired eostokendot thats another silly mi...            402                 2
5  INTJ   eostokendot science perfect eostokendot scien...            245                10
6  INFJ   cant draw nails haha eostokendot those were d...            970                 8
7  INTJ   tend build collection things desktop that fre...            140                10
8  INFJ   sure thats good question eostokendot distinct...            522                 8
9  INTP   this position where have actually person vari...            130                11
'''

# Converting posts (or training or X feature) into numerical form by count vectorization
# 将上面得到的处理后的数据通过CountVectorizer转换为int类型

vect = CountVectorizer(stop_words='english')
train = vect.fit_transform(new_df["posts"])

''' Output >> train
  (0, 26125)	2
  (0, 43213)	2
  (0, 55130)	1
  (0, 81682)	1
  (0, 65533)	1
  (0, 66790)	1
  (0, 48910)	1
  (0, 28344)	2
  (0, 48898)	2
  (0, 26798)	2
  (0, 71903)	1
  (0, 88338)	1
  (0, 26796)	81
  (0, 63913)	1
  (0, 40920)	1
  (0, 87211)	2
  (0, 41847)	2
  (0, 32315)	1
  (0, 66450)	1
  (0, 28892)	1
  (0, 15600)	1
  (0, 83980)	1
  (0, 72331)	1
  (0, 63442)	1
  (0, 37848)	1
  :	:
  (8424, 96087)	1
  (8424, 81605)	1
  (8424, 64302)	1
  (8424, 87571)	1
  (8424, 8629)	1
  (8424, 3007)	1
  (8424, 20238)	1
  (8424, 16788)	2
  (8424, 25314)	1
  (8424, 85649)	1
  (8424, 72380)	1
  (8424, 11892)	1
  (8424, 57151)	1
  (8424, 37167)	1
  (8424, 16917)	1
  (8424, 48850)	1
  (8424, 86355)	1
  (8424, 77323)	1
  (8424, 15204)	1
  (8424, 88723)	1
  (8424, 70638)	1
  (8424, 67062)	1
  (8424, 85664)	1
  (8424, 6874)	1
  (8424, 5723)	1
'''

# Let us see what is 'stop_words'
# 上面的参数中填入了stop_words='english',stopwords可以理解为python中的一个可以提供高频(无实际意义)词的库
'''Output >> stopwords.words('english')) ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
"you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 
'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 
'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 
'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 
'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 
'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 
'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 
'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 
'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', 
"mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', 
"won't", 'wouldn', "wouldn't"] 
'''

# Check the shape of the data we got above
print(f"There are {train.shape[1]} features in the dataset for {train.shape[0]} rows")

# Now we can make train set and test set!
# 接下来我们就可以生成训练集和测试集了
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.15, stratify=target, random_state=1)


def get_pred_info(y, pred, model_name):
    pred_chart = []
    for _ in range(16):
        temp = []
        for _ in range(16):
            temp.append(0)
        pred_chart.append(temp)
    __ = []
    for _ in y:
        __.append(_)
    for _ in range(len(__)):
        pred_chart[pred[_]][__[_]] += 1
    output_csv(DataFrame(pred_chart), model_name, False, False)


accuracies = {}

# Random Forest
random_forest = RandomForestClassifier(n_estimators=200)
random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)
predictions = [round(value) for value in Y_pred]

accuracy = accuracy_score(y_test, predictions)
get_pred_info(y_test, predictions, 'Random_Forest')

accuracies['Random Forest'] = accuracy * 100.0
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# XG boost Classifier
xgb = XGBClassifier(n_estimators=200)
xgb.fit(X_train, y_train)

Y_pred = xgb.predict(X_test)
predictions = [round(value) for value in Y_pred]

accuracy = accuracy_score(y_test, predictions)
get_pred_info(y_test, predictions, 'XG_Boost')

accuracies['XG Boost'] = accuracy * 100.0
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Gradient Descent
sgd = SGDClassifier(max_iter=10, tol=None)
sgd.fit(X_train, y_train)

Y_pred = sgd.predict(X_test)
predictions = [round(value) for value in Y_pred]

accuracy = accuracy_score(y_test, predictions)
get_pred_info(y_test, predictions, 'Gradient_Descent')

accuracies['Gradient Descent'] = accuracy * 100.0
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)
predictions = [round(value) for value in Y_pred]

accuracy = accuracy_score(y_test, predictions)
get_pred_info(y_test, predictions, 'Logistic_Regression')

accuracies['Logistic Regression'] = accuracy * 100.0
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=2)  # n_neighbors means k
knn.fit(X_train, y_train)

Y_pred = knn.predict(X_test)
predictions = [round(value) for value in Y_pred]

accuracy = accuracy_score(y_test, predictions)
get_pred_info(y_test, predictions, 'KNN')

accuracies['KNN'] = accuracy * 100.0
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# SVM
svm = SVC(random_state=1)
svm.fit(X_train, y_train)

Y_pred = svm.predict(X_test)

predictions = [round(value) for value in Y_pred]
accuracy = accuracy_score(y_test, predictions)
get_pred_info(y_test, predictions, 'SVM')

accuracies['SVM'] = accuracy * 100.0
print("Accuracy: %.2f%%" % (accuracy * 100.0))

''' Output >> accuracies
{
'Random Forest':       43.20774614472123, 
'XG Boost':            58.701613285883746, 
'Gradient Descent':    37.43408066429419, 
'Logistic Regression': 59.34372479240806, 
'KNN':                 17.72946619217082, 
'SVM':                 37.90278766310795
}
'''
# For more details see /src/5.png and /src/6.png

'''
Conclusion:
We can clearly see that the accuracy is pretty bad for a predictor. I think that the main reason is we choose too much
labels, also the count of people who have specific personality (ESxx etc.) is pretty low, that makes model has not
learned enough form the training data, and resulting in unreliable predictions.
结论：
明显可见，作为一个性格测试预测器来说，模型所体现出的准确度非常不令人满意，分析原因大概有以下两点：
1）标签被分为16类，过多的标签导致拟合效果降低
2）拥有不同性格标签的人数差距很大（例如ES**类型的人数明显要比其他类型人数要少），导致模型不能从训练集中学习到足够的特征信息，进而导致不可靠的
预测结果。
'''
