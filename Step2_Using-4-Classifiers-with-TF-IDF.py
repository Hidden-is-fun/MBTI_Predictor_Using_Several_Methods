import re

import numpy as np
import pandas as pd
import sklearn.metrics
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

'''
Intro:
In Step1 we do some process to the dataset and get the 'unreliable' predict models, so how to solve it?
The answer is: We can try to split 'type' into 4 binary classification.
前言：
在Step1中我们通过直接对posts和type进行处理，得到了'不是那么可靠的'预测模型，那么该如何提升模型准确度呢？
我猜其中一种解决办法是：将type分为4个2分类
'''

data = pd.read_csv("data/mbti.csv")


def get_types(row):
    t = row['type']

    I = 1 if t[0] == 'E' else 0
    N = 1 if t[1] == 'S' else 0
    F = 1 if t[2] == 'T' else 0
    J = 1 if t[3] == 'P' else 0

    return pd.Series({'IE': I, 'NS': N, 'FT': F, 'JP': J})


data = data.join(data.apply(lambda row: get_types(row), axis=1))

''' Output >> data.head(10)
   type                                              posts  IE  NS  FT  JP
0  INFJ  'http://www.youtube.com/watch?v=qsXHcwe3krw|||...   0   0   0   0
1  ENTP  'I'm finding the lack of me in these posts ver...   1   0   1   1
2  INTP  'Good one  _____   https://www.youtube.com/wat...   0   0   1   1
3  INTJ  'Dear INTP,   I enjoyed our conversation the o...   0   0   1   0
4  ENTJ  'You're fired.|||That's another silly misconce...   1   0   1   0
5  INTJ  '18/37 @.@|||Science  is not perfect. No scien...   0   0   1   0
6  INFJ  'No, I can't draw on my own nails (haha). Thos...   0   0   0   0
7  INTJ  'I tend to build up a collection of things on ...   0   0   1   0
8  INFJ  I'm not sure, that's a good question. The dist...   0   0   0   0
9  INTP  'https://www.youtube.com/watch?v=w8-egj0y8Qs||...   0   0   1   1
'''

# As you see, We 'translate' type to 4 binary labels successfully
# 成功将type标签转换为4个二分类标签

'''
I = 0
N = 0
F = 0
J = 0
total = 0
for _ in data['type']:
    total += 1
    I += 1 if 'I' in _ else 0
    N += 1 if 'N' in _ else 0
    F += 1 if 'F' in _ else 0
    J += 1 if 'J' in _ else 0
print(I, N, F, J, total)
print(total - I, total - N, total - F, total - J)
'''

''' Trying to checkout the count of each labels
Introversion (I) / Extroversion (E):	 6766  /  1999
Intuition (N) / Sensing (S):		     7478  /  1197
Feeling (F) / Thinking (T):		         4694  /  3981
Judging (J) / Perceiving (P):	    	 3434  /  5241
# Also see 7.png for details
'''

'''
We use similar method to process the data, but add WordNetLemmatizer() now
What is WordNetLemmatizer, see the example below

wnl = WordNetLemmatizer()
# lemmatize nouns
print(wnl.lemmatize('cars', 'n'))
print(wnl.lemmatize('men', 'n'))
# lemmatize verbs
print(wnl.lemmatize('running', 'v'))
print(wnl.lemmatize('ate', 'v'))
# lemmatize adjectives
print(wnl.lemmatize('saddest', 'a'))
print(wnl.lemmatize('fancier', 'a'))

Output:
car
men
run
eat
sad
fancy

在数据处理上，我们对posts使用了相似的操作，但是添加了WordNetLemmatizer进行词形还原处理，示例见上
'''

lemmatiser = WordNetLemmatizer()

# Remove the stop words for speed (Same with Step1)
useless_words = stopwords.words("english")

# Remove these from the posts
unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
                    'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
unique_type_list = [x.lower() for x in unique_type_list]

# Splitting the MBTI personality into 4 letters and binarizing it
# 将 MBTI个性拆分为4个字母并对其进行二值化

b_Pers = {'I': 0, 'E': 1, 'N': 0, 'S': 1, 'F': 0, 'T': 1, 'J': 0, 'P': 1}
b_Pers_list = [{0: 'I', 1: 'E'}, {0: 'N', 1: 'S'}, {0: 'F', 1: 'T'}, {0: 'J', 1: 'P'}]


def translate_personality(personality):
    # transform mbti to binary vector
    return [b_Pers[l] for l in personality]


def translate_back(personality):
    # transform binary vector to mbti personality
    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s


def pre_process_text(data, remove_stop_words=True, remove_mbti_profiles=True):
    list_personality = []
    list_posts = []

    for row in data.iterrows():
        # Remove and clean comments
        posts = row[1].posts

        # Remove url links
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)

        # Remove Non-words
        temp = re.sub("[^a-zA-Z]", " ", temp)

        # Remove spaces > 1
        temp = re.sub(' +', ' ', temp).lower()

        # Remove multiple letter repeating words
        temp = re.sub(r'([a-z])\1{2,}[\s|\w]*', '', temp)

        # Remove stop words
        if remove_stop_words:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in useless_words])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])

        # Remove MBTI personality words from posts
        if remove_mbti_profiles:
            for t in unique_type_list:
                temp = temp.replace(t, "")

        # transform mbti to binary vector
        type_labelized = translate_personality(row[1].type)
        list_personality.append(type_labelized)
        # the cleaned data temp is passed here
        list_posts.append(temp)

    # returns the result
    list_posts = np.array(list_posts)
    list_personality = np.array(list_personality)
    return list_posts, list_personality


list_posts, list_personality = pre_process_text(data, remove_stop_words=True, remove_mbti_profiles=True)

''' Let us print the first piece of data to see how it works
Post before preprocessing:
 'http://www.youtube.com/watch?v=qsXHcwe3krw|||http://41.media.tumblr.com/tumblr_lfouy03PMA1qa1rooo1_500.jpg|||enfp 
 and intj moments  https://www.youtube.com/watch?v=iz7lE1g4XM4  sportscenter not top ten plays  
 https://www.youtube.com/watch?v=uCdfze1etec  pranks|||What has been the most life-changing experience in your 
 life?|||http://www.youtube.com/watch?v=vXZeYwwRDw8   http://www.youtube.com/watch?v=u8ejam5DP3E  On repeat for most 
 of today.|||May the PerC Experience immerse you.|||The last thing my INFJ friend posted on his facebook before 
 committing suicide the next day. Rest in peace~   http://vimeo.com/22842206|||Hello ENFJ7. Sorry to hear of your 
 distress. It's only natural for a relationship to not be perfection all the time in every moment of existence. Try 
 to figure the hard times as times of growth, as...|||84389  84390  
 http://wallpaperpassion.com/upload/23700/friendship-boy-and-girl-wallpaper.jpg  
 http://assets.dornob.com/wp-content/uploads/2010/04/round-home-design.jpg ...|||Welcome and 
 stuff.|||http://playeressence.com/wp-content/uploads/2013/08/RED-red-the-pokemon-master-32560474-450-338.jpg  Game. 
 Set. Match.|||Prozac, wellbrutin, at least thirty minutes of moving your legs (and I don't mean moving them while 
 sitting in your same desk chair), weed in moderation (maybe try edibles as a healthier alternative...|||Basically 
 come up with three items you've determined that each type (or whichever types you want to do) would more than likely 
 use, given each types' cognitive functions and whatnot, when left by...|||All things in moderation.  Sims is indeed 
 a video game, and a good one at that. Note: a good one at that is somewhat subjective in that I am not completely 
 promoting the death of any given Sim...|||Dear ENFP:  What were your favorite video games growing up and what are 
 your now, current favorite video games? :cool:|||https://www.youtube.com/watch?v=QyPqT8umzmY|||It appears to be too 
 late. :sad:|||There's someone out there for everyone.|||Wait... I thought confidence was a good thing.|||I just 
 cherish the time of solitude b/c i revel within my inner world more whereas most other time i'd be workin... just 
 enjoy the me time while you can. Don't worry, people will always be around to...|||Yo entp ladies... if you're into 
 a complimentary personality,well, hey.|||... when your main social outlet is xbox live conversations and even then 
 you verbally fatigue quickly.|||http://www.youtube.com/watch?v=gDhy7rdfm14  I really dig the part from 1:46 to 
 2:50|||http://www.youtube.com/watch?v=msqXffgh7b8|||Banned because this thread requires it of me.|||Get high in 
 backyard, roast and eat marshmellows in backyard while conversing over something intellectual, followed by massages 
 and kisses.|||http://www.youtube.com/watch?v=Mw7eoU3BMbE|||http://www.youtube.com/watch?v=4V2uYORhQOk|||http://www
 .youtube.com/watch?v=SlVmgFQQ0TI|||Banned for too many b's in that sentence. How could you! Think of the B!|||Banned 
 for watching movies in the corner with the dunces.|||Banned because Health class clearly taught you nothing about 
 peer pressure.|||Banned for a whole host of reasons!|||http://www.youtube.com/watch?v=IRcrv41hgz4|||1) Two baby deer 
 on left and right munching on a beetle in the middle.  2) Using their own blood, two cavemen diary today's latest 
 happenings on their designated cave diary wall.  3) I see it as...|||a pokemon world  an infj society  everyone 
 becomes an optimist|||49142|||http://www.youtube.com/watch?v=ZRCEq_JFeFM|||http://discovermagazine.com/2012/jul-aug
 /20-things-you-didnt-know-about-deserts/desert.jpg|||http://oyster.ignimgs.com/mediawiki/apis.ign.com/pokemon-silver
 -version/d/dd/Ditto.gif|||http://www.serebii.net/potw-dp/Scizor.jpg|||Not all artists are artists because they draw. 
 It's the idea that counts in forming something of your own... like a signature.|||Welcome to the robot ranks, 
 person who downed my self-esteem cuz I'm not an avid signature artist like herself. :proud:|||Banned for taking all 
 the room under my bed. Ya gotta learn to share with the 
 roaches.|||http://www.youtube.com/watch?v=w8IgImn57aQ|||Banned for being too much of a thundering, grumbling kind of 
 storm... yep.|||Ahh... old high school music I haven't heard in ages.   
 http://www.youtube.com/watch?v=dcCRUPCdB1w|||I failed a public speaking class a few years ago and I've sort of 
 learned what I could do better were I to be in that position again. A big part of my failure was just overloading 
 myself with too...|||I like this person's mentality. He's a confirmed INTJ by the way. 
 http://www.youtube.com/watch?v=hGKLI-GEc6M|||Move to the Denver area and start a new life for myself.' 

Post after preprocessing: 
moment sportscenter top ten play prank life changing experience life repeat today may perc 
experience immerse last thing  friend posted facebook committing suicide next day rest peace hello  sorry hear 
distress natural relationship perfection time every moment existence try figure hard time time growth welcome stuff 
game set match prozac wellbrutin least thirty minute moving leg mean moving sitting desk chair weed moderation maybe 
try edible healthier alternative basically come three item determined type whichever type want would likely use given 
type cognitive function whatnot left thing moderation sims indeed video game good one note good one somewhat 
subjective completely promoting death given sim dear favorite video game growing current favorite video game cool 
appears late sad someone everyone wait thought confidence good thing cherish time solitude b c revel within inner 
world whereas time workin enjoy time worry people always around yo lady complimentary personality well hey main 
social outlet xbox live conversation even verbally fatigue quickly really dig part banned thread requires get high 
backyard roast eat marshmellows backyard conversing something intellectual followed massage kiss banned many b 
sentence could think b banned watching movie corner dunce banned health class clearly taught nothing peer pressure 
banned whole host reason two baby deer left right munching beetle middle using blood two caveman diary today latest 
happening designated cave diary wall see pokemon world  society everyone becomes optimist artist artist draw idea 
count forming something like signature welcome robot rank person downed self esteem cuz avid signature artist like 
proud banned taking room bed ya gotta learn share roach banned much thundering grumbling kind storm yep ahh old high 
school music heard age failed public speaking class year ago sort learned could better position big part failure 
overloading like person mentality confirmed  way move denver area start new life 

MBTI before preprocessing:
 INFJ

MBTI after preprocessing:
 [0 0 0 0]
'''

# 在Step1的基础上，新增了TF-IDF算法
cntizer = CountVectorizer(analyzer="word",
                          max_features=1000,
                          max_df=0.7,
                          min_df=0.1)
X_cnt = cntizer.fit_transform(list_posts)
tfizer = TfidfTransformer()
X_tfidf = tfizer.fit_transform(X_cnt).toarray()

''' Output >> X_tfidf[0]
[0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.08105478 0.07066064
 0.         0.         0.         0.         0.         0.
 0.         0.04516864 0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.05321691 0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.0871647  0.         0.         0.
 0.         0.         0.         0.05506308 0.0708757  0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.16585935 0.         0.         0.09676192 0.
 0.         0.04970682 0.         0.         0.         0.
 0.07397056 0.         0.         0.         0.         0.
 0.         0.0748045  0.07639898 0.09185775 0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.05133662 0.         0.09442732
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.09657087 0.         0.         0.         0.
 0.         0.         0.         0.07062049 0.         0.
 0.         0.         0.04405493 0.         0.05892624 0.11838033
 0.         0.         0.         0.         0.1245151  0.
 0.         0.         0.         0.         0.         0.
 0.         0.15886654 0.         0.         0.         0.
 0.         0.         0.         0.08435344 0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.04471932
 0.         0.         0.         0.07063387 0.         0.
 0.29304485 0.         0.         0.         0.         0.
 0.         0.18141448 0.         0.         0.         0.
 0.         0.12564763 0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.05834299 0.         0.         0.08003112
 0.08435344 0.         0.         0.09115444 0.         0.08189961
 0.         0.13411106 0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.05631577 0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.05557735 0.         0.
 0.         0.         0.         0.         0.0611348  0.09448648
 0.         0.         0.         0.         0.         0.
 0.07930797 0.09349615 0.         0.06379457 0.         0.17160741
 0.         0.         0.         0.13427658 0.         0.
 0.08006774 0.         0.         0.         0.         0.
 0.         0.06885442 0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.09793451 0.         0.
 0.         0.         0.05268964 0.         0.05993985 0.05477763
 0.         0.04997463 0.         0.         0.         0.
 0.         0.         0.09867431 0.         0.         0.
 0.09487394 0.         0.15341726 0.         0.         0.
 0.         0.         0.08813151 0.07690585 0.07177948 0.
 0.         0.09843694 0.         0.         0.         0.
 0.         0.05941035 0.08188014 0.         0.         0.
 0.         0.         0.09612974 0.06395483 0.         0.
 0.         0.         0.         0.         0.         0.
 0.06973746 0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.12067215
 0.         0.         0.         0.         0.         0.09674118
 0.         0.0615769  0.         0.         0.         0.
 0.         0.         0.         0.07446814 0.         0.
 0.         0.         0.         0.08993396 0.         0.
 0.         0.         0.         0.         0.09901671 0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.05946565 0.         0.         0.         0.05873657 0.
 0.         0.         0.         0.09634949 0.         0.04805982
 0.         0.08968056 0.         0.         0.0860263  0.
 0.         0.         0.         0.06318282 0.         0.
 0.         0.04256832 0.         0.         0.         0.
 0.06642087 0.         0.         0.         0.09201473 0.
 0.         0.0831116  0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.06971149
 0.09554125 0.04625983 0.08531558 0.         0.         0.
 0.06799661 0.07466644 0.         0.         0.         0.09843694
 0.         0.         0.         0.06502346 0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.06869092 0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.08067849 0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.0466968  0.0539756  0.08760887 0.
 0.1533845  0.         0.         0.         0.         0.09298479
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.10580777 0.
 0.         0.11100899 0.13361762 0.         0.         0.
 0.06046932 0.         0.08258902 0.         0.         0.2392193
 0.         0.09217882 0.         0.04223906 0.         0.
 0.08665848 0.04111178 0.         0.         0.16434695 0.04117693
 0.         0.         0.         0.07231244 0.         0.
 0.         0.         0.         0.         0.         0.
 0.11467609 0.09387096 0.         0.         0.         0.
 0.         0.         0.04833236 0.         0.         0.
 0.        ]
 
 We transform posts into vector matrix successfully
 我们成功的将posts转化为了词向量矩阵
'''

personality_type = ["IE: Introversion (I) / Extroversion (E)",
                    "NS: Intuition (N) / Sensing (S)",
                    "FT: Feeling (F) / Thinking (T)",
                    "JP: Judging (J) / Perceiving (P)"]

X = X_tfidf
Y = list_personality

'''
We translate Y into 16 labels first for a better train/test data quality.
'''
label = []
for _ in range(len(Y)):
    lb = 0
    muti = 8
    for i in range(4):
        lb += muti if Y[_][i] == 1 else 0
        muti /= 2
    label.append(int(lb))

Y = label

# 生成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, stratify=Y, random_state=1)


def pred_res_process(res, model):
    # ENFJ ENFP ENTJ ENTP
    # ESFJ ESFP ESTJ ESTP
    # INFJ INFP INTJ INTP
    # ISFJ ISFP ISTJ ISTP
    # INFJ -> 0000
    pred_chart = [[0 for _ in range(16)] for __ in range(16)]
    seq = [8, 9, 10, 11,
           12, 13, 14, 15,
           0, 1, 2, 3,
           4, 5, 6, 7]
    test_data = []
    pred_data = []
    for _ in range(len(res[0])):
        test_data.append([res[i][_][0] for i in range(4)])
        pred_data.append([res[i][_][1] for i in range(4)])
    for _ in range(len(res[0])):
        lb_t = 0
        lb_p = 0
        muti = 8
        for i in range(4):
            lb_t += muti if test_data[_][i] else 0
            lb_p += muti if pred_data[_][i] else 0
            muti /= 2
        x = 0
        y = 0
        for __ in range(16):
            if seq[__] == lb_p:
                x = __
            if seq[__] == lb_t:
                y = __
        pred_chart[x][y] += 1
    print(pred_chart)
    pd.DataFrame.to_csv(DataFrame(pred_chart), f'{model}..csv', index=False, header=False)

    '''
    for _ in range(16):
        temp = []
        for _ in range(16):
            temp.append(0)
        pred_chart.append(temp)
    seq = [8, 9, 10, 11,
           12, 13, 14, 15,
           0, 1, 2, 3,
           4, 5, 6, 7]
    for _ in range(int(len(res) / 8)):
        test = []
        pred = []
        for __ in range(4):
            test.append(res[int(2 * _ + (__ * len(res) / 8))])
            pred.append(res[int(2 * _ + (__ * len(res) / 8)) + 1])
        print(test)
        _test = 0
        _pred = 0
        muti = 8
        for __ in range(4):
            _test += muti if test[__] else 0
            _pred += muti if pred[__] else 0
            muti /= 2
        x = 0
        y = 0
        for __ in range(16):
            if seq[__] == _pred:
                x = __
            if seq[__] == _test:
                y = __
        pred_chart[x][y] += 1
    pd.DataFrame.to_csv(DataFrame(pred_chart), f'{model}..csv', index=False, header=False)'''


def decode(y_piece, state):
    res = []
    for _ in range(4):
        if y_piece % 2 == 0:
            res.append(0)
        else:
            y_piece -= 1
            res.append(1)
        y_piece /= 2
    return list(reversed(res))[state]


# Random Forest model for MBTI dataset

result = []
for l in range(len(personality_type)):
    ytr = []
    yte = []

    for _ in range(len(y_train)):
        ytr.append(decode(y_train[_], l))

    for _ in range(len(y_test)):
        yte.append(decode(y_test[_], l))

    # fit model on training data
    # exec()
    model = XGBClassifier()
    model.fit(X_train, ytr)

    # make predictions for test data
    y_pred = model.predict(X_test)

    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(yte, predictions)
    result.append([[yte[_], predictions[_]] for _ in range(len(yte))])
    print("%s Accuracy: %.2f%%" % (personality_type[l], accuracy * 100.0))
    print(f"F1 score:{f1_score(yte, predictions, labels=None, pos_label=1, average='binary', sample_weight=None)}")
pred_res_process(result, 'RF')
