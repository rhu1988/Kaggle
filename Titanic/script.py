# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
titanic_train=pd.read_csv('../input/train.csv')

#print(titanic_train.shape)


#Embarked list

Embarked_list=titanic_train['Embarked']
#print(type(Embarked_list))
#print(Embarked_list.astype('category'))
#print(Embarked_list[0])
Embarked=[]
for i in range(len(Embarked_list)):
    if Embarked_list[i]=='C':
        Embarked.append(1)
    elif Embarked_list[i]=='Q':
        Embarked.append(2)
    elif Embarked_list[i]=='S':
        Embarked.append(3)
    else:
        Embarked.append(0)
#print(Embarked)
#print(len(Embarked))

#Survived list

Survived_list=titanic_train['Survived']
Survived=[]
for i in range(len(Survived_list)):
    Survived.append(Survived_list[i])
#print(Survived)
#print(len(Survived))

#Pclass list

Pclass_list=titanic_train['Pclass']
Pclass=[]
for i in range(len(Pclass_list)):
    Pclass.append(Pclass_list[i])
#print(Pclass)
#print(len(Pclass))

#Sex list

Sex_list=titanic_train['Sex']
Sex=[]
for i in range(len(Sex_list)):
    if Sex_list[i]=='male':
        Sex.append(1)
    elif Sex_list[i]=='female':
        Sex.append(2)
    else:
        Sex.append(0)
#print(Sex)
#print(len(Sex))

#Age list

Age_list=titanic_train['Age']#.fillna(value=0)
Age=[]
for i in range(len(Age_list)):
    Age.append(Age_list[i])
#print(Age)
#print(len(Age))
#print(Age_list[19])

#SibSp list

SibSp_list=titanic_train['SibSp']
SibSp=[]
for i in range(len(SibSp_list)):
    SibSp.append(SibSp_list[i])
#print(SibSp)
#print(len(SibSp))

#Parch list

Parch_list=titanic_train['Parch']
Parch=[]
for i in range(len(Parch_list)):
    Parch.append(Parch_list[i])
#print(Parch)
#print(len(Parch))

#Fare list

Fare_list=titanic_train['Fare']
Fare=[]
for i in range(len(Fare_list)):
    Fare.append(Fare_list[i])
#print(Fare)
#print(len(Fare))

#survived_sample

survived_sample=[]
for i in range(len(Fare_list)):
    survived_sample.append([Survived[i]])
survived_sample=np.array(survived_sample)
#print(survived_sample)

#train_sample

train_sample=[]
for i in range(len(Fare_list)):
    train_sample.append([1,Pclass[i],Sex[i],Age[i],SibSp[i],Parch[i],Fare[i],Embarked[i],survived_sample[i]])
#print(train_sample)
train_sample=np.array(train_sample)
#print(train_sample)
#print(train_sample.shape)

train=pd.DataFrame(train_sample).dropna(axis=0,how='any')
#print(train.shape)
#print(train)

survived_sample_2=train[8]
#print(survived_sample_2)
survived_sample_array=survived_sample_2.as_matrix(columns=None)
#print(survived_sample_array)

survived_sample_column=[]
for i in range(len(survived_sample_array)):
    survived_sample_column.append([survived_sample_array[i]])
survived_sample_column=np.array(survived_sample_column)
#print(survived_sample_column)

train=train.as_matrix(columns=[0,1,2,3,4,5,6,7])
#print(train)



W=tf.Variable(tf.random_normal([8,1],stddev=1/np.sqrt(8)))
#W=tf.Variable(np.array([[1],[1],[1],[1],[1],[1],[1],[1]],dtype='float32'))
x=tf.placeholder(tf.float32,[len(train),8])
y_=tf.placeholder(tf.float32,[len(train),1])

#a=tf.sigmoid(tf.matmul(x,W))
z=tf.matmul(x,W)
cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=z,labels=y_))
#cost=tf.reduce_mean(-y_*tf.log(a)-(1-y_)*tf.log(1-a))

regularizer=tf.nn.l2_loss(W)
cost=tf.reduce_mean(cost+0.01*regularizer)


train_step=tf.train.GradientDescentOptimizer(1e-4).minimize(cost)

sess=tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(100000):
    #idx=np.random.choice(len(train),20,replace=False)
    _,l=sess.run([train_step,cost],feed_dict={x:train,y_:survived_sample_column})
    if i%10000==0:
        print('loss:'+str(l))
print(sess.run(W))
#sess.run([train_step,cost],feed_dict={x:train_sample,y_:survived_sample})
w=sess.run(W)
#print(np.dot(train[8],w))
def sigmoid(x,W):
    Wx=np.dot(x,W)
    return 1/(1+np.exp(-Wx))
print(sigmoid(train[8],w))

#test
titanic_test=pd.read_csv('../input/test.csv')

#test_embarked_list

test_embarked_list=titanic_test['Embarked']
Embarked=[]
for i in range(len(test_embarked_list)):
    if test_embarked_list[i]=='C':
        Embarked.append(1)
    elif test_embarked_list[i]=='Q':
        Embarked.append(2)
    elif test_embarked_list[i]=='S':
        Embarked.append(3)
    else:
        Embarked.append(0)
        
#test_Pclass_list

test_Pclass_list=titanic_test['Pclass']
Pclass=[]
for i in range(len(test_Pclass_list)):
    Pclass.append(test_Pclass_list[i])
    
#test_Sex_list

test_Sex_list=titanic_test['Sex']
Sex=[]
for i in range(len(test_Sex_list)):
    if test_Sex_list[i]=='male':
        Sex.append(1)
    elif test_Sex_list[i]=='female':
        Sex.append(2)
    else:
        Sex.append(0)
        
#test_Age_list

test_Age_list=titanic_test['Age'].fillna(value=0)
Age=[]
for i in range(len(test_Age_list)):
    Age.append(test_Age_list[i])
    
#test_SibSp list

test_SibSp_list=titanic_test['SibSp']
SibSp=[]
for i in range(len(test_SibSp_list)):
    SibSp.append(test_SibSp_list[i])

#test_Parch_list

test_Parch_list=titanic_test['Parch']
Parch=[]
for i in range(len(test_Parch_list)):
    Parch.append(test_Parch_list[i])
    
#test_Fare list

test_Fare_list=titanic_test['Fare']
Fare=[]
for i in range(len(test_Fare_list)):
    Fare.append(test_Fare_list[i])
    
#test_sample

test_sample=[]
for i in range(len(test_Fare_list)):
    test_sample.append([1,Pclass[i],Sex[i],Age[i],SibSp[i],Parch[i],Fare[i],Embarked[i]])
#print(train_sample)
test_sample=np.array(test_sample)

test_survived=sigmoid(test_sample,w)
print(test_survived)