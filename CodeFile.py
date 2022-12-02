import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import io
from scipy import misc
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve
a=pd.read_csv('Employee.csv')
df = pd.DataFrame(a)

# Renaming certain columns for better readability
df = df.rename(columns={'satisfaction_level': 'satisfaction',
                        'last_evaluation': 'evaluation',
                        'number_project': 'projectCount',
                        'average_montly_hours': 'averageMonthlyHours',
                        'time_spend_company': 'yearsAtCompany',
                        'Work_accident': 'workAccident',
                        'promotion_last_5years': 'promotion',
                        'dept' : 'department',
                        'Left' : 'turnover'
                        })

# Convert these variables into categorical variables
df["department"] = df["department"].astype('category').cat.codes
df["salary"] = df["salary"].astype('category').cat.codes
front = df['turnover']
df.drop(labels=['turnover'], axis=1,inplace = True)
df.insert(0, 'turnover', front)

f_cols = ['satisfaction','evaluation','projectCount','averageMonthlyHours','yearsAtCompany','workAccident','promotion','department','salary']
x = df[f_cols]
y = df.turnover

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names = f_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Decision_Tree.png')
Image(graph.create_png())










'''
def show_tree(tree,features,path):
    f=io.StringIO()
    export_graphviz(tree,out_file=f,feature_names=features)
    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
    img =misc.imread(path)
    plt.rcParams["figure.figsize"]=(20,20)
    plt.imshow(img)


show_tree(clf,f_cols,'omnamoom.png')

print(y_pred)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)

pos= df[df ['turnover']== 'Yes']['projectCount']
neg= df[df ['turnover']== 'No']['projectCount']
fig=plt.figure(figsize=(12,8))
pos.hist(alpha=0.7,bins=30,label='positive').get_figure().savefig('omom.png')
neg.hist(alpha=0.7,bins=30,label='negative').get_figure().savefig('shri.png')

# Create an intercept term for the logistic regression equation
df['int'] = 1
indep_var = ['satisfaction', 'evaluation', 'yearsAtCompany', 'int', 'turnover']
df = df[indep_var]

# Create train and test splits
target_name = 'turnover'
X = df.drop('turnover', axis=1)

y=df[target_name]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)

X_train.head()
#satisfaction	evaluation	yearsAtComp

'''
