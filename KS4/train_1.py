import sys
import datetime
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def now():
        return str(datetime.datetime.now())

# Load dataset
url = "~/Dropbox/RandomBergman/KS4/data.csv"
#firstbins = ['0','1','2','3','01','02','03','12','13','23','001','101','002','202','003','303','112','212','113','313','223','323']
#hs = ['h0','h1','h2','h3','h4','h5','h6']
#names = []
#for b in firstbins:
#        for h in hs:
#                names.append(b+h)
#names.append('nvs')
#names.append('class')
#print names
#print len(names)
if 'dataset' not in vars():
        print 'begin loading csv', now()
        dataset = pandas.read_csv(url, names=['Ginf','Go','share2cone','Vinf','Vo','Vrat','Graphd','Crat'])
        print 'end loading csv', now()

if 'shuffled' not in vars():
        from sklearn.utils import shuffle
        dataset = shuffle(dataset)
        shuffled = True
        group_map = {"E8": 1, "E7": 2, "E6": 3, "F4": 4, "G2": 5, "SO8": 6, "SO7": 7, "S3": 8, "S2": 9 }
        # for idx, d in enumerate(dataset['Ginf']):
        #         print idx, d
        #         dataset['Ginf'][idx] = group_map[d]
        
# shape
print(dataset.shape)

# head
print(dataset.head(20))

# descriptions
print(dataset.describe())

# Crat distribution
# print(dataset.groupby('Crat').size())

# box and whisker plots
# dataset.plot(kind='box', subplots=True, layout=(7,7), sharex=False, sharey=False)
# plt.show()

# histograms
# dataset.hist()
# plt.show()

# # scatter plot matrix
# #scatter_matrix(dataset)
# #plt.show()

# # Split-out validation dataset
array = dataset.values
X = array[:,0:5]
Y = array[:,5]
validation_size = 0.10
seed=10
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
        
# # Test options and evaluation metric
# seed = 11
# scoring = 'accuracy
def mean_absolute_percentage_error(y_true, y_pred): 
  return numpy.mean(numpy.abs((y_true - y_pred) / y_true)) * 100
from sklearn.metrics import make_scorer
scoring = make_scorer(mean_absolute_percentage_error)

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
# #models.append(('SVM', SVC()))

# # evaluate each model in turn
results = []
names = []
for name, model in models:
       print name, str(datetime.datetime.now())
       kfold = model_selection.KFold(n_splits=10, random_state=seed)
       cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
       results.append(cv_results)
       names.append(name)
       msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
       print(msg)

# for i in range(1,10):
#         ds = dataset.truncate(len(dataset)-4000*i)
#         results, names = [], []
#         array = ds.values
#         X = array[:,0:155]
#         Y = array[:,155]
#         validation_size = 0.20
#         seed=10
#         X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#         for name, model in models:
#                 print name, i, len(ds)
#                 kfold = model_selection.KFold(n_splits=10,random_state=seed)
#                 cv_results = model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
#                 msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#                 print msg, len(dataset)-4000*i

# # Compare Algorithms
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()

# # Make predictions on validation dataset
# lor = LogisticRegression()
# print 'begin logistic regression fit'
# lor.fit(X_train, Y_train)
# print 'end logistic regression fit'
# predictions = lor.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))
