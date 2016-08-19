import pandas as pd
import numpy as np

import matplotlib
from matplotlib.colors import ListedColormap
import seaborn as sns
sns.set()
# Save a nice dark grey as a variable
almost_black = '#262626'

import itertools
from unbalanced_dataset import *
from sklearn.decomposition import *
from sklearn import preprocessing, tree, neighbors, cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import classification_report, accuracy_score
# from sklearn.preprocessing import label_binarize
from sklearn.externals.six import StringIO

# from IPython.display import Image
import pprint
import pydot 
import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')

from time import time

#################################################
# Data Preparation and Setup
# Import the data and declare the properties
#################################################

filepath = 'data/full_data_genre.csv'

header = None
header_included = True

if header_included:
    header = 0

# different sets of features
genres = ['country', 'dance', 'hip_hop', 'pop', 'r&b', 'rock', 'alternative']
accoustic = ['key', 'energy', 'liveness', 'tempo', 'speechiness', 'acousticness', 'instrumentalness',
             'danceability', 'time_signature', 'loudness', 'duration', 'mode']
artist = ['artist_familiarity', 'artist_hottness']

# feature properties, 0: numerical, 1: categorical
genres_types = [1,1,1,1,1,1,1]
accoustic_types = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
artist_types = [0, 0]

# features used in prediction
feature_names = accoustic + genres + artist
feature_num = len(feature_names)
feature_types = accoustic_types + genres_types + artist_types 

# predict if the song is in billboard or not
pred_name = 'billboard'
# the percentage of entire dataset using for test dataset
test_percent = 0.3 

df = pd.read_csv(filepath, header = header)
df['billboard'] = df['weeks'].map(lambda x: x != 0)

raw_X = np.array(df[feature_names])
raw_Y = np.array(df[pred_name].map(lambda x: int(x)).tolist())

target_names_bool = list(set(raw_Y.tolist()))


# ## Another test dataset that is not included in original dataset

new = pd.read_csv('data/new_song.csv', header = header)
new_raw_X = np.array(new[feature_names])
new_raw_Y = np.array(new[pred_name].map(lambda x: int(x)).tolist())


#################################################
# Classification report options
#################################################

class Options(object):
    pass

opts = Options()

opts.DTree = False
# opts.lsvc = False

# Save decision tree to pdf
opts.save_DTree = False

# Print a detailed classification report.
opts.print_report = True
# Select some number of features using a chi-squared test
opts.select_chi2 = 3
# Print the confusion matrix.
opts.print_cm = False
# Print normalized confusion matrix.
opts.print_norm_cm = True

#################################################
# Other functions and class
#################################################

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    
    plt.figure(figsize=(5, 5))
 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.show()

class pilot:
    def __init__(self, x, y, dataset_split_random_state = None):
        self.X_train, self.X_test, self.Y_train, self.Y_test =         cross_validation.train_test_split(x, y,                                           random_state                                           = dataset_split_random_state)
    
    def get_dataset(self):
        return self.X_train, self.X_test, self.Y_train, self.Y_test
    
    def benchmark(self,clf):
        t0 = time()
        clf = clf.fit(self.X_train, self.Y_train)
        train_time = time() - t0

        t0 = time()
        Y_predict = clf.predict(self.X_test)
        test_time = time() - t0

        score = accuracy_score(self.Y_test, Y_predict)
        cm = confusion_matrix(self.Y_test, Y_predict)
        report = classification_report(self.Y_test,Y_predict,
                                       target_names = target_names)
        
        print('_' * 80)
        print("Training: ")
        print(clf)
        print()
        print("train time: %0.3fs" % train_time)
        print("test time:  %0.3fs" % test_time)
        print("accuracy:   %0.3f" % score)
        print()
        if opts.print_report:
            print("Classification report:")
            print(report)

        if opts.print_cm:
            plot_confusion_matrix(cm)

        if opts.print_norm_cm:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            plot_confusion_matrix(cm_normalized, 
                                  title='Normalized confusion matrix')

        if opts.DTree and opts.save_DTree:
            opts.DTree = False
            # save dot file
            with open("data/DTree.dot", 'w') as f:
                f = tree.export_graphviz(clf, out_file=f)
            # save to pdf
            dot_data = StringIO() 
            tree.export_graphviz(clf, out_file=dot_data,  
                                 feature_names=feature_names,  
                                 class_names=target_names,  
                                 filled=True)  
            graph = pydot.graph_from_dot_data(dot_data.getvalue())  
            graph.write_pdf("data/DTree.pdf")
            
        return clf, score, report, train_time, test_time
    
    def DTree(self, **params): # **params set parameters for classifier
        
        clf = tree.DecisionTreeClassifier()

        default_params = clf.get_params(deep=True)
        if params!= {}:
            for key in params:
                if key in default_params:
                    default_params[key] = params[key]

            clf.set_params(class_weight = default_params['class_weight'], 
                           criterion = default_params['criterion'], 
                           max_depth = default_params['max_depth'], 
                           max_leaf_nodes = default_params['max_leaf_nodes'], 
                           min_samples_leaf = default_params['min_samples_leaf'], 
                           min_weight_fraction_leaf \
                           = default_params['min_weight_fraction_leaf'],
                           presort = default_params['presort'], 
                           random_state = default_params['random_state'],
                           splitter = default_params['splitter'])
        
        opts.DTree = True
        return self.benchmark(clf)
    
    
    def KNN(self, **params): # **params set parameters for classifier
        clf = neighbors.KNeighborsClassifier()
        default_params = clf.get_params(deep=True)
        if params!= {}:
            for key in params:
                if key in default_params:
                    default_params[key] = params[key]
                    
        clf.set_params(algorithm = default_params['algorithm'], 
                       leaf_size = default_params['leaf_size'], 
                       metric = default_params['metric'],
                       metric_params = default_params['metric_params'], 
                       n_jobs=default_params['n_jobs'],
                       n_neighbors = default_params['n_neighbors'],
                       p = default_params['p'],
                       weights = default_params['weights'])
        
        return self.benchmark(clf)
    
    def GaussianNaiveBayes(self):
        gnb = GaussianNB()
        return self.benchmark(gnb)
    
    def linearSVC(self, **params):
        lsvc = LinearSVC()
        default_params = lsvc.get_params(deep=True)
        if params!= {}:
            for key in params:
                if key in default_params:
                    default_params[key] = params[key]
                    
        lsvc.set_params(penalty = default_params['penalty'], 
                        loss = default_params['loss'], 
                        dual = default_params['dual'],
                        tol = default_params['tol'], 
                        C=default_params['C'], 
                        multi_class = default_params['multi_class'],
                        fit_intercept = default_params['fit_intercept'],
                        intercept_scaling = default_params['intercept_scaling'],
                        class_weight = default_params['class_weight'],
                        verbose = default_params['verbose'],
                        random_state = default_params['random_state'],
                        max_iter = default_params['max_iter'])
    
        return self.benchmark(lsvc)
    
    def RandomForest(self, **params):
        rfc = RandomForestClassifier()
        return self.benchmark(rfc)


#################################################
# Data normalization:
#  - numerical data: normalize to range [0, 1]
#  - categorical data: label with 0 to k-1
#################################################

def data_norm(x, y, fnames, ftypes, tnames):
    new_x = np.array(x)

    fnum = len(fnames)
    numerical_features = [fnames[i] for i in range(fnum) if ftypes[i] == 0]
    categorical_features = [fnames[i] for i in range(fnum) if ftypes[i] == 1]

    le = preprocessing.LabelEncoder()

    for i in range(fnum):
        if ftypes[i] == 0:
            new_x[:, i] = preprocessing.minmax_scale(x[:, i],
                                                     feature_range=(-1, 1),
                                                     axis=0, copy=True)
        elif ftypes[i] == 1:
            le.fit(list(set(x[:, i])))
            new_x[:, i] = le.transform(x[:, i]) 

    le.fit(tnames)
    new_y = le.transform(y)
    new_tnames = list(map(str,tnames))
    
    return new_x, new_y, new_tnames


#################################################
# Plot data distribution
#################################################

# Instanciate a PCA object for the sake of easy visualisation
pca = PCA(n_components=2)

# Fit and transform x to visualise inside a 2D feature space
x_vis = pca.fit_transform(raw_X)
                          
# Plot the original data
# Plot the two classes
palette = sns.color_palette()
plt.scatter(x_vis[raw_Y == 0, 0], x_vis[raw_Y == 0, 1], label="Class #0", alpha=0.5,
            edgecolor=almost_black, facecolor=palette[0], linewidth=0.15)
plt.scatter(x_vis[raw_Y == 1, 0], x_vis[raw_Y == 1, 1], label="Class #1", alpha=0.5,
            edgecolor=almost_black, facecolor=palette[2], linewidth=0.15)

plt.legend()
plt.show()

#################################################
# Imbalanced dataset
#################################################

X_norm, Y_norm, target_names = data_norm(x=raw_X, y=raw_Y, 
          fnames=feature_names, 
          ftypes=feature_types, 
          tnames=target_names_bool)

new_X_norm, new_Y_norm, target_names = data_norm(x=new_raw_X, 
          y=new_raw_Y,
          fnames=feature_names, 
          ftypes=feature_types, 
          tnames=target_names_bool)

#################################################
# Training models and results without resampling
#################################################

example = pilot(X_norm,Y_norm)

tree_result = example.DTree(min_samples_leaf = 25)

print('prediction:', tree_result[0].predict(new_X_norm))
print('prediction prob:', tree_result[0].predict_proba(new_X_norm,
                                                       check_input=True))

knn_result = example.KNN(n_neighbors = 7)

print('prediction:', knn_result[0].predict(new_X_norm))
print('prediction prob:', knn_result[0].predict_proba(new_X_norm))

gnb_result = example.GaussianNaiveBayes()

print('prediction:', gnb_result[0].predict(new_X_norm))
try:
    print('prediction prob:', gnb_result[0].predict_proba(new_X_norm))
except:
    pass

lsvc_result = example.linearSVC()

print('prediction:', lsvc_result[0].predict(new_X_norm))
try:
    print('prediction prob:', lsvc_result[0].predict_proba(new_X_norm))
except:
    pass

lsvc_result2 = example.linearSVC(class_weight='balanced')

print('prediction:', lsvc_result2[0].predict(new_X_norm))
try:
    print('prediction prob:', lsvc_result2[0].predict_proba(new_X_norm))
except:
    pass

lsvc_result3 = example.linearSVC(class_weight='balanced',
                                 penalty='l1', dual=False)

print('prediction:', lsvc_result3[0].predict(new_X_norm))
try:
    print('prediction prob:', lsvc_result3[0].predict_proba(new_X_norm))
except:
    pass


rfc_result = example.RandomForest()

print('prediction:', rfc_result[0].predict(new_X_norm))
print('prediction prob:', rfc_result[0].predict_proba(new_X_norm))

# #################################################
#  Under-Sampling
#################################################

# Random UnderSampling

sampler = UnderSampler(ratio = 1)
X_under, Y_under = sampler.fit_transform(X_norm,Y_norm)

example = pilot(X_under,Y_under)

tree_result = example.DTree(min_samples_leaf = 25)

print('prediction:', tree_result[0].predict(new_X_norm))
print('prediction prob:', tree_result[0].predict_proba(new_X_norm))

knn_result = example.KNN(n_neighbors = 7)

print('prediction:', knn_result[0].predict(new_X_norm))
try:
    print('prediction prob:', knn_result[0].predict_proba(new_X_norm))
except:
    pass

gnb_result = example.GaussianNaiveBayes()

print('prediction:', gnb_result[0].predict(new_X_norm))
try:
    print('prediction prob:', gnb_result[0].predict_proba(new_X_norm))
except:
    pass

lsvc_result = example.linearSVC()

print('prediction:', lsvc_result[0].predict(new_X_norm))
try:
    print('prediction prob:', lsvc_result[0].predict_proba(new_X_norm))
except:
    pass

rfc_result = example.RandomForest()

print('prediction:', rfc_result[0].predict(new_X_norm))
try:
    print('prediction prob:', rfc_result[0].predict_proba(new_X_norm))
except:
    pass

# NearMiss

sampler = NearMiss()
X_nm, Y_nm = sampler.fit_transform(X_norm,Y_norm)

example = pilot(X_nm,Y_nm)

tree_result = example.DTree(min_samples_leaf = 25)

print('prediction:', tree_result[0].predict(new_X_norm))
try:
    print('prediction prob:', tree_result[0].predict_proba(new_X_norm))
except:
    pass

knn_result = example.KNN(n_neighbors = 7)

print('prediction:', knn_result[0].predict(new_X_norm))
try:
    print('prediction prob:', knn_result[0].predict_proba(new_X_norm))
except:
    pass

gnb_result = example.GaussianNaiveBayes()

print('prediction:', gnb_result[0].predict(new_X_norm))
try:
    print('prediction prob:', gnb_result[0].predict_proba(new_X_norm))
except:
    pass

lsvc_result = example.linearSVC()

print('prediction:', lsvc_result[0].predict(new_X_norm))
try:
    print('prediction prob:', lsvc_result[0].predict_proba(new_X_norm))
except:
    pass

rfc_result = example.RandomForest()

print('prediction:', rfc_result[0].predict(new_X_norm))
try:
    print('prediction prob:', rfc_result[0].predict_proba(new_X_norm))
except:
    pass

#################################################
# Over-Sampling
#################################################

# Random OverSampling

sampler = OverSampler(ratio = 4)
X_os, Y_os = sampler.fit_transform(X_norm,Y_norm)

example = pilot(X_os,Y_os)

tree_result = example.DTree(min_samples_leaf = 25)

print('prediction:', tree_result[0].predict(new_X_norm))
try:
    print('prediction prob:', tree_result[0].predict_proba(new_X_norm))
except:
    pass

knn_result = example.KNN(n_neighbors = 7)

print('prediction:', knn_result[0].predict(new_X_norm))
try:
    print('prediction prob:', knn_result[0].predict_proba(new_X_norm))
except:
    pass

gnb_result = example.GaussianNaiveBayes()

print('prediction:', gnb_result[0].predict(new_X_norm))
try:
    print('prediction prob:', gnb_result[0].predict_proba(new_X_norm))
except:
    pass

lsvc_result = example.linearSVC()

print('prediction:', lsvc_result[0].predict(new_X_norm))
try:
    print('prediction prob:', lsvc_result[0].predict_proba(new_X_norm))
except:
    pass

rfc_result = example.RandomForest()

print('prediction:', rfc_result[0].predict(new_X_norm))
try:
    print('prediction prob:', rfc_result[0].predict_proba(new_X_norm))
except:
    pass

# SMOTE

sampler = SMOTE(ratio = 4)
X_smote, Y_smote = sampler.fit_transform(X_norm,Y_norm)

example = pilot(X_smote,Y_smote)

tree_result = example.DTree(min_samples_leaf = 25)

print('prediction:', tree_result[0].predict(new_X_norm))
try:
    print('prediction prob:', tree_result[0].predict_proba(new_X_norm))
except:
    pass

knn_result = example.KNN(n_neighbors = 7)

print('prediction:', knn_result[0].predict(new_X_norm))
try:
    print('prediction prob:', knn_result[0].predict_proba(new_X_norm))
except:
    pass

gnb_result = example.GaussianNaiveBayes()

print('prediction:', gnb_result[0].predict(new_X_norm))
try:
    print('prediction prob:', gnb_result[0].predict_proba(new_X_norm))
except:
    pass

lsvc_result = example.linearSVC()

print('prediction:', lsvc_result[0].predict(new_X_norm))
try:
    print('prediction prob:', lsvc_result[0].predict_proba(new_X_norm))
except:
    pass

rfc_result = example.RandomForest()

print('prediction:', rfc_result[0].predict(new_X_norm))
try:
    print('prediction prob:', rfc_result[0].predict_proba(new_X_norm))
except:
    pass

