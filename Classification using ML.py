
# Importing the libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import time

# Making a list of missing value types
missing_values = ["n/a", "na", "Infinity"]
# Importing the dataset
df = pd.read_csv("Clean_phising.csv", na_values = missing_values,engine='python',skipinitialspace=True)
X = df.iloc[:,:-1].values
y= df['URL_Type_obf_Type']

from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
'''
Index(['id', 'Querylength', 'domain_token_count', 'path_token_count',
       'avgdomaintokenlen', 'longdomaintokenlen', 'avgpathtokenlen', 'tld',
       'charcompvowels', 'charcompace', 'ldl_url', 'ldl_domain', 'ldl_path',
       'ldl_filename', 'ldl_getArg', 'dld_url', 'dld_domain', 'dld_path',
       'dld_filename', 'dld_getArg', 'urlLen', 'domainlength', 'pathLength',
       'subDirLen', 'fileNameLen', 'this.fileExtLen', 'ArgLen', 'pathurlRatio',
       'ArgUrlRatio', 'argDomanRatio', 'domainUrlRatio', 'pathDomainRatio',
       'argPathRatio', 'executable', 'isPortEighty', 'NumberofDotsinURL',
       'ISIpAddressInDomainName', 'CharacterContinuityRate',
       'LongestVariableValue', 'URL_DigitCount', 'host_DigitCount',
       'Directory_DigitCount', 'File_name_DigitCount', 'Extension_DigitCount',
       'Query_DigitCount', 'URL_Letter_Count', 'host_letter_count',
       'Directory_LetterCount', 'Filename_LetterCount',
       'Extension_LetterCount', 'Query_LetterCount', 'LongestPathTokenLength',
       'Domain_LongestWordLength', 'Path_LongestWordLength',
       'sub-Directory_LongestWordLength', 'Arguments_LongestWordLength',
       'URL_sensitiveWord', 'URLQueries_variable', 'spcharUrl',
       'delimeter_Domain', 'delimeter_path', 'delimeter_Count',
       'NumberRate_URL', 'NumberRate_Domain', 'NumberRate_DirectoryName',
       'NumberRate_FileName', 'NumberRate_Extension', 'NumberRate_AfterPath',
       'SymbolCount_URL', 'SymbolCount_Domain', 'SymbolCount_Directoryname',
       'SymbolCount_FileName', 'SymbolCount_Extension',
       'SymbolCount_Afterpath', 'Entropy_URL', 'Entropy_Domain',
       'Entropy_DirectoryName', 'Entropy_Filename', 'Entropy_Extension',
       'Entropy_Afterpath', 'URL_Type_obf_Type'],
      dtype='object')'''


# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state =42)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
print("RandomForestClassifier")
start_time = time.time()
classifier.fit(X_train, y_train)
print('Time taken for training phase : {}'.format(time.time() - start_time))
start_time = time.time()
y_pred = classifier.predict(X_test)
print('Time taken for training  phase : {}'.format(time.time() - start_time))

print(accuracy_score(y_pred, y_test))
print(classification_report(y_pred, y_test))

# with gini
#DT
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
start_time = time.time()
classifier.fit(X_train, y_train)
print('Time taken for training phase : {}'.format(time.time() - start_time))
# Predicting the training  set results
print("DecisionTreeClassifier")

start_time = time.time()
y_pred = classifier.predict(X_test)


print('Time taken for testing phase : {}'.format(time.time() - start_time))
# Making the Confusion Matrix
# Predicting the Test set results

print(accuracy_score(y_pred, y_test))
print(classification_report(y_pred, y_test))


#QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
classifier = OneVsRestClassifier(QuadraticDiscriminantAnalysis())
print("QDA")

start_time = time.time()
classifier.fit(X_train, y_train)
print('Time taken for training  phase : {}'.format(time.time() - start_time))
# Predicting the Test set results
start_time = time.time()
y_pred = classifier.predict(X_test)
print('Time taken for testing phase : {}'.format(time.time() - start_time))
# Making the Confusion Matrix
# Predicting the Test set results
print(accuracy_score(y_pred, y_test))
print(classification_report(y_pred, y_test))


# from sklearn.multiclass import OneVsRestClassifier

from sklearn.naive_bayes import GaussianNB
classifier = OneVsRestClassifier(GaussianNB())
print("GaussianNB")

start_time = time.time()

classifier.fit(X_train, y_train)
print('Time taken for trainig phase : {}'.format(time.time() - start_time))
# Predicting the Test set results
start_time = time.time()
y_pred = classifier.predict(X_test)

print('Time taken for testing phase : {}'.format(time.time() - start_time))
# Making the Confusion Matrix
# Predicting the Test set results
print(accuracy_score(y_pred, y_test))
print(classification_report(y_pred, y_test))
# Presptron
from sklearn.linear_model import Perceptron
classifier = OneVsRestClassifier(Perceptron(tol=1e-3, random_state=0))
print("Perceptron")

start_time = time.time()
classifier.fit(X_train, y_train)
print('Time taken for testing phase : {}'.format(time.time() - start_time))
# Predicting the Test set results
start_time = time.time()
y_pred = classifier.predict(X_test)
classifier.fit(X_train, y_train)
print('Time taken for testing phase : {}'.format(time.time() - start_time))
print(accuracy_score(y_pred, y_test))
print(classification_report(y_pred, y_test))
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,p=2)
print("KNN")

start_time = time.time()
classifier.fit(X_train, y_train)
print('Time taken for testing phase : {}'.format(time.time() - start_time))
start_time = time.time()
y_pred = classifier.predict(X_test)
classifier.fit(X_train, y_train)
print('Time taken for testing phase : {}'.format(time.time() - start_time))
print(accuracy_score(y_pred, y_test))
print(classification_report(y_pred, y_test))


classifier = LogisticRegression()
print("LR")

start_time = time.time()
classifier.fit(X_train, y_train)
print('Time taken for testing phase : {}'.format(time.time() - start_time))
start_time = time.time()
y_pred = classifier.predict(X_test)
classifier.fit(X_train, y_train)
print('Time taken for testing phase : {}'.format(time.time() - start_time))
print(accuracy_score(y_pred, y_test))
print(classification_report(y_pred, y_test))
#
from sklearn.svm import SVC
classifier= SVC(kernel='rbf')
print("SVC")

start_time = time.time()
classifier.fit(X_train, y_train)
print('Time taken for testing phase : {}'.format(time.time() - start_time))
start_time = time.time()
y_pred = classifier.predict(X_test)
classifier.fit(X_train, y_train)
print('Time taken for testing phase : {}'.format(time.time() - start_time))
print(accuracy_score(y_pred, y_test))
print(classification_report(y_pred, y_test))
