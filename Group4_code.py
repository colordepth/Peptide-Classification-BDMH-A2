import pandas as pd
import argparse
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef
from imblearn.over_sampling import BorderlineSMOTE, SMOTE, ADASYN, KMeansSMOTE, SVMSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier, RUSBoostClassifier, EasyEnsembleClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from os import path
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from collections import Counter
from imblearn.metrics import classification_report_imbalanced
import os
from sklearn import metrics
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from autogluon.tabular import TabularDataset, TabularPredictor

parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--train_csv", required=True, help="Path to train CSV.")
parser.add_argument("-te", "--test_csv", required=True, help="Path to test CSV.")
parser.add_argument("-o", "--output", required=True, help="Filename of output csv, which will be exported.")

args = parser.parse_args()
train_csv = args.train_csv
test_csv = args.test_csv
output_csv = args.output

# Loading data
df_train_orig = pd.read_csv(train_csv)
df_test_orig = pd.read_csv(test_csv)
df_test_orig.rename(columns={' Sequence': 'Sequence'}, inplace=True)

df_train = df_train_orig.copy()
df_test = df_test_orig.copy()

combined_sequences = np.array(list(df_train.Sequence) + list(df_test.Sequence))

# TF-IDF vectorizing
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), analyzer='char')
tfidf_sequences = tfidf_vectorizer.fit_transform(combined_sequences)
tfidf_sequences
tf_idf_df = pd.DataFrame(tfidf_sequences.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Joining TF-IDF vectors to our dataset
df_full = pd.DataFrame(combined_sequences, columns=['Sequence'])
df_full = df_full.join(tf_idf_df, rsuffix='-tfidf')

df_full['SeqLength'] = [len(seq) for seq in df_full['Sequence']]

# Joining PFeature features
for filename in os.listdir('Pfeature_scripts/features'):
    try:
        df = pd.read_csv(path.join('Pfeature_scripts/features', filename))
        if len(df) == len(df_full):
            df_full = df_full.join(df, rsuffix=filename)
    except:
        pass

# Drop fully NaN columns
df_full = df_full.dropna(axis=1, how='all')
df_numeric = df_full._get_numeric_data()

# Imputing NaN values with mean
X = df_numeric.copy()
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imp.fit_transform(X)

# Standardize the features (optional but recommended for k-nearest neighbor algorithm)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Performing PCA
n_components=50
pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X_scaled)

pca_names = ['PCA_' + str(i+1) for i in range(n_components)]

for i in range(n_components):
    df_numeric[pca_names[i]] = X_pca[:, i]

N = len(df_train)
n = len(df_test)

df_train_numeric = df_numeric.iloc[:N]
df_test_numeric = df_numeric.iloc[N:]
df_test_numeric = df_test_numeric.reset_index(drop=True)

df_train = df_train_orig.join(df_train_numeric, rsuffix='-numeric').copy()
df_test = df_test_orig.join(df_test_numeric, rsuffix='-numeric').copy()

# Joining BioPy features
df = pd.read_csv('train_feats_biopy.csv')
df_train = df_train.join(df._get_numeric_data(), rsuffix='-biopy')

df = pd.read_csv('test_feats_biopy.csv')
df_test = df_test.join(df.drop(columns=['ID'])._get_numeric_data(), rsuffix='-biopy')

# Joining ESM Features
pca = PCA(n_components=100, random_state=42)
df = pd.read_csv('train_esm_features.csv')
X_scaled = scaler.fit_transform(df)
X_pca = pca.fit_transform(X_scaled)
df_train = df_train.join(pd.DataFrame(X_pca), rsuffix='-esm')

df = pd.read_csv('test_esm_features.csv')
X_scaled = scaler.transform(df)
X_pca = pca.transform(X_scaled)
df_test = df_test.join(pd.DataFrame(X_pca), rsuffix='-esm')

# Creating a validation set of 600 samples
val_size = 600
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True) # shuffle
df_v = df_train.iloc[:val_size].reset_index(drop=True)
df_train = df_train.iloc[val_size:].reset_index(drop=True)

# Preparing data and metrics
X = np.array(np.matrix(df_train.drop(columns=['Sequence', 'Label'])))
y = [ord(c) - ord('A') for c in df_train['Label']]

# Oversampling method for minority classes
method = BorderlineSMOTE(n_jobs=-1, random_state=42,
                        kind='borderline-1',
                        # kind='borderline-2',
                        )
print(method)
X_resampled, y_resampled = method.fit_resample(X, y)
print(sorted([(chr(ord('A')+k),v) for k,v in Counter(y_resampled).items()]))
print(X_resampled.shape)

X_val = np.array(np.matrix(df_v.drop(columns=['Sequence', 'Label'])))
y_val = [ord(c) - ord('A') for c in df_v['Label']]

tabularX = TabularDataset(X)
tabularX['Label'] = y
valTabularX = TabularDataset(X_val)
valTabularX['Label'] = y_val

clf = TabularPredictor(label='Label',
                       path='autog',
                       eval_metric='mcc'
)

# Start training the classifier
print("Training a", clf)
print(X.shape)
clf.fit(tabularX,
        presets='best_quality',
        time_limit=60*30
)

# Show the ensembles performance on validation data
leaderboard = clf.leaderboard(valTabularX, silent=True)
print(leaderboard)

print("Evaluating RF on validation")
y_val_pred = np.array(clf.predict(valTabularX))

accuracy = accuracy_score(y_val, y_val_pred)
mcc = matthews_corrcoef(y_val, y_val_pred)
balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)
report = classification_report_imbalanced(y_val, y_val_pred, output_dict=True)

print("Accuracy:", accuracy)
print("MCC:", mcc)
print("Balanced Accuracy:", balanced_accuracy)
print("Geometric:", report['avg_geo'])
print("f1:", report['avg_f1'])
print("AUC Score (OVR):", auc_score_ovr)
print("AUC Score (OVO):", auc_score_ovo)
print()
print(classification_report_imbalanced(y_val, y_val_pred, target_names=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']))

# Plot the confusion matrix
cf_mat = metrics.confusion_matrix(y_val, y_val_pred)
df_cm = pd.DataFrame(cf_mat, index = [i for i in "ABCDEFGH"],
                  columns = [i for i in "ABCDEFGH"])
plt.figure(figsize = (14,11))
sn.heatmap(df_cm, annot=True, fmt='d',)
plt.title(str(clf))
plt.show()

# Inference on test data
print("Creating submission.csv for test data")
X_test = np.array(np.matrix(df_test.drop(columns=['Sequence', 'ID'])))
testTabular = TabularDataset(X_test)
y_test_pred = np.array(clf.predict(testTabular))

df_test_submission = df_test[['ID']].copy()
df_test_submission['Lable'] = [chr(label + ord('A')) for label in y_test_pred]

print("Test submission distribution:")
print(df_test_submission['Lable'].value_counts())

filename = output_csv
df_test_submission.to_csv(filename, index=False)
print("Saved output as", filename)