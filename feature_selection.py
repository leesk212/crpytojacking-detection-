import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import shap
from matplotlib import pyplot as plt
import os

os.chdir("forlstm/")
filenames = os.listdir()
print(len(filenames))
raw = pd.read_csv('init.csv')
# trainX = raw.values[0:70000, 0:32] 2000*35
# trainy = raw.values[0:350, 32]
# testX = raw.values[70000:, 0:32]
# testy = raw.values[350:500, 32]
trainX = raw.values[2000:, 1:65]  # 10,800,64
trainy = raw.values[2000:, 0]
testX = raw.values[:2000, 1:65]  # 10,200,64
testy = raw.values[:2000, 0]
print('data loding..')
print(str(0) + ': ', end='')
print(trainX.shape, end="")
print(testX.shape)

for i, filename in enumerate(filenames):
    if i > 0:
        raws = pd.read_csv(filename, thousands=',')
        new_trainX = raws.values[2000:, 1:65]
        new_trainy = raws.values[2000:, 0]
        new_testX = raws.values[:2000, 1:65]
        new_testy = raws.values[:2000, 0]

        trainX = np.append(trainX, new_trainX, axis=0)
        trainy = np.append(trainy, new_trainy, axis=0)
        testX = np.append(testX, new_testX, axis=0)
        testy = np.append(testy, new_testy, axis=0)

        print(str(i) + ': ', end='')
        print(trainX.shape, end="")
        print(testX.shape, end=' - check: ')
        print(str(i + 1) + '*' + str(2000) + '=' + str((i + 1) * 2000))

os.chdir('../PycharmProjects/ML_for_lab/crpytojacking-detection--main/')

rf = RandomForestRegressor(n_estimators=100)

index = list(raw.columns)
del index[0]
rf.fit(trainX,trainy)
print(rf.feature_importances_)

df2 = pd.DataFrame({"index":index, "importance":rf.feature_importances_})
plt.barh('index','importance',data=df2)
plt.show()
df_sorted = df2.sort_values('importance')
plt.barh('index','importance',data=df_sorted)
plt.show()

#plt.barh(index, rf.feature_importances_)
#plt.show()
#plt.barh(index, sorted(rf.feature_importances_))
#plt.show()
#sorted_idx = rf.feature_importances_.argsort()

print('permutation importances..')

perm_importance = permutation_importance(rf, trainX, trainy)
df3 = pd.DataFrame({"index":index, "importance": perm_importance.importances_mean})
plt.barh('index','importance',data=df3)
plt.show()
df_sorted = df3.sort_values('importance')
plt.barh('index','importance',data=df_sorted)
plt.show()

print('shap importances..')

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(trainX)
shap.summary_plot(shap_values, trainX, plot_type="bar")
shap.summary_plot(shap_values, trainX)