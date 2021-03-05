import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


def data_split(data,ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data)*ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

if __name__ == "__main__":
    df=pd.read_csv('data.csv')
    train,test=data_split(df,0.3)
    train_corr = train.select_dtypes(include=[np.number])
    corr = train_corr.corr()
    plt.subplots(figsize=(20,9))
    swarm_plot=sns.heatmap(corr, annot=True)
    fig = swarm_plot.get_figure()
    fig.savefig('visualization.png') 
    x_train=train[['fever','bodypain','age','runnynose','diffBreath']].to_numpy()
    x_test=test[['fever','bodypain','age','runnynose','diffBreath']].to_numpy()
    y_train=train[['infectionProb']].to_numpy().reshape(7289,)
    y_test=test[['infectionProb']].to_numpy().reshape(3123,)
    clf1=LogisticRegression().fit(x_train,y_train)
    score1=clf1.score(x_test,y_test)
    print(score1," LogisticRegression")
    clf2= RandomForestRegressor(max_depth=2, random_state=0).fit(x_train, y_train)
    score2=clf2.score(x_test,y_test)
    print(score2," RandomForestRegressor")
    clf3 = GradientBoostingRegressor(n_estimators=500, learning_rate=0.5,
    max_depth=3, random_state=0).fit(x_train, y_train)
    score3=clf3.score(x_test,y_test)
    print(score3," GradientBoostingRegressor")
    
    if score1 > score3 and score2 :
        mdl=clf1
    elif score2 > score3:
        mdl=clf2
    else:
        mdl=clf3

    file = open('model.pkl','wb')
    pickle.dump(mdl,file)
    file.close()