import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as lr
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import ExtraTreesClassifier as et
import pickle


def zero_one_loss(preds, true):
    
    preds=np.array(preds).squeeze()
    true=np.array(true).squeeze()
    
    return sum(abs(true-preds))/len(true)


if __name__ == '__main__':

    #read in data and shuffle rows
    data = pd.read_csv(sys.argv[1])
    data=data.sample(frac=1)

    #create train test split
    train=data[:int(len(data)*0.8)]
    test=data[int(len(data)*0.8):]

    #create instances of models
    mod=lr()
    mod_=rf()
    mod__=et()

    #fit all 3 model architectures
    mod.fit(train.iloc[:,:-1],train.iloc[:,-1])
    mod_.fit(train.iloc[:,:-1],train.iloc[:,-1])
    mod__.fit(train.iloc[:,:-1],train.iloc[:,-1])

    #write model errors
    with open(sys.argv[2],'w') as outfile:

        outfile.write(f'logistic regression test error: {zero_one_loss(mod.predict(test.iloc[:,:-1]),test.iloc[:,-1].to_numpy())}')
        outfile.write('\n')
        outfile.write(f'random forest test error: {zero_one_loss(mod_.predict(test.iloc[:,:-1]),test.iloc[:,-1].to_numpy())}')
        outfile.write('\n')
        outfile.write(f'extra trees test error: {zero_one_loss(mod__.predict(test.iloc[:,:-1]),test.iloc[:,-1].to_numpy())}')

    #save pickled model for all 3 architectures
    with open(f'{sys.argv[3]}/logReg.pkl','wb') as f:
        pickle.dump(mod,f)
    
    with open(f'{sys.argv[3]}/rf.pkl','wb') as f:
        pickle.dump(mod_,f)

    with open(f'{sys.argv[3]}/et.pkl','wb') as f:
        pickle.dump(mod__,f)