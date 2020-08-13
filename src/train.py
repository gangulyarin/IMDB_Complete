import pandas as pd
from sklearn import model_selection, linear_model, metrics, decomposition
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

if __name__=="__main__":
    df = pd.read_csv("input/imdb.csv", nrows=10000)
    df["sentiment"] = df.sentiment.apply(lambda x: 1 if x == "positive" else 0)
    df["fold"]=-1
    skfold = model_selection.StratifiedKFold(n_splits=5)
    y = df.sentiment.values
    for f,(t,v) in enumerate(skfold.split(df,y)):
        df.loc[v,"fold"] = f

    for fold in range(5):
        train = df[df["fold"] != fold].reset_index(drop=True)
        test = df[df["fold"] == fold].reset_index(drop=True)
        countvec = CountVectorizer(tokenizer=word_tokenize)
        countvec.fit(train.review)
        X_train = countvec.transform(train.review)
        X_test = countvec.transform(test.review)
        model = linear_model.LinearRegression()
        model.fit(X_train,train.sentiment)
        preds = model.predict(X_test)
        # calculate accuracy
        accuracy = metrics.accuracy_score(test.sentiment, preds)
        print(f"Fold: {fold}")
        print(f"Accuracy = {accuracy}")
        print("")