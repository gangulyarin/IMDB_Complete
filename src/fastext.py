import pandas as pd
import numpy as np
import io
from nltk.tokenize import word_tokenize
from sklearn import model_selection, linear_model, metrics

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding="utf-8", newline="\n", errors="ignore")
    n,d = map((int), fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data

def sentence_to_vec(s, embedding_dict, stop_words, tokenizer):
    words = str(s).lower()
    words = tokenizer(s)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]

    M=[]
    for w in words:
        if w in embedding_dict:
            M.append(embedding_dict[w])
    if len(M)==0:
        return np.zeros(300)
    M = np.array(M)
    v = M.sum(axis=0)
    return v/np.sqrt((v**2).sum())

if __name__=="__main__":
    df = pd.read_csv("input/imdb.csv")
    df.sentiment = df.sentiment.apply(lambda x:1 if x=="positive" else 0)
    df = df.sample(frac=1).reset_index(drop=True)
    # load embeddings into memory
    print("Loading embeddings")
    embeddings = load_vectors("input/crawl-300d-2M.vec")
    vectors=[]
    for review in df.review:
        vectors.append(sentence_to_vec(review, embeddings,stop_words=[], tokenizer=word_tokenize))
    vectors = np.array(vectors)
    y = df.sentiment.values
    sk = model_selection.StratifiedKFold(n_splits=5)

    for f, (t,v) in sk.split(X=vectors, y=y):
        X_train = vectors[t,:]
        X_test = vectors[v,:]
        y_train = y[t]
        y_test = y[v]
        model = linear_model.LogisticRegression()
        model.fit(X_train,y_train)
        preds = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test,preds)
        print(f"Accuracy={accuracy}")
        print()