import numpy as np
import re
from collections import Counter

class NaiveBayesScratch:

    def __init__(self,alpha=1.0,max_features=5000):
        self.alpha=alpha
        self.max_features=max_features


    def _tokenize(self,text):
        text=text.lower()
        return re.findall(r'\b[a-z]{2,}\b',text)


    def buid_vocabulary(self,texts):
        counter=Counter()
        for text  in texts:
            counter.update(self._tokenize(text))

        most_common=counter.most_common(self.max_features)
        self.vocab={word:i for i,(word,_) in enumerate(most_common)}
        self.vocab_size=len(self.vocab)


    def text_to_vec(self,text):
        vec=np.zeros(self.vocab_size)
        for word in self._tokenize(text):
            if word in self.vocab:
                vec[self.vocab[word]]+=1
        return vec



    def fit(self,texts,labels):
        texts=np.array(texts)
        labels=np.array(labels)

        self.classes=np.unique(labels)
        self.buid_vocabulary(texts)

        X=np.array([self.text_to_vec(t) for t in texts])

        self.class_priors={}
        self.word_probs={}

        n_samples=X.shape[0]

        for c in self.classes:
            X_c=X[labels==c]

            self.class_priors[c]=np.log(X_c.shape[0]/n_samples)

            word_count=X_c.sum(axis=0)
            total_words=word_count.sum()

            self.word_probs[c]=np.log(
                (word_count+self.alpha)/
                (total_words+self.alpha*self.vocab_size)
            )

    def predict(self,texts):
        predictions=[]

        for text in texts:
            x=self.text_to_vec(text)
            class_scores={}

            for c in self.classes:
                class_scores[c]=(
                    self.class_priors[c]+
                    np.sum(x*self.word_probs[c])
                )

            predictions.append(max(class_scores,key=class_scores.get))
        return np.array(predictions)

    def score(self,texts,labels):
        preds=self.predict(texts)
        return np.mean(preds==labels)

#Testing prototype
if __name__ == "__main__":
    # Sample dataset
    texts = [
        "Buy cheap phones now",
        "Limited time offer, buy now",
        "Discussion on government policy",
        "Parliament debates policy reforms"
    ]

    labels = [1, 1, 0, 0]  # 1=spam, 0=not spam

    # Create model
    model = NaiveBayesScratch(alpha=1.0, max_features=20)

    # Train
    model.fit(texts, labels)

    # Test predictions
    test_texts = [
        "Buy cheap now",
        "Government debates new policy"
    ]

    preds = model.predict(test_texts)

    for text, pred in zip(test_texts, preds):
        print(f"Text: {text} -> Predicted: {pred}")

    # Test accuracy on training set
    print("Training accuracy:", model.score(texts, labels))
