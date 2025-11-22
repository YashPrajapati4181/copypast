MODULE 5

SKILL BUILDER


Jegan


import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,recall_score,classification_report

filepath = os.path.join(sys.path[0], input().strip())
df = pd.read_csv(filepath)
X = df['document']
y = df['label']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec,y_train)
y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test,y_pred)
print(f"Accuracy: {acc:.2f}")

mr = recall_score(y_test,y_pred,average='macro',zero_division=0)
print(f"Recall: {mr:.2f}")

print("\nClassification Report:")
print(classification_report(y_test,y_pred,digits=2,zero_division=0))



Dwashik




import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,precision_score

filepath = os.path.join(sys.path[0], input().strip())
df = pd.read_csv(filepath)

X = df['document']
y = df['label']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec,y_train)

y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test,y_pred)
pre = precision_score(y_test,y_pred,pos_label='Yes',zero_division=1)

print(f"Accuracy: {acc:.2f}")
print(f"Precision: {pre:.2f}")




James






import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report

filepath = os.path.join(sys.path[0], input().strip())
df = pd.read_csv(filepath)

X = df['document']
y = df['label']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec,y_train)

y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test,y_pred)
cr = classification_report(y_test,y_pred,zero_division=1)

print(f"Accuracy: {acc:.2f}")
print("Classification Report:")
print(cr)




Kenichi





import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,recall_score

filepath = os.path.join(sys.path[0], input().strip())
df = pd.read_csv(filepath)

X = df['Item']
y = df['Quality']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec,y_train)

y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test,y_pred)
re = recall_score(y_test,y_pred,average='macro',zero_division=1)

print(f"Accuracy: {acc:.2f}")
print(f"Recall: {re:.2f}")



Reena





import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

filepath = os.path.join(sys.path[0], input().strip())
df = pd.read_csv(filepath)

num_cols = df.select_dtypes(include=['number']).columns
for col in num_cols:
    df[col].fillna(df[col].mean(),inplace=True)

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df.iloc[:, 0])
y = df.iloc[:, 1].apply(lambda x: 1 if str(x).strip().lower() == 'spam' else 0)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

model = GaussianNB()
model.fit(X_train_dense,y_train)

y_pred = model.predict(X_test_dense)
acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred,zero_division=1)

print(f"Accuracy: {acc:.2f}")
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(cr)
