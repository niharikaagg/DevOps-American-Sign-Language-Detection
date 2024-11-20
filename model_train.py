import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict=pickle.load(open('./preprocessed_data.pickle','rb'))
# print(data_dict.keys())
# print(data_dict)

# Find the length of the longest sequence
max_length = max(len(features) for features in data_dict['data'])

# Standardize lengths by padding each sequence to the max length
data = []
for features in data_dict['data']:
    if len(features) < max_length:
        # Pad shorter sequences with zeros
        padded_features = features + [0] * (max_length - len(features))
    else:
        # If too long, truncate to max_length
        padded_features = features[:max_length]
    data.append(padded_features)

# Convert to a numpy array
data = np.array(data)
# data=np.asarray(data_dict['data'])
labels=np.asarray(data_dict['labels'])

x_train, x_text, y_train, y_test = train_test_split(data,labels,test_size=0.2,shuffle=True,stratify=labels)

model=RandomForestClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_text)

score=accuracy_score(y_predict,y_test)
print('{}% of samples were classified correctly'.format(score*100))
# print("Model Accuracy:", score)

f=open('trained_data_model.p','wb')
pickle.dump({'model':model},f)
f.close()