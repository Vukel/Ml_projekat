from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Učitaj podatke
data = load_iris()
X = data.data
y = data.target

# Podeli podatke na trening i test skup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Kreiraj model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Testiraj model
predictions = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, predictions)}')