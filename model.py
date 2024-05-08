import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.inspection import permutation_importance
# قراءة البيانات
data = pd.read_csv("medical_data.csv"  ).iloc[:, :-2]
features_name = list(data['spicitalist'])
features_name.append('spicitalist')
class_names = data.columns[1:]
data_reshaped = data.drop(['spicitalist'], axis=1)
data_reshaped.loc[len(data_reshaped.index)] = class_names

data_reshaped = pd.DataFrame(data_reshaped.values.T, columns=features_name)
# تحديد المتغيرات المستقلة (الأعراض) والمستهدفة (التخصص الطبي)
X =  data_reshaped.drop(['spicitalist'], axis=1)
encoder= LabelEncoder()
y =  encoder.fit_transform(data_reshaped['spicitalist'])
# تقسيم البيانات إلى مجموعة تدريب ومجموعة اختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
g= data_reshaped['spicitalist']
scikit_model = SVC(kernel='rbf')

scikit_model.fit(X, y)

accuracy = scikit_model.score(X_test, y_test)
print("Accuracy:", accuracy)
# Calculate permutation importance
perm_importance = permutation_importance(scikit_model, X, y)

# Get feature importance scores
feature_importance_scores = perm_importance.importances_mean

print("Feature Importance Scores:", feature_importance_scores)

# Save the trained model
with open('model.pkl', "wb") as file:
    pickle.dump(scikit_model, file)
