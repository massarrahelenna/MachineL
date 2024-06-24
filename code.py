import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
    
# Dados fornecidos
dados = [
    [6, 148, 72, 35, 0, 33.6, 0.627, 50, 1],
    [1, 85, 66, 29, 0, 26.6, 0.351, 31, 0],
    [8, 183, 64, 0, 0, 23.3, 0.672, 32, 1],
    [1, 89, 66, 23, 94, 28.1, 0.167, 21, 0],
    [0, 137, 40, 35, 168, 43.1, 2.288, 33, 1],
    [5, 116, 74, 0, 0, 25.6, 0.201, 30, 0],
    [3, 78, 50, 32, 88, 31.0, 0.248, 26, 1],
    [10, 115, 0, 0, 0, 35.3, 0.134, 29, 0],
    [2, 197, 70, 45, 543, 30.5, 0.158, 53, 1],
    [8, 125, 96, 0, 0, 0.0, 0.232, 54, 1],
    [4, 110, 92, 0, 0, 37.6, 0.191, 30, 0],
    [10, 168, 74, 0, 0, 38.0, 0.537, 34, 1],
    [10, 139, 80, 0, 0, 27.1, 1.441, 57, 0],
    [1, 189, 60, 23, 846, 30.1, 0.398, 59, 1],
    [5, 166, 72, 19, 175, 25.8, 0.587, 51, 1],
    [7, 100, 0, 0, 0, 30.0, 0.484, 32, 1],
    [0, 118, 84, 47, 230, 45.8, 0.551, 31, 1],
    [7, 107, 74, 0, 0, 29.6, 0.254, 31, 1],
    [1, 103, 30, 38, 83, 43.3, 0.183, 33, 0],
    [1, 115, 70, 30, 96, 34.6, 0.529, 32, 1],
    [3, 126, 88, 41, 235, 39.3, 0.704, 27, 0],
    [8, 99, 84, 0, 0, 35.4, 0.388, 50, 0],
    [7, 196, 90, 0, 0, 39.8, 0.451, 41, 1],
    [9, 119, 80, 35, 0, 29.0, 0.263, 29, 1],
    [11, 143, 94, 33, 146, 36.6, 0.254, 51, 1],
    [10, 125, 70, 26, 115, 31.1, 0.205, 41, 1],
    [7, 147, 76, 0, 0, 39.4, 0.257, 43, 1],
    [1, 97, 66, 15, 140, 23.2, 0.487, 22, 0],
    [13, 145, 82, 19, 110, 22.2, 0.245, 57, 0],
    [5, 117, 92, 0, 0, 34.1, 0.337, 38, 0],
    [5, 109, 75, 26, 0, 36.0, 0.546, 60, 0],
    [3, 158, 76, 36, 245, 31.6, 0.851, 28, 1],
    [3, 88, 58, 11, 54, 24.8, 0.267, 22, 0],
    [6, 92, 92, 0, 0, 19.9, 0.188, 28, 0],
    [10, 122, 78, 31, 0, 27.6, 0.512, 45, 0],
    [4, 103, 60, 33, 192, 24.0, 0.966, 33, 0],
    [11, 138, 76, 0, 0, 33.2, 0.420, 35, 0],
    [9, 102, 76, 37, 0, 32.9, 0.665, 46, 1],
    [2, 90, 68, 42, 0, 38.2, 0.503, 27, 1],
    [4, 111, 72, 47, 207, 37.1, 1.390, 56, 1],
    [3, 180, 64, 25, 70, 34.0, 0.271, 26, 0],
    [7, 133, 84, 0, 0, 40.2, 0.696, 37, 0],
    [7, 106, 92, 18, 0, 22.7, 0.235, 48, 0],
    [9, 171, 110, 24, 240, 45.4, 0.721, 54, 1],
    [7, 159, 64, 0, 0, 27.4, 0.294, 40, 0],
    [0, 180, 66, 39, 0, 42.0, 1.893, 25, 1],
    [1, 146, 56, 0, 0, 29.7, 0.564, 29, 0],
    [2, 71, 70, 27, 0, 28.0, 0.586, 22, 0],
    [7, 103, 66, 32, 0, 39.1, 0.344, 31, 1],
    [7, 105, 0, 0, 0, 0.0, 0.305, 24, 0],
    [1, 103, 80, 11, 82, 19.4, 0.491, 22, 0],
    [1, 101, 50, 15, 36, 24.2, 0.526, 26, 0],
    [5, 88, 66, 21, 23, 24.4, 0.342, 30, 0],
    [8, 176, 90, 34, 300, 33.7, 0.467, 58, 1],
    [7, 150, 66, 42, 342, 34.7, 0.718, 42, 0],
    [1, 73, 50, 10, 0, 23.0, 0.248, 21, 0],
    [7, 187, 68, 39, 304, 37.7, 0.254, 41, 1],
    [0, 100, 88, 60, 110, 46.8, 0.962, 31, 0],
    [0, 146, 82, 0, 0, 40.5, 1.781, 44, 0],
    [0, 105, 64, 41, 142, 41.5, 0.173, 22, 0],
    [2, 84, 0, 0, 0, 0.0, 0.304, 21, 0],
    [8, 133, 72, 0, 0, 32.9, 0.270, 39, 1],
    [5, 44, 62, 0, 0, 25.0, 0.587, 36, 0],
    [2, 141, 58, 34, 128, 25.4, 0.699, 24, 0],
    [7, 114, 66, 0, 0, 32.8, 0.258, 42, 0],
    [5, 99, 74, 27, 0, 29.0, 0.203, 32, 0],
    [0, 109, 88, 30, 0, 32.5, 0.855, 38, 0],
    [2, 109, 92, 0, 0, 42.7, 0.845, 54, 0],
    [1, 95, 66, 13, 38, 19.6, 0.334, 25, 0],
    [4, 146, 85, 27, 100, 28.9, 0.189, 27, 0],
    [2, 100, 66, 20, 90, 32.9, 0.867, 28, 1],
    [5, 139, 64, 35, 140, 28.6, 0.411, 26, 0],
    [13, 126, 90, 0, 0, 43.4, 0.583, 42, 1],
    [4, 129, 86, 20, 270, 35.1, 0.231, 23, 0],
    [1, 79, 75, 30, 0, 32.0, 0.396, 22, 0],
    [1, 0, 48, 20, 0, 24.7, 0.140, 22, 0]
]

# Criação do DataFrame
data = pd.DataFrame(dados, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])

# Remova a coluna 'Outcome'
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Divida os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crie e treine o modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Faça previsões no conjunto de teste
y_pred = model.predict(X_test)

# Previsões de probabilidade
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilidade da classe 1 (desenvolver diabetes)

# Calcule a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Calculando probabilidades no conjunto de teste
probabilities = model.predict_proba(X_test)
print("\nProbabilidades de desenvolver diabetes:")
for i, prob in enumerate(probabilities):
    print(f"Amostra {i+1}: Probabilidade de não desenvolver diabetes = {prob[0]:.3f}, Probabilidade de desenvolver diabetes = {prob[1]:.3f}")
    
# Salvando o modelo treinado
joblib.dump(model, '/home/lenna/desktop/.MachineL/diabetes_model.pkl')

# Salvando os dados de teste, previsões e probabilidades em um arquivo CSV
resultados = X_test.copy()
resultados['Actual Outcome'] = y_test
resultados['Predicted Outcome'] = y_pred
resultados['Probability of Diabetes'] = y_prob

resultados.to_csv('/home/lenna/desktop/.MachineL/diabetes_predictions.csv', index=False)
