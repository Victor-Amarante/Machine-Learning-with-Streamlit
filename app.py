# pacotes
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# titulo
st.header("""
Prevendo Diabetes\n
Aplicação que utiliza machine learning para prever possível diabetes dos pacientes\n
Fonte: PIMA - INDIA (Kaggle)
""")

# Dataset
df = pd.read_csv('diabetes_clean.csv')

# cabecalho
st.subheader('Informações dos dados')

# nome do usuario
user_name = st.sidebar.text_input('Digite seu nome: ')

st.write(f'Paciente: {user_name}')

# dados de entrada
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2, random_state = 42)

# captar as informacoes dos usuario pela funcao
def get_user_data():
    pregnancies = st.sidebar.slider('Gravidez', 0, 15, 1)
    glucose = st.sidebar.slider('Glicose', 0, 200, 110)
    blood_pressure = st.sidebar.slider('Pressão Sanguínea', 0, 122, 72)
    skin_thickness = st.sidebar.slider('Espessura da pele', 0, 99, 20)
    insulin = st.sidebar.slider('Insulina', 0, 900, 30)
    bmi = st.sidebar.slider('Indice de massa corporal', 0.0, 70.0, 15.0)
    dpf = st.sidebar.slider('Historico familiar de diabetes', 0.0, 3.0, 0.0)
    age = st.sidebar.slider('Idade', 15, 100, 21)

    user_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    features = pd.DataFrame(user_data, index=[0])

    return features

user_input_variables = get_user_data()

grafico = st.bar_chart(user_input_variables)

st.subheader('Dados do Usuário')
st.write(user_input_variables)

modelo = DecisionTreeClassifier(criterion='entropy', max_depth=3)
modelo.fit(x_train, y_train)

st.subheader('Acurácia do modelo')
st.write(round(accuracy_score(y_test, modelo.predict(x_test)) * 100, 2))

prediction = modelo.predict(user_input_variables)

st.subheader('Previsão: ')
st.write(prediction)
