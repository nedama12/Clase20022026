import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64

#  Cargar dataset
df = pd.read_excel("F1 - Entrega Final.xlsx")

#  Seleccionar columnas necesarias
df = df[['grid', 'laps', 'positionOrder']].dropna()

#  Variables independientes (X) y dependiente (y)
X = df[['grid', 'laps']]
y = df['positionOrder']

#  Crear y entrenar modelo
model = LinearRegression()
model.fit(X, y)

#  FUNCIÓN DE PREDICCIÓN
def calculatePosition(grid, laps):
    result = model.predict([[grid, laps]])[0]
    return round(result, 2)


#  FUNCIÓN PARA GENERAR GRÁFICA
def generate_plot():
    import matplotlib.pyplot as plt
    import io
    import base64

    plt.figure()

    # Solo puntos 
    plt.scatter(df['grid'], y, alpha=0.5)

    plt.xlabel("Grid (Starting Position)")
    plt.ylabel("Final Position")
    plt.title("Grid vs Final Position")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    return base64.b64encode(img.getvalue()).decode()