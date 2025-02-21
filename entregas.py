import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def generar_datos_entregas(n_entregas=5000):
    fecha_base = datetime(2024, 1, 1)
    zonas = ['Norte', 'Sur', 'Este', 'Oeste', 'Centro']
    distancias_zona = {
        'Norte': (1, 40),
        'Sur': (1, 60),
        'Este': (1, 60),
        'Oeste': (1, 40),
        'Centro': (1, 20)
    }
    datos = []
    for _ in range(n_entregas):
        zona = random.choice(zonas)
        dist_min, dist_max = distancias_zona[zona]
        fecha = fecha_base + timedelta(days=random.randint(0, 60))
        distancia = round(random.uniform(dist_min, dist_max), 2)
        tiempo_estimado = distancia * 4
        variacion = random.uniform(0.8, 1.3)
        tiempo_real = round(tiempo_estimado * variacion, 2)
        costo_combustible = distancia * 0.5
        costo_personal = tiempo_real * 0.2
        costo_total = round(costo_combustible + costo_personal, 2)
        if tiempo_real <= tiempo_estimado * 1.1:
            estado = 'A tiempo'
        elif tiempo_real <= tiempo_estimado * 1.3:
            estado = 'Demorado'
        else:
            estado = 'Muy demorado'
        datos.append({
            'fecha': fecha,
            'zona': zona,
            'distancia_km': distancia,
            'tiempo_estimado_min': round(tiempo_estimado, 2),
            'tiempo_real_min': tiempo_real,
            'costo_total': costo_total,
            'estado': estado
        })
    return pd.DataFrame(datos)

def optimizar_modelo(df):
    df['dia_semana'] = df['fecha'].dt.weekday
    df['eficiencia'] = df['tiempo_estimado_min'] / df['tiempo_real_min']

    zonas = ['Norte', 'Sur', 'Este', 'Oeste', 'Centro']
    filas_ficticias = pd.DataFrame([{'zona': zona, 'distancia_km': 0, 'tiempo_real_min': 0, 'costo_total': 0, 'dia_semana': 0, 'eficiencia': 1} for zona in zonas])
    df = pd.concat([df, filas_ficticias], ignore_index=True)

    df_encoded = pd.get_dummies(df, columns=['zona'], drop_first=True)

    X = df_encoded[['distancia_km', 'dia_semana', 'eficiencia'] + [col for col in df_encoded.columns if col.startswith('zona_')]]
    y = df_encoded['tiempo_real_min']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = RandomForestRegressor()
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(modelo, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    mejor_modelo = grid_search.best_estimator_
    y_pred_test = mejor_modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)

    print(f"Mejor modelo: {grid_search.best_params_}")
    print(f"Error cuadrático medio (MSE) optimizado: {mse:.4f}")

    return mejor_modelo, mse

def crear_dashboard(df, modelo, mse):
    app = Dash(__name__)

    app.layout = html.Div([
        html.H1("Dashboard de Entregas"),
        html.Div(f"Error cuadrático medio (MSE): {mse:.4f}"),

        html.Label("Introduce una distancia (km):"),
        dcc.Input(id='input-distancia', type='number', value=5, step=0.1),

        dcc.Graph(id='grafico-zona'),
        dcc.Graph(id='grafico-estado'),
    ])

    @app.callback(
        [Output('grafico-zona', 'figure'),
         Output('grafico-estado', 'figure')],
        [Input('input-distancia', 'value')]
    )
    def actualizar_graficos(distancia):
        df_filtrado = df[df['distancia_km'] <= distancia]

        tiempo_por_zona = df_filtrado.groupby('zona')['tiempo_real_min'].mean()
        costo_por_zona = df_filtrado.groupby('zona')['costo_total'].mean()
        estado_counts = df_filtrado['estado'].value_counts()

        fig_zona = {
            'data': [
                go.Bar(x=tiempo_por_zona.index, y=tiempo_por_zona.values, name='Tiempo Promedio por Zona'),
                go.Bar(x=costo_por_zona.index, y=costo_por_zona.values, name='Costo Promedio por Zona')
            ],
            'layout': go.Layout(title='Análisis por Zona', barmode='group')
        }

        fig_estado = {
            'data': [
                go.Pie(labels=estado_counts.index, values=estado_counts.values, name='Estados de Entregas')
            ],
            'layout': go.Layout(title='Distribución de Estados de Entregas')
        }

        return fig_zona, fig_estado

    app.run_server(debug=True)

if __name__ == "__main__":
    df_entregas = generar_datos_entregas(5000)
    modelo, mse = optimizar_modelo(df_entregas)
    crear_dashboard(df_entregas, modelo, mse)
