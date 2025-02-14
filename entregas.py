import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def generar_datos_entregas(n_entregas=1000):
    """
    Genera un conjunto de datos simulado de entregas a domicilio
    """
    # Fecha base para las entregas
    fecha_base = datetime(2024, 1, 1)
    
    # Zonas de entrega con diferentes características
    zonas = ['Norte', 'Sur', 'Este', 'Oeste', 'Centro']
    distancias_zona = {
        'Norte': (5, 15),
        'Sur': (4, 12),
        'Este': (6, 18),
        'Oeste': (5, 16),
        'Centro': (2, 8)
    }
    
    datos = []
    
    for _ in range(n_entregas):
        zona = random.choice(zonas)
        dist_min, dist_max = distancias_zona[zona]
        
        # Generar datos de entrega
        fecha = fecha_base + timedelta(days=random.randint(0, 60))
        distancia = round(random.uniform(dist_min, dist_max), 2)
        tiempo_estimado = distancia * 4  # 4 minutos por kilómetro en promedio
        
        # Agregar variabilidad al tiempo real
        variacion = random.uniform(0.8, 1.3)
        tiempo_real = round(tiempo_estimado * variacion, 2)
        
        # Calcular costos
        costo_combustible = distancia * 0.5  # $0.50 por kilómetro
        costo_personal = tiempo_real * 0.2   # $0.20 por minuto
        costo_total = round(costo_combustible + costo_personal, 2)
        
        # Estado de la entrega
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

def analizar_datos_logisticos(df):
    """
    Realiza un análisis completo de los datos logísticos
    """
    # Análisis general
    metricas_generales = {
        'total_entregas': len(df),
        'tiempo_promedio': round(df['tiempo_real_min'].mean(), 2),
        'costo_promedio': round(df['costo_total'].mean(), 2),
        'porcentaje_a_tiempo': round((df['estado'] == 'A tiempo').mean() * 100, 2)
    }
    
    # Análisis por zona
    analisis_zona = df.groupby('zona').agg({
        'tiempo_real_min': 'mean',
        'costo_total': 'mean',
        'distancia_km': 'mean',
        'estado': lambda x: (x == 'A tiempo').mean() * 100
    }).round(2)
    
    analisis_zona.columns = ['tiempo_promedio', 'costo_promedio', 
                            'distancia_promedio', 'porcentaje_a_tiempo']
    
    # Análisis temporal
    df['semana'] = df['fecha'].dt.isocalendar().week
    tendencia_temporal = df.groupby('semana').agg({
        'tiempo_real_min': 'mean',
        'costo_total': 'mean',
        'estado': lambda x: (x == 'A tiempo').mean() * 100
    }).round(2)
    
    # Análisis de eficiencia
    df['eficiencia_tiempo'] = df['tiempo_estimado_min'] / df['tiempo_real_min']
    df['costo_por_km'] = df['costo_total'] / df['distancia_km']
    
    metricas_eficiencia = {
        'eficiencia_promedio': round(df['eficiencia_tiempo'].mean(), 3),
        'costo_por_km_promedio': round(df['costo_por_km'].mean(), 2)
    }
    
    return {
        'metricas_generales': metricas_generales,
        'analisis_zona': analisis_zona,
        'tendencia_temporal': tendencia_temporal,
        'metricas_eficiencia': metricas_eficiencia
    }

def crear_dashboard_entregas(df):
    """
    Crea un dashboard interactivo para visualizar los datos de entrega
    """
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Crear figura con subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Tiempo Promedio por Zona', 'Costos por Zona',
                       'Tendencia Temporal', 'Distribución de Estados'),
        specs=[[{'type': 'xy'}, {'type': 'xy'}],
               [{'type': 'xy'}, {'type': 'domain'}]]
    )
    
    # 1. Tiempo promedio por zona
    tiempos_zona = df.groupby('zona')['tiempo_real_min'].mean()
    fig.add_trace(
        go.Bar(x=tiempos_zona.index, y=tiempos_zona.values,
               name='Tiempo Promedio'),
        row=1, col=1
    )
    
    # 2. Costos por zona
    costos_zona = df.groupby('zona')['costo_total'].mean()
    fig.add_trace(
        go.Bar(x=costos_zona.index, y=costos_zona.values,
               name='Costo Promedio'),
        row=1, col=2
    )
    
    # 3. Tendencia temporal
    df['semana'] = df['fecha'].dt.isocalendar().week
    tendencia = df.groupby('semana')['tiempo_real_min'].mean()
    fig.add_trace(
        go.Scatter(x=tendencia.index, y=tendencia.values,
                  mode='lines+markers', name='Tendencia Temporal'),
        row=2, col=1
    )
    
    # 4. Distribución de estados
    estados_count = df['estado'].value_counts()
    fig.add_trace(
        go.Pie(labels=estados_count.index, values=estados_count.values,
               name='Estados'),
        row=2, col=2
    )
    
    # Actualizar diseño
    fig.update_layout(height=800, showlegend=True,
                     title_text="Dashboard de Análisis Logístico")
    
    
    
    return fig

def entrenar_modelo_prediccion(df):
    """
    Entrena un modelo de regresión para predecir el tiempo de entrega
    """
    # Añadir filas ficticias para cada zona para asegurar que todas las zonas estén presentes
    zonas = ['Norte', 'Sur', 'Este', 'Oeste', 'Centro']
    filas_ficticias = pd.DataFrame([{'zona': zona, 'distancia_km': 0, 'tiempo_real_min': 0, 'costo_total': 0} for zona in zonas])
    df = pd.concat([df, filas_ficticias], ignore_index=True)
    
    # Convertir variables categóricas a variables dummy
    df = pd.get_dummies(df, columns=['zona'], drop_first=True)
    
    # Seleccionar características y variable objetivo
    X = df[['distancia_km'] + [col for col in df.columns if col.startswith('zona_')]]
    y = df['tiempo_real_min']
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar el modelo de regresión lineal
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Error cuadrático medio: {mse:.2f}")
    
    return modelo

def predecir_tiempo_entrega(modelo, distancia, zona):
    """
    Predice el tiempo de entrega dado un modelo entrenado, distancia y zona
    """
    # Crear un DataFrame con la información de entrada
    zonas = ['Norte', 'Sur', 'Este', 'Oeste', 'Centro']
    input_data = pd.DataFrame([[distancia] + [1 if z == zona else 0 for z in zonas[1:]]],
                              columns=['distancia_km'] + [f'zona_{z}' for z in zonas[1:]])
    
    # Asegurarse de que las columnas coincidan con las utilizadas durante el entrenamiento
    for col in modelo.feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Reorder columns to match the model's expected input
    input_data = input_data[modelo.feature_names_in_]
    
    # Realizar la predicción
    tiempo_predicho = modelo.predict(input_data)[0]
    return tiempo_predicho

# Ejemplo de uso
if __name__ == "__main__":
    # Generar datos
    df_entregas = generar_datos_entregas(1000)
    
    # Realizar análisis
    resultados = analizar_datos_logisticos(df_entregas)
    
    # Crear visualización
    dashboard = crear_dashboard_entregas(df_entregas)
    dashboard.show()
    
    # Entrenar modelo de predicción
    modelo = entrenar_modelo_prediccion(df_entregas)
    
    # Predecir tiempo de entrega para un ejemplo
    distancia = 5  # km
    zona = 'Oeste'
    tiempo_predicho = predecir_tiempo_entrega(modelo, distancia, zona)
    print(f"Tiempo de entrega predicho para {distancia} km en la zona {zona}: {tiempo_predicho:.2f} minutos")