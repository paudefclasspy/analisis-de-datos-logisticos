from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import json
from typing import List, Dict, Any, Optional
import uvicorn

app = FastAPI(title="API de Entregas", description="API para gestionar y analizar datos de entregas")

# Modelos Pydantic para validación de datos
class EntregaRequest(BaseModel):
    n_entregas: int = 5000

class PrediccionRequest(BaseModel):
    zona: str
    distancia_km: float
    dia_semana: int
    eficiencia: Optional[float] = 1.0

class EntregaData(BaseModel):
    fecha: str
    zona: str
    distancia_km: float
    tiempo_estimado_min: float
    tiempo_real_min: float
    costo_total: float
    estado: str

# Variables globales para almacenar el modelo y los datos
modelo_global = None
df_entregas_global = None

# Funciones del código original adaptadas para API
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
            'fecha': fecha.strftime("%Y-%m-%d"),
            'zona': zona,
            'distancia_km': distancia,
            'tiempo_estimado_min': round(tiempo_estimado, 2),
            'tiempo_real_min': tiempo_real,
            'costo_total': costo_total,
            'estado': estado
        })
    return pd.DataFrame(datos)

def optimizar_modelo(df):
    # Convertir columna fecha a datetime si viene como string
    if isinstance(df['fecha'].iloc[0], str):
        df['fecha'] = pd.to_datetime(df['fecha'])
    
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
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(modelo, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    mejor_modelo = grid_search.best_estimator_
    y_pred_test = mejor_modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)

    return mejor_modelo, mse, X.columns.tolist()

def predecir_tiempo(modelo, columnas, zona, distancia_km, dia_semana, eficiencia=1.0):
    # Crear un DataFrame con una fila
    zonas = ['Norte', 'Sur', 'Este', 'Oeste', 'Centro']
    
    if zona not in zonas:
        raise HTTPException(status_code=400, detail=f"Zona inválida. Zonas permitidas: {zonas}")
    
    if dia_semana < 0 or dia_semana > 6:
        raise HTTPException(status_code=400, detail="Día de la semana debe estar entre 0 (lunes) y 6 (domingo)")
    
    # Crear un diccionario con las características necesarias
    data = {
        'distancia_km': distancia_km,
        'dia_semana': dia_semana,
        'eficiencia': eficiencia
    }
    
    # Agregar columnas para zonas (one-hot encoding)
    for z in zonas[1:]:  # Excluimos 'Norte' ya que suele ser la zona de referencia en drop_first=True
        zona_col = f'zona_{z}'
        if zona_col in columnas:
            data[zona_col] = 1 if zona == z else 0
    
    # Asegurarnos que el diccionario tiene todas las columnas necesarias
    for col in columnas:
        if col not in data:
            data[col] = 0
    
    # Crear DataFrame y ordenar las columnas según lo esperado por el modelo
    input_data = pd.DataFrame([data])
    input_data = input_data[columnas]
    
    # Realizar la predicción
    tiempo_predicho = modelo.predict(input_data)[0]
    
    return tiempo_predicho

def analizar_datos(df, filtro_distancia=None):
    if filtro_distancia:
        df_filtrado = df[df['distancia_km'] <= filtro_distancia]
    else:
        df_filtrado = df.copy()
    
    # Convertir fecha a datetime si es necesario
    if isinstance(df_filtrado['fecha'].iloc[0], str):
        df_filtrado['fecha'] = pd.to_datetime(df_filtrado['fecha'])
    
    # Análisis por zona
    tiempo_por_zona = df_filtrado.groupby('zona')['tiempo_real_min'].mean().to_dict()
    costo_por_zona = df_filtrado.groupby('zona')['costo_total'].mean().to_dict()
    
    # Análisis de estados
    estado_counts = df_filtrado['estado'].value_counts().to_dict()
    
    # Análisis por día de la semana
    df_filtrado['dia_semana'] = df_filtrado['fecha'].dt.weekday
    dias_mapping = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
    df_filtrado['nombre_dia'] = df_filtrado['dia_semana'].map(dias_mapping)
    tiempo_por_dia = df_filtrado.groupby('nombre_dia')['tiempo_real_min'].mean().reindex(
        [dias_mapping[i] for i in range(7)]).to_dict()
    
    return {
        "tiempo_por_zona": tiempo_por_zona,
        "costo_por_zona": costo_por_zona,
        "estado_entregas": estado_counts,
        "tiempo_por_dia": tiempo_por_dia,
        "n_entregas_analizadas": len(df_filtrado),
        "distancia_promedio": df_filtrado['distancia_km'].mean(),
        "tiempo_promedio": df_filtrado['tiempo_real_min'].mean(),
        "costo_promedio": df_filtrado['costo_total'].mean()
    }

# Rutas de la API
@app.get("/")
def read_root():
    return {"mensaje": "API de Entregas v1.0"}

@app.post("/generar-datos/", response_model=Dict[str, Any])
def generar_datos(request: EntregaRequest):
    global df_entregas_global
    
    df_entregas_global = generar_datos_entregas(request.n_entregas)
    
    # Convertir el DataFrame a una lista de diccionarios
    muestra_datos = df_entregas_global.head(5).to_dict(orient="records")
    
    return {
        "mensaje": f"Se generaron {request.n_entregas} registros de entregas correctamente",
        "total_registros": len(df_entregas_global),
        "muestra_datos": muestra_datos
    }

@app.get("/datos/", response_model=List[EntregaData])
def obtener_datos(limit: int = Query(10, ge=1, le=100)):
    global df_entregas_global
    
    try:
        if df_entregas_global is None:
            # Log para depuración
            print("Generando datos predeterminados ya que df_entregas_global es None")
            df_entregas_global = generar_datos_entregas(1000)  # Generar datos predeterminados
        
        # Asegurarse de que las fechas estén en formato string
        if not isinstance(df_entregas_global['fecha'].iloc[0], str):
            df_entregas_global['fecha'] = df_entregas_global['fecha'].dt.strftime("%Y-%m-%d")
        
        # Convertir a diccionario con orientación "records" y limitar por parámetro
        datos = df_entregas_global.head(limit).to_dict(orient="records")
        
        # Verificar la estructura de los datos para depuración
        print(f"Muestra de datos: {datos[0] if datos else 'No hay datos'}")
        
        return datos
    except Exception as e:
        # Log del error
        print(f"Error en /datos/: {str(e)}")
        # Re-lanzar para que FastAPI pueda manejarlo
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.post("/entrenar-modelo/", response_model=Dict[str, Any])
def entrenar_modelo():
    global df_entregas_global, modelo_global, columnas_modelo
    
    if df_entregas_global is None:
        df_entregas_global = generar_datos_entregas(1000)  # Generar datos predeterminados
    
    modelo_global, mse, columnas_modelo = optimizar_modelo(df_entregas_global)
    
    return {
        "mensaje": "Modelo entrenado correctamente",
        "mse": mse,
        "parametros_modelo": str(modelo_global.get_params())
    }

@app.post("/predecir/", response_model=Dict[str, Any])
def predecir(request: PrediccionRequest):
    global modelo_global, columnas_modelo
    
    if modelo_global is None:
        return {"error": "Primero debe entrenar el modelo con /entrenar-modelo/"}
    
    tiempo_predicho = predecir_tiempo(
        modelo_global,
        columnas_modelo,
        request.zona,
        request.distancia_km,
        request.dia_semana,
        request.eficiencia
    )
    
    return {
        "tiempo_predicho_minutos": round(tiempo_predicho, 2),
        "zona": request.zona,
        "distancia_km": request.distancia_km,
        "dia_semana": request.dia_semana,
        "eficiencia_estimada": request.eficiencia
    }

@app.get("/analisis/", response_model=Dict[str, Any])
def obtener_analisis(distancia_maxima: Optional[float] = None):
    global df_entregas_global
    
    if df_entregas_global is None:
        df_entregas_global = generar_datos_entregas(1000)  # Generar datos predeterminados
    
    resultados_analisis = analizar_datos(df_entregas_global, distancia_maxima)
    
    return resultados_analisis

# Inicializar variables globales con datos predeterminados
@app.on_event("startup")
def startup_event():
    global df_entregas_global, modelo_global, columnas_modelo
    
    # Generar datos iniciales
    df_entregas_global = generar_datos_entregas(1000)
    
    # Entrenar modelo inicial
    modelo_global, _, columnas_modelo = optimizar_modelo(df_entregas_global)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)