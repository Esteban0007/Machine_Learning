import gradio as gr
import pandas as pd
from prophet import Prophet
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

# Cargar los modelos entrenados para cada isla
modelos = {
    "Mallorca": pickle.load(open("Models/model_Mallorca.pkl", "rb")),
    "Menorca": pickle.load(open("Models/model_Menorca.pkl", "rb")),
    "Ibiza": pickle.load(open("Models/model_Ibiza.pkl", "rb")),
}

# Mapeo de los nombres de los meses
meses = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4,
    "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8,
    "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
}

# Función para realizar la predicción y generar los gráficos
def predecir_turistas(isla, año, mes_nombre):
    # Seleccionar el modelo de la isla elegida
    modelo = modelos[isla]
    mes = meses[mes_nombre]  # Convertir el nombre del mes a número
    
    # Crear una fecha con el año y mes ingresado
    fecha = pd.to_datetime(f"{año}-{mes}-01")
    
    # Generar predicciones extendidas en el tiempo
    future_anual = modelo.make_future_dataframe(periods=365 * (año - datetime.now().year + 1), freq='D')
    
    # Realizar predicciones sobre el periodo extendido
    forecast_anual = modelo.predict(future_anual)
    
    # Filtrar la predicción específica del mes y año seleccionados
    prediccion_mes = forecast_anual[(forecast_anual['ds'].dt.year == año) & (forecast_anual['ds'].dt.month == mes)]
    if prediccion_mes.empty:
        return "No hay predicción disponible para esta fecha.", None, None, None
    
    prediccion = prediccion_mes['yhat'].values[0]
    
    # Filtrar las predicciones del año completo para el gráfico
    forecast_anio = forecast_anual[forecast_anual['ds'].dt.year == año]
    
    # Generar el gráfico de predicciones para el año completo
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_anio['ds'], forecast_anio['yhat'], color='red', label="Predicciones")
    plt.fill_between(forecast_anio['ds'], forecast_anio['yhat_lower'], forecast_anio['yhat_upper'], color='pink', alpha=0.3, label="Intervalo de confianza")
    plt.xlabel("Fecha")
    plt.ylabel("Número de Turistas")
    plt.title(f"Predicción de turistas en {isla} para {año}")
    plt.legend()
    plt.grid(True)
    
    # Guardar el gráfico anual
    plt.savefig("prediccion_anual.png")
    plt.close()
    
    # Generar el gráfico por isla
    fig_isla = modelo.plot(forecast_anual)
    ax_isla = fig_isla.gca()
    ax_isla.lines[0].set_color('blue')  # Datos reales
    ax_isla.lines[0].set_linestyle('-')
    ax_isla.lines[1].set_color('red')   # Predicciones
    plt.title(f'Predicción de turistas en {isla}')
    
    # Guardar el gráfico por isla
    plt.savefig(f'grafico_prediccion_{isla}.png')
    plt.close()

    # Filtrar las predicciones solo para el mes específico
    forecast_mes = forecast_anual[(forecast_anual['ds'].dt.year == año) & (forecast_anual['ds'].dt.month == mes)]
    
    # Generar el gráfico de predicción solo para el mes
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_mes['ds'], forecast_mes['yhat'], color='orange', label="Predicción del Mes")
    plt.fill_between(forecast_mes['ds'], forecast_mes['yhat_lower'], forecast_mes['yhat_upper'], color='yellow', alpha=0.3, label="Intervalo de confianza")
    plt.xlabel("Fecha")
    plt.ylabel("Número de Turistas")
    plt.title(f"Predicción de turistas en {isla} para {mes_nombre} {año}")
    plt.legend()
    plt.grid(True)
    
    # Guardar el gráfico del mes
    plt.savefig(f'grafico_prediccion_mes_{isla}_{año}_{mes_nombre}.png')
    plt.close()
    
    # Formatear la salida
    prediccion_texto = f"Predicción para {isla} en {mes_nombre} {año}: {int(prediccion):,} turistas"
    return prediccion_texto, "prediccion_anual.png", f'grafico_prediccion_{isla}.png', f'grafico_prediccion_mes_{isla}_{año}_{mes_nombre}.png'

# Crear la interfaz Gradio
interfaz = gr.Interface(
    fn=predecir_turistas,
    inputs=[
        gr.Dropdown(choices=['Mallorca', 'Menorca', 'Ibiza'], label="Isla"),
        gr.Number(label="Año", value=2024, precision=0),
        gr.Dropdown(choices=list(meses.keys()), label="Mes")
    ],
    outputs=[
        gr.Textbox(label="Predicción del Número de Turistas"),
        gr.Image(label="Gráfico de Predicción Anual"),
        gr.Image(label="Gráfico de Predicción por Isla"),
        gr.Image(label="Gráfico de Predicción del Mes")
    ],
    title="Predicción de turistas por medio aéreo en las Islas Baleares",
    description=(
    "Seleccione una isla, un año y un mes para obtener la predicción del número de turistas y los gráficos correspondientes.\n\n"
    "He desarrollado un modelo predictivo que permite anticipar el turismo aéreo en las Islas Baleares.\n"
    "Para entrenar los modelos por isla, he construido una base de datos (representada en la línea azul) y he limpiado los datos (como los efectos del COVID).\n"
    "Los datos fueron obtenidos de la Agencia de Estrategia Turística de las Islas Baleares, "
    "Consejería de Modelo Económico, Turismo y Trabajo del Gobierno de las Islas Baleares.\n"
    "Los datos son recopilados de AENA, específicamente para los pasajeros por vía aérea.\n\n"
    "¡Interactúe con la herramienta para explorar las tendencias turísticas!"
)

)

# Lanzar la interfaz
interfaz.launch()
