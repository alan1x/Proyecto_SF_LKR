"""
Dashboard Ejecutivo de Análisis Multivariado de Ventas
======================================================
Presentación interactiva de resultados del análisis de ventas
Incluye: Clustering de Productos, Clustering de Clientes, Modelos Predictivos, Series de Tiempo
"""

import dash
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# Intentar importar Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
    print("✅ Prophet disponible")
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet no instalado. Ejecuta: pip install prophet")

# Importar skforecast
try:
    from skforecast.ForecasterAutoreg import ForecasterAutoreg
    from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
    from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
    from skforecast.model_selection import grid_search_forecaster, backtesting_forecaster
    SKFORECAST_AVAILABLE = True
except ImportError:
    SKFORECAST_AVAILABLE = False
    print("⚠️ skforecast no instalado. Ejecuta: pip install skforecast")

# =============================================================================
# CONFIGURACIÓN DE COLORES Y ESTILOS
# =============================================================================

# =============================================================================
# CONFIGURACIÓN DE COLORES Y ESTILOS (ACTUALIZAR)
# =============================================================================

COLORS = {
    'primary': '#3498db',
    'secondary': '#2ecc71',
    'accent': '#e74c3c',
    'warning': '#f39c12',
    'purple': '#9b59b6',
    'dark': '#2c3e50',
    'light': '#ecf0f1',
    'clusters': ['#ff0000', '#31688e', '#61fdff', '#ffff71'],
    'clusters_clientes': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#F8FC00']
}

# Mapeo de colores para clusters de PRODUCTOS
CLUSTER_PRODUCTO_COLORS = {
    'Bajo Rendimiento': '#ff0000',
    'Alto Stock': '#31688e',
    'Estrella': '#61fdff',
    'Nicho': '#ffff71'
}

# Mapeo fijo de segmentos a colores (para clientes)
SEGMENTO_COLORS = {
    'Premium y Frecuentes': '#e41a1c',
    'Exploradores de Nicho': '#377eb8',
    'Clientes Estables': '#4daf4a',
    'Ocasionales Económicos': '#984ea3',
    'Cazadores de Oferta': '#ff7f00',
    'Nuevos o Dormidos': '#F8FC00'
}
# =============================================================================
# CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

def cargar_datos():
    """Carga y prepara todos los datos necesarios"""
    categorias = pd.read_csv("../data_raw/categorias.csv")
    clientes = pd.read_csv("../data_raw/clientes.csv")
    productos = pd.read_csv("../data_raw/productos.csv")
    ventas = pd.read_csv("../data_raw/ventas.csv")
    metodos_pago = pd.read_csv("../data_raw/metodos_pago.csv")
    
    productos["Precio_Unitario"] = productos["Precio_Unitario"].str.replace(",", ".").astype(float)
    
    df = ventas.merge(clientes, on="ID_Cliente", how="inner")
    if "Email" in df.columns:
        df.drop(columns=["Email"], inplace=True)
    df = df.merge(productos[["ID_Producto", "Precio_Unitario"]], on="ID_Producto", how="inner")
    df["Cantidad_dinero"] = df["Cantidad"] * df["Precio_Unitario"]
    df = df.merge(productos[["ID_Producto", "Stock"]], on="ID_Producto", how="inner")
    df = df.merge(productos[["ID_Producto", "Categoría"]], on="ID_Producto", how="left")
    
    # Verificar si existe ID_Metodo o Metodo_Pago para hacer el merge
    if "ID_Metodo" in df.columns and "ID_Metodo" in metodos_pago.columns:
        df = df.merge(metodos_pago, on="ID_Metodo", how="left")
    elif "Metodo_Pago" in df.columns:
        pass  # Ya tiene la info de método de pago
    elif "ID_Metodo_Pago" in df.columns and "ID_Metodo_Pago" in metodos_pago.columns:
        df = df.merge(metodos_pago, on="ID_Metodo_Pago", how="left")
    # Si no hay columna de método, continuar sin ella
    
    dict_stock = productos.set_index("ID_Producto")["Stock"].to_dict()
    cantidad_de_producto_actual = []
    for cantidad, id_producto in zip(df["Cantidad"], df["ID_Producto"]):
        dict_stock[id_producto] -= cantidad
        cantidad_de_producto_actual.append(dict_stock[id_producto])
    df["Producto_actual_stock"] = cantidad_de_producto_actual
    
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d/%m/%Y")
    df["Dia_semana"] = df["Fecha"].dt.day_name()
    df["Mes"] = df["Fecha"].dt.month
    df["Año"] = df["Fecha"].dt.year
    
    productos_vendidos = df.groupby("ID_Producto")["Producto_actual_stock"].min().reset_index()
    productos_vendidos = productos_vendidos.merge(productos[["ID_Producto", "Stock"]], on="ID_Producto", how="inner")
    productos_vendidos["Cantidad_vendida"] = productos_vendidos["Stock"] - productos_vendidos["Producto_actual_stock"]
    
    return df, productos, productos_vendidos, categorias, clientes, metodos_pago

def preparar_clusters_productos(df, productos_vendidos):
    """Prepara datos de clustering de productos"""
    cluster_data = productos_vendidos[['ID_Producto', 'Cantidad_vendida', 'Stock', 'Producto_actual_stock']].copy()
    cluster_data["Porcentaje_venta"] = (cluster_data["Cantidad_vendida"] / cluster_data["Stock"]) * 100
    
    producto_categoria = df.groupby('ID_Producto')['Categoría'].first().reset_index()
    cluster_data = cluster_data.merge(producto_categoria, on='ID_Producto', how='left')
    
    producto_region = df.groupby('ID_Producto')['Región'].agg(lambda x: x.value_counts().index[0]).reset_index()
    producto_region.columns = ['ID_Producto', 'Region_principal']
    cluster_data = cluster_data.merge(producto_region, on='ID_Producto', how='left')
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_data[['Cantidad_vendida', 'Stock', 'Producto_actual_stock', 'Porcentaje_venta']])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_data['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Asignar nombres descriptivos a clusters
    cluster_names = {
        0: 'Bajo Rendimiento',
        1: 'Alto Stock',
        2: 'Estrella',
        3: 'Nicho'
    }
    cluster_data['Cluster_Nombre'] = cluster_data['Cluster'].map(cluster_names)
    
    return cluster_data, kmeans, scaler

def preparar_clusters_clientes(df):
    """Prepara datos de clustering de clientes - K=6 como en el notebook"""
    df_temp = df.copy()
    
    # Calcular fecha de referencia
    fecha_referencia = df_temp['Fecha'].max() + pd.Timedelta(days=1)
    
    # Calcular métricas por cliente (como en el notebook)
    cliente_stats = df_temp.groupby('ID_Cliente').agg(
        frecuencia_compras=('ID_Venta', 'count'),              # Cuántas veces compra
        total_comprado=('Cantidad', 'sum'),                    # Cuántos productos compra
        gasto_total=('Cantidad_dinero', 'sum'),                # Cuánto dinero ha gastado
        categorias_distintas=('Categoría', pd.Series.nunique), # Cuántas categorías distintas
        promedio_gasto_por_compra=('Cantidad_dinero', 'mean'), # Ticket promedio
        region=('Región', 'first')                             # Región geográfica
    ).reset_index()
    
    # Crear dummies para región (como en el notebook)
    cliente_stats_clustering = pd.get_dummies(cliente_stats, columns=['region'], drop_first=True)
    
    # Features para clustering (sin ID_Cliente)
    X = cliente_stats_clustering.drop(columns=['ID_Cliente'])
    
    # Normalizar
    scaler_rfm = StandardScaler()
    X_scaled = scaler_rfm.fit_transform(X)
    
    # Clustering con K=6 (como en el notebook)
    kmeans_clientes = KMeans(n_clusters=6, random_state=42, n_init=10)
    cliente_stats['Cluster'] = kmeans_clientes.fit_predict(X_scaled)
    
    # Calcular estadísticas por cluster para asignar nombres
    cluster_stats = cliente_stats.groupby('Cluster').agg({
        'frecuencia_compras': 'mean',
        'gasto_total': ['mean', 'sum'],
        'categorias_distintas': 'mean',
        'promedio_gasto_por_compra': 'mean'
    })
    cluster_stats.columns = ['Frecuencia_mean', 'Gasto_mean', 'Gasto_sum', 'Categorias_mean', 'Ticket_mean']
    
    # Ordenar por gasto total para identificar clusters
    cluster_order = cluster_stats.sort_values('Gasto_sum', ascending=False).index.tolist()
    
    # Nombres según el notebook
    nombres_segmentos = [
        'Premium y Frecuentes',      # Mayor gasto, alta frecuencia
        'Exploradores de Nicho',     # Gasto medio-alto, categorías específicas
        'Clientes Estables',         # Gasto medio, frecuencia regular
        'Ocasionales Económicos',    # Baja frecuencia, variedad
        'Cazadores de Oferta',       # Gasto medio-bajo, pocas categorías
        'Nuevos o Dormidos'          # Pocos clientes, bajo engagement
    ]
    
    nombre_mapping = {}
    for i, cluster_id in enumerate(cluster_order):
        nombre_mapping[cluster_id] = nombres_segmentos[i]
    
    cliente_stats['Segmento'] = cliente_stats['Cluster'].map(nombre_mapping)
    
    # Renombrar columnas para consistencia con el resto del dashboard
    cliente_stats = cliente_stats.rename(columns={
        'frecuencia_compras': 'Frequency',
        'gasto_total': 'Monetary',
        'categorias_distintas': 'Categorias',
        'total_comprado': 'Total_Productos',
        'promedio_gasto_por_compra': 'Ticket_Promedio'
    })
    
    # Agregar columna de color para consistencia
    cliente_stats['Color'] = cliente_stats['Segmento'].map(SEGMENTO_COLORS)
    
    return cliente_stats, kmeans_clientes, scaler_rfm

# =============================================================================
# MODIFICAR FUNCIÓN entrenar_modelos PARA GUARDAR IDs DE PRODUCTOS
# =============================================================================

# =============================================================================
# MODIFICAR FUNCIÓN entrenar_modelos PARA PREDECIR TODOS LOS PRODUCTOS
# =============================================================================

def entrenar_modelos(df, cluster_data):
    """Entrena modelos de predicción - MODIFICADO para predecir TODOS los productos"""
    df_regression = df.groupby('ID_Producto').agg({
        'Cantidad': 'sum',
        'Cantidad_dinero': 'sum',
        'Precio_Unitario': 'first',
        'Región': lambda x: x.value_counts().index[0],
        'Categoría': 'first'
    }).reset_index()
    
    df_regression = df_regression.merge(
        cluster_data[['ID_Producto', 'Cluster', 'Cluster_Nombre', 'Stock', 'Cantidad_vendida', 'Porcentaje_venta', 'Producto_actual_stock']], 
        on='ID_Producto'
    )
    
    # Guardar info de productos ANTES de get_dummies
    productos_info = df_regression[['ID_Producto', 'Categoría', 'Cluster_Nombre', 'Cantidad_dinero', 'Precio_Unitario', 'Stock']].copy()
    productos_info.columns = ['ID_Producto', 'Categoria', 'Segmento', 'Ingreso_Real', 'Precio', 'Stock']
    
    # Guardar IDs de todos los productos
    todos_los_ids = df_regression['ID_Producto'].values.copy()
    
    df_regression = pd.get_dummies(df_regression, columns=['Región', 'Categoría'], drop_first=True)
    
    feature_cols = [col for col in df_regression.columns if col not in 
                   ['ID_Producto', 'Cluster', 'Cluster_Nombre', 'Cantidad_dinero', 'Cantidad_vendida', 'Cantidad']]
    
    X = df_regression[feature_cols]
    y = df_regression['Cantidad_dinero']
    producto_ids = df_regression['ID_Producto'].values
    
    # Split para evaluación del modelo (solo para métricas)
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, producto_ids, test_size=0.25, random_state=42
    )
    
    modelos = {}
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb_test = gb.predict(X_test)
    y_pred_gb_all = gb.predict(X)  # Predicción para TODOS los productos
    modelos['Gradient Boosting'] = {
        'model': gb,
        'predictions': y_pred_gb_test,
        'predictions_all': y_pred_gb_all,  # Predicciones de todos
        'metrics': {
            'R2': r2_score(y_test, y_pred_gb_test),
            'RMSE': mean_squared_error(y_test, y_pred_gb_test) ** 0.5,
            'MAE': mean_absolute_error(y_test, y_pred_gb_test)
        }
    }
    
    # AdaBoost
    ada = AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    ada.fit(X_train, y_train)
    y_pred_ada_test = ada.predict(X_test)
    y_pred_ada_all = ada.predict(X)
    modelos['AdaBoost'] = {
        'model': ada,
        'predictions': y_pred_ada_test,
        'predictions_all': y_pred_ada_all,
        'metrics': {
            'R2': r2_score(y_test, y_pred_ada_test),
            'RMSE': mean_squared_error(y_test, y_pred_ada_test) ** 0.5,
            'MAE': mean_absolute_error(y_test, y_pred_ada_test)
        }
    }
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf_test = rf.predict(X_test)
    y_pred_rf_all = rf.predict(X)
    modelos['Random Forest'] = {
        'model': rf,
        'predictions': y_pred_rf_test,
        'predictions_all': y_pred_rf_all,
        'metrics': {
            'R2': r2_score(y_test, y_pred_rf_test),
            'RMSE': mean_squared_error(y_test, y_pred_rf_test) ** 0.5,
            'MAE': mean_absolute_error(y_test, y_pred_rf_test)
        }
    }
    
    # SVR
    scaler_svr = StandardScaler()
    X_train_scaled = scaler_svr.fit_transform(X_train)
    X_test_scaled = scaler_svr.transform(X_test)
    X_all_scaled = scaler_svr.transform(X)
    
    svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
    svr.fit(X_train_scaled, y_train)
    y_pred_svr_test = svr.predict(X_test_scaled)
    y_pred_svr_all = svr.predict(X_all_scaled)
    modelos['SVR'] = {
        'model': svr,
        'predictions': y_pred_svr_test,
        'predictions_all': y_pred_svr_all,
        'metrics': {
            'R2': r2_score(y_test, y_pred_svr_test),
            'RMSE': mean_squared_error(y_test, y_pred_svr_test) ** 0.5,
            'MAE': mean_absolute_error(y_test, y_pred_svr_test)
        },
        'scaler': scaler_svr
    }
    
    # Guardar datos completos para predicciones
    datos_modelo = {
        'X_train': X_train,
        'X_test': X_test,
        'X_all': X,  # Todas las features
        'y_train': y_train,
        'y_test': y_test,
        'y_all': y,  # Todos los valores reales
        'ids_train': ids_train,
        'ids_test': ids_test,
        'ids_all': todos_los_ids,  # Todos los IDs
        'feature_cols': feature_cols,
        'productos_info': productos_info,
        'df_regression': df_regression
    }
    
    return modelos, feature_cols, X_test, y_test, datos_modelo

def entrenar_skforecast(df_filtrado, forecast_days=30, modelo_tipo='gradient_boosting'):
    """
    Entrena modelo de series temporales usando skforecast
    
    Parámetros:
    - df_filtrado: DataFrame con los datos
    - forecast_days: días a pronosticar
    - modelo_tipo: 'gradient_boosting', 'random_forest', 'ridge'
    """
    if not SKFORECAST_AVAILABLE:
        return None, "skforecast no está instalado. Ejecuta: pip install skforecast"
    
    try:
        # Preparar serie temporal
        ventas_diarias = df_filtrado.groupby('Fecha')['Cantidad_dinero'].sum().reset_index()
        ventas_diarias = ventas_diarias.sort_values('Fecha')
        ventas_diarias.set_index('Fecha', inplace=True)
        ventas_diarias = ventas_diarias.asfreq('D')
        ventas_diarias = ventas_diarias.fillna(method='ffill').fillna(0)
        
        if len(ventas_diarias) < 30:
            return None, "Datos insuficientes (mínimo 30 días)"
        
        # Seleccionar modelo base
        if modelo_tipo == 'gradient_boosting':
            regressor = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            modelo_nombre = 'Gradient Boosting'
        elif modelo_tipo == 'random_forest':
            regressor = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            modelo_nombre = 'Random Forest'
        else:
            regressor = Ridge(alpha=1.0)
            modelo_nombre = 'Ridge Regression'
        
        # Determinar lags óptimos (últimos 7-14 días)
        lags = min(14, len(ventas_diarias) // 3)
        
        # Crear forecaster autoregresivo
        forecaster = ForecasterAutoreg(
            regressor=regressor,
            lags=lags
        )
        
        # Entrenar
        forecaster.fit(y=ventas_diarias['Cantidad_dinero'])
        
        # Generar predicciones
        predicciones = forecaster.predict(steps=forecast_days)
        
        # Calcular intervalos de confianza usando bootstrapping
        # (aproximación con desviación estándar de residuos)
        residuos = ventas_diarias['Cantidad_dinero'].iloc[lags:] - forecaster.predict(steps=len(ventas_diarias) - lags).values
        std_residuos = residuos.std() if len(residuos) > 0 else predicciones.std() * 0.1
        
        # Intervalos de confianza al 95%
        conf_int_lower = predicciones - 1.96 * std_residuos
        conf_int_upper = predicciones + 1.96 * std_residuos
        
        # Asegurar que los límites inferiores no sean negativos
        conf_int_lower = np.maximum(conf_int_lower, 0)
        
        # Obtener importancia de características (lags)
        if hasattr(regressor, 'feature_importances_'):
            feature_importance = dict(zip(
                [f'lag_{i}' for i in range(1, lags + 1)],
                regressor.feature_importances_
            ))
        else:
            feature_importance = None
        
        return {
            'historico': ventas_diarias,
            'forecast': predicciones.values,
            'forecast_index': predicciones.index,
            'conf_int_lower': conf_int_lower.values,
            'conf_int_upper': conf_int_upper.values,
            'model_name': f'skforecast ({modelo_nombre})',
            'lags': lags,
            'feature_importance': feature_importance,
            'std_residuos': std_residuos
        }, None
        
    except Exception as e:
        return None, f"Error en skforecast: {str(e)}"


def entrenar_skforecast_directo(df_filtrado, forecast_days=30):
    """
    Entrena modelo usando ForecasterAutoregDirect para predicciones multi-step
    Más preciso para horizontes largos
    """
    if not SKFORECAST_AVAILABLE:
        return None, "skforecast no está instalado"
    
    try:
        # Preparar serie temporal
        ventas_diarias = df_filtrado.groupby('Fecha')['Cantidad_dinero'].sum().reset_index()
        ventas_diarias = ventas_diarias.sort_values('Fecha')
        ventas_diarias.set_index('Fecha', inplace=True)
        ventas_diarias = ventas_diarias.asfreq('D')
        ventas_diarias = ventas_diarias.fillna(method='ffill').fillna(0)
        
        if len(ventas_diarias) < 30:
            return None, "Datos insuficientes (mínimo 30 días)"
        
        lags = min(14, len(ventas_diarias) // 3)
        
        # ForecasterAutoregDirect: un modelo por cada step
        forecaster = ForecasterAutoregDirect(
            regressor=Ridge(alpha=1.0),
            steps=forecast_days,
            lags=lags
        )
        
        forecaster.fit(y=ventas_diarias['Cantidad_dinero'])
        predicciones = forecaster.predict()
        
        # Intervalos aproximados
        std_approx = ventas_diarias['Cantidad_dinero'].std() * 0.15
        conf_int_lower = np.maximum(predicciones - 1.96 * std_approx, 0)
        conf_int_upper = predicciones + 1.96 * std_approx
        
        return {
            'historico': ventas_diarias,
            'forecast': predicciones.values,
            'forecast_index': predicciones.index,
            'conf_int_lower': conf_int_lower.values,
            'conf_int_upper': conf_int_upper.values,
            'model_name': 'skforecast Direct (Multi-step)',
            'lags': lags,
            'steps': forecast_days
        }, None
        
    except Exception as e:
        return None, f"Error: {str(e)}"


def entrenar_prophet(df_filtrado, forecast_days=30, modelo_tipo='default'):
    """
    Entrena modelo Prophet y genera forecast
    
    Parámetros:
    - df_filtrado: DataFrame con los datos
    - forecast_days: días a pronosticar
    - modelo_tipo: 'default', 'multiplicative', 'logistic'
    """
    if not PROPHET_AVAILABLE:
        return None, "Prophet no está instalado. Ejecuta: pip install prophet"
    
    try:
        # Preparar datos en formato Prophet (ds, y)
        ventas_diarias = df_filtrado.groupby('Fecha')['Cantidad_dinero'].sum().reset_index()
        ventas_diarias.columns = ['ds', 'y']
        ventas_diarias = ventas_diarias.sort_values('ds')
        
        if len(ventas_diarias) < 14:
            return None, "Datos insuficientes para Prophet (mínimo 14 días)"
        
        # Configurar Prophet según el tipo de modelo
        if modelo_tipo == 'multiplicative':
            model = Prophet(
                seasonality_mode='multiplicative',
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                interval_width=0.95
            )
            modelo_nombre = 'Prophet (Multiplicativo)'
        elif modelo_tipo == 'logistic':
            # Para crecimiento logístico, necesitamos cap y floor
            ventas_diarias['cap'] = ventas_diarias['y'].max() * 1.5
            ventas_diarias['floor'] = 0
            model = Prophet(
                growth='logistic',
                seasonality_mode='additive',
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.1,
                interval_width=0.95
            )
            modelo_nombre = 'Prophet (Logístico)'
        else:
            model = Prophet(
                seasonality_mode='additive',
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                interval_width=0.95
            )
            modelo_nombre = 'Prophet (Aditivo)'
        
        # Agregar estacionalidad mensual
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        # Entrenar modelo (silenciar output)
        model.fit(ventas_diarias)
        
        # Crear dataframe futuro
        future = model.make_future_dataframe(periods=forecast_days)
        
        # Si es logístico, agregar cap y floor al futuro
        if modelo_tipo == 'logistic':
            future['cap'] = ventas_diarias['cap'].iloc[0]
            future['floor'] = 0
        
        # Generar predicciones
        forecast = model.predict(future)
        
        # Separar histórico y pronóstico
        fecha_corte = ventas_diarias['ds'].max()
        pronostico = forecast[forecast['ds'] > fecha_corte]
        
        return {
            'historico': ventas_diarias.set_index('ds'),
            'historico_col': 'y',
            'forecast': pronostico['yhat'].values,
            'forecast_index': pronostico['ds'].values,
            'conf_int_lower': pronostico['yhat_lower'].values,
            'conf_int_upper': pronostico['yhat_upper'].values,
            'model_name': modelo_nombre,
            'model': model,
            'full_forecast': forecast,
            'trend': forecast['trend'].values,
            'weekly': forecast['weekly'].values if 'weekly' in forecast.columns else None,
        }, None
        
    except Exception as e:
        return None, f"Error en Prophet: {str(e)}"


def entrenar_prophet_con_regresores(df_filtrado, forecast_days=30):
    """
    Prophet con regresores adicionales (día de semana, mes)
    """
    if not PROPHET_AVAILABLE:
        return None, "Prophet no está instalado"
    
    try:
        ventas_diarias = df_filtrado.groupby('Fecha')['Cantidad_dinero'].sum().reset_index()
        ventas_diarias.columns = ['ds', 'y']
        ventas_diarias = ventas_diarias.sort_values('ds')
        
        if len(ventas_diarias) < 14:
            return None, "Datos insuficientes"
        
        # Agregar regresores
        ventas_diarias['es_fin_semana'] = ventas_diarias['ds'].dt.dayofweek.isin([5, 6]).astype(int)
        ventas_diarias['dia_mes'] = ventas_diarias['ds'].dt.day
        
        model = Prophet(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.1,
            interval_width=0.95
        )
        
        # Agregar regresores
        model.add_regressor('es_fin_semana')
        model.add_regressor('dia_mes')
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        model.fit(ventas_diarias)
        
        # Crear futuro con regresores
        future = model.make_future_dataframe(periods=forecast_days)
        future['es_fin_semana'] = future['ds'].dt.dayofweek.isin([5, 6]).astype(int)
        future['dia_mes'] = future['ds'].dt.day
        
        forecast = model.predict(future)
        
        fecha_corte = ventas_diarias['ds'].max()
        pronostico = forecast[forecast['ds'] > fecha_corte]
        
        return {
            'historico': ventas_diarias.set_index('ds'),
            'historico_col': 'y',
            'forecast': pronostico['yhat'].values,
            'forecast_index': pronostico['ds'].values,
            'conf_int_lower': pronostico['yhat_lower'].values,
            'conf_int_upper': pronostico['yhat_upper'].values,
            'model_name': 'Prophet (Con Regresores)',
            'model': model,
        }, None
        
    except Exception as e:
        return None, f"Error: {str(e)}"


# Mantener ARIMA como fallback
def entrenar_arima(df_filtrado, forecast_days=30):
    """Entrena modelo ARIMA y genera forecast (fallback)"""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller
        
        ventas_diarias = df_filtrado.groupby('Fecha')['Cantidad_dinero'].sum().reset_index()
        ventas_diarias = ventas_diarias.sort_values('Fecha')
        ventas_diarias.set_index('Fecha', inplace=True)
        ventas_diarias = ventas_diarias.resample('D').sum().fillna(0)
        
        if len(ventas_diarias) < 10:
            return None, "Datos insuficientes para ARIMA"
        
        adf_result = adfuller(ventas_diarias['Cantidad_dinero'].dropna())
        d = 0 if adf_result[1] < 0.05 else 1
        
        model = ARIMA(ventas_diarias['Cantidad_dinero'], order=(2, d, 2))
        model_fit = model.fit()
        
        forecast = model_fit.forecast(steps=forecast_days)
        forecast_index = pd.date_range(
            start=ventas_diarias.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        forecast_ci = model_fit.get_forecast(steps=forecast_days)
        conf_int = forecast_ci.conf_int()
        
        return {
            'historico': ventas_diarias,
            'historico_col': 'Cantidad_dinero',
            'forecast': forecast.values,
            'forecast_index': forecast_index,
            'conf_int_lower': conf_int.iloc[:, 0].values,
            'conf_int_upper': conf_int.iloc[:, 1].values,
            'aic': model_fit.aic,
            'bic': model_fit.bic,
            'order': (2, d, 2),
            'model_name': f'ARIMA(2,{d},2)'
        }, None
        
    except Exception as e:
        return None, str(e)


def entrenar_mejor_forecast(df_filtrado, forecast_days=30, modelo_preferido='auto'):
    """
    Selecciona y entrena el mejor modelo de forecast disponible
    
    modelo_preferido: 'auto', 'prophet_default', 'prophet_mult', 'prophet_reg', 'arima'
    """
    if modelo_preferido == 'auto':
        # Intentar Prophet primero, luego ARIMA
        if PROPHET_AVAILABLE:
            resultado, error = entrenar_prophet(df_filtrado, forecast_days, 'default')
            if resultado:
                return resultado, None
        
        # Fallback a ARIMA
        return entrenar_arima(df_filtrado, forecast_days)
    
    elif modelo_preferido == 'prophet_default':
        return entrenar_prophet(df_filtrado, forecast_days, 'default')
    
    elif modelo_preferido == 'prophet_mult':
        return entrenar_prophet(df_filtrado, forecast_days, 'multiplicative')
    
    elif modelo_preferido == 'prophet_reg':
        return entrenar_prophet_con_regresores(df_filtrado, forecast_days)
    
    elif modelo_preferido == 'arima':
        return entrenar_arima(df_filtrado, forecast_days)
    
    else:
        return entrenar_prophet(df_filtrado, forecast_days, 'default')

def entrenar_holt_winters(df_filtrado, forecast_days=30):
    """Entrena modelo Holt-Winters (Triple Exponential Smoothing)"""
    try:
        ventas_diarias = df_filtrado.groupby('Fecha')['Cantidad_dinero'].sum().reset_index()
        ventas_diarias = ventas_diarias.sort_values('Fecha')
        ventas_diarias.set_index('Fecha', inplace=True)
        ventas_diarias = ventas_diarias.resample('D').sum().fillna(method='ffill').fillna(0)
        
        # Reemplazar ceros con pequeño valor para evitar errores
        ventas_diarias['Cantidad_dinero'] = ventas_diarias['Cantidad_dinero'].replace(0, 0.01)
        
        if len(ventas_diarias) < 14:
            return None, "Datos insuficientes para Holt-Winters (mínimo 14 días)"
        
        # Holt-Winters con estacionalidad multiplicativa
        seasonal_period = min(7, len(ventas_diarias) // 2)  # Semanal o menos si hay pocos datos
        
        model = ExponentialSmoothing(
            ventas_diarias['Cantidad_dinero'],
            seasonal_periods=seasonal_period,
            trend='add',
            seasonal='mul' if ventas_diarias['Cantidad_dinero'].min() > 0 else 'add',
            damped_trend=True
        )
        
        model_fit = model.fit(optimized=True)
        
        # Forecast
        forecast = model_fit.forecast(steps=forecast_days)
        
        forecast_index = pd.date_range(
            start=ventas_diarias.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        # Calcular intervalos de confianza (aproximación)
        residuals = model_fit.resid
        std_resid = residuals.std()
        conf_int_lower = forecast - 1.96 * std_resid
        conf_int_upper = forecast + 1.96 * std_resid
        
        return {
            'historico': ventas_diarias,
            'forecast': forecast.values,
            'forecast_index': forecast_index,
            'conf_int_lower': conf_int_lower.values,
            'conf_int_upper': conf_int_upper.values,
            'aic': model_fit.aic,
            'model_name': 'Holt-Winters'
        }, None
        
    except Exception as e:
        return None, f"Holt-Winters: {str(e)}"


def entrenar_mejor_modelo_temporal(df_filtrado, forecast_days=30, modelo_preferido='auto'):
    """
    Entrena el mejor modelo de series temporales disponible.
    Orden de preferencia: Prophet > SARIMA > Holt-Winters
    """
    errores = []
    
    if modelo_preferido == 'auto' or modelo_preferido == 'prophet':
        if PROPHET_AVAILABLE:
            resultado, error = entrenar_prophet(df_filtrado, forecast_days)
            if resultado:
                return resultado, None
            errores.append(f"Prophet: {error}")
    
    if modelo_preferido == 'auto' or modelo_preferido == 'sarima':
        resultado, error = entrenar_sarima(df_filtrado, forecast_days)
        if resultado:
            return resultado, None
        errores.append(error)
    
    if modelo_preferido == 'auto' or modelo_preferido == 'holt_winters':
        resultado, error = entrenar_holt_winters(df_filtrado, forecast_days)
        if resultado:
            return resultado, None
        errores.append(error)
    
    return None, " | ".join(errores)

# =============================================================================
# CARGAR DATOS
# =============================================================================

print("Cargando datos...")
df, productos, productos_vendidos, categorias, clientes_df, metodos_pago = cargar_datos()
print("Preparando clusters de productos...")
cluster_productos, kmeans_productos, scaler_productos = preparar_clusters_productos(df, productos_vendidos)
print("Preparando clusters de clientes...")
cluster_clientes, kmeans_clientes, scaler_clientes = preparar_clusters_clientes(df)
print("Entrenando modelos predictivos...")
modelos, feature_cols, X_test, y_test, datos_modelo = entrenar_modelos(df, cluster_productos)

# Extraer variables - ahora con TODOS los productos
ids_test = datos_modelo['ids_test']
ids_all = datos_modelo['ids_all']  # Todos los IDs
y_all = datos_modelo['y_all']  # Todos los valores reales
productos_info = datos_modelo['productos_info']

# =============================================================================
# CÁLCULOS GLOBALES
# =============================================================================

total_ventas = df['Cantidad_dinero'].sum()
total_unidades = df['Cantidad'].sum()
ticket_promedio = df.groupby('ID_Venta')['Cantidad_dinero'].sum().mean()
productos_unicos = df['ID_Producto'].nunique()
clientes_unicos = df['ID_Cliente'].nunique()
regiones_activas = df['Región'].nunique()
total_transacciones = df['ID_Venta'].nunique()

dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dias_espanol = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

best_model = max(modelos.keys(), key=lambda x: modelos[x]['metrics']['R2'])

# =============================================================================
# FUNCIONES DE GRÁFICAS ESTÁTICAS
# =============================================================================

def crear_kpi_card(titulo, valor, icono="📊", color="primary"):
    """Crea una tarjeta KPI estilizada"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span(icono, style={'fontSize': '2rem'}),
                html.H6(titulo, className="text-muted mt-2 mb-1", style={'fontSize': '0.85rem'}),
                html.H4(valor, className=f"text-{color} mb-0 fw-bold")
            ], className="text-center")
        ])
    ], className="h-100 shadow-sm border-0")

def crear_grafica_tendencia_global():
    """Gráfica de tendencia global de ventas"""
    ventas_temp = df.groupby('Fecha')['Cantidad_dinero'].sum().reset_index()
    ventas_temp = ventas_temp.sort_values('Fecha')
    ventas_temp['MA7'] = ventas_temp['Cantidad_dinero'].rolling(window=7, min_periods=1).mean()
    ventas_temp['MA30'] = ventas_temp['Cantidad_dinero'].rolling(window=30, min_periods=1).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ventas_temp['Fecha'], 
        y=ventas_temp['Cantidad_dinero'],
        mode='lines',
        name='Ingresos Diarios',
        line=dict(color=COLORS['primary'], width=1),
        opacity=0.5
    ))
    fig.add_trace(go.Scatter(
        x=ventas_temp['Fecha'], 
        y=ventas_temp['MA7'],
        mode='lines',
        name='Media Móvil 7 días',
        line=dict(color=COLORS['secondary'], width=2)
    ))
    fig.add_trace(go.Scatter(
        x=ventas_temp['Fecha'], 
        y=ventas_temp['MA30'],
        mode='lines',
        name='Media Móvil 30 días',
        line=dict(color=COLORS['accent'], width=2, dash='dash')
    ))
    fig.update_layout(
        template='plotly_dark',
        title=dict(text='Evolución Temporal de Ingresos', font=dict(size=16)),
        xaxis_title='Fecha',
        yaxis_title='Ingresos ($)',
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='x unified'
    )
    return fig

def crear_grafica_regiones():
    """Gráfica de ventas por región"""
    ventas_region = df.groupby('Región').agg({
        'Cantidad_dinero': 'sum',
        'ID_Venta': 'count'
    }).reset_index()
    ventas_region.columns = ['Región', 'Ingresos', 'Transacciones']
    ventas_region = ventas_region.sort_values('Ingresos', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=ventas_region['Región'],
        x=ventas_region['Ingresos'],
        orientation='h',
        marker=dict(
            color=ventas_region['Ingresos'],
            colorscale='Greens',
            showscale=False
        ),
        text=[f'${x:,.0f}' for x in ventas_region['Ingresos']],
        textposition='inside',
        hovertemplate='<b>%{y}</b><br>Ingresos: $%{x:,.2f}<extra></extra>'
    ))
    fig.update_layout(
        template='plotly_dark',
        title=dict(text='Ingresos por Región', font=dict(size=16)),
        xaxis_title='Ingresos ($)',
        margin=dict(l=20, r=20, t=50, b=20),
        height=350
    )
    return fig

def crear_grafica_clusters_productos_3d():
    """Gráfica 3D de clusters de productos"""
    fig = go.Figure()
    
    for cluster_id in sorted(cluster_productos['Cluster'].unique()):
        cluster_subset = cluster_productos[cluster_productos['Cluster'] == cluster_id]
        nombre = cluster_subset['Cluster_Nombre'].iloc[0] if 'Cluster_Nombre' in cluster_subset.columns else f'Cluster {cluster_id}'
        
        fig.add_trace(go.Scatter3d(
            x=cluster_subset['Stock'],
            y=cluster_subset['Cantidad_vendida'],
            z=cluster_subset['Porcentaje_venta'],
            mode='markers',
            name=nombre,
            marker=dict(size=6, color=COLORS['clusters'][cluster_id % len(COLORS['clusters'])], opacity=0.8),
            text=[f'<b>ID: {id}</b><br>Stock: {s}<br>Vendido: {v}<br>% Venta: {p:.1f}%' 
                  for id, s, v, p in zip(cluster_subset['ID_Producto'], 
                                         cluster_subset['Stock'],
                                         cluster_subset['Cantidad_vendida'],
                                         cluster_subset['Porcentaje_venta'])],
            hoverinfo='text'
        ))
    
    fig.update_layout(
        template='plotly_dark',
        title=dict(text='Segmentación de Productos (3D)', font=dict(size=16)),
        scene=dict(
            xaxis_title='Stock Inicial',
            yaxis_title='Cantidad Vendida',
            zaxis_title='% de Venta'
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=450
    )
    return fig

def crear_grafica_clusters_productos_pie():
    """Gráfica pie de distribución de clusters de productos"""
    cluster_dist = cluster_productos.groupby(['Cluster', 'Cluster_Nombre']).size().reset_index(name='Count')
    
    fig = go.Figure(go.Pie(
        labels=cluster_dist['Cluster_Nombre'],
        values=cluster_dist['Count'],
        hole=0.4,
        marker=dict(colors=COLORS['clusters']),
        textinfo='label+percent',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>Productos: %{value}<br>Porcentaje: %{percent}<extra></extra>'
    ))
    fig.update_layout(
        template='plotly_dark',
        title=dict(text='Distribución de Productos por Segmento', font=dict(size=16)),
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
        showlegend=False
    )
    return fig

def crear_grafica_radar_productos():
    """Gráfica radar de perfil de clusters de productos"""
    cluster_means = cluster_productos.groupby('Cluster')[['Cantidad_vendida', 'Stock', 'Producto_actual_stock', 'Porcentaje_venta']].mean()
    cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 0.001)
    
    categories = ['Ventas', 'Stock Inicial', 'Stock Actual', '% Rotación']
    
    fig = go.Figure()
    
    cluster_nombres = cluster_productos.groupby('Cluster')['Cluster_Nombre'].first().to_dict()
    
    for idx, (cluster, row) in enumerate(cluster_means_norm.iterrows()):
        values = row.values.tolist()
        values += values[:1]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=cluster_nombres.get(cluster, f'Cluster {cluster}'),
            line=dict(color=COLORS['clusters'][idx % len(COLORS['clusters'])])
        ))
    
    fig.update_layout(
        template='plotly_dark',
        title=dict(text='Perfil de Segmentos de Productos', font=dict(size=16)),
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        margin=dict(l=60, r=60, t=50, b=20),
        height=380
    )
    return fig

def crear_grafica_clusters_clientes_3d():
    """Gráfica 3D de clusters de clientes con colores consistentes"""
    fig = go.Figure()
    
    # Ordenar segmentos para consistencia
    segmentos_ordenados = list(SEGMENTO_COLORS.keys())
    
    for segmento in segmentos_ordenados:
        subset = cluster_clientes[cluster_clientes['Segmento'] == segmento]
        if len(subset) == 0:
            continue
        
        color = SEGMENTO_COLORS[segmento]
        
        fig.add_trace(go.Scatter3d(
            x=subset['Categorias'],
            y=subset['Frequency'],
            z=subset['Monetary'],
            mode='markers',
            name=segmento,
            marker=dict(
                size=5,
                color=color,
                opacity=0.7
            ),
            text=[f'<b>Cliente: {id}</b><br>Categorías: {c}<br>Frecuencia: {f}<br>Valor: ${m:,.2f}' 
                  for id, c, f, m in zip(subset['ID_Cliente'], 
                                         subset['Categorias'],
                                         subset['Frequency'],
                                         subset['Monetary'])],
            hoverinfo='text'
        ))
    
    fig.update_layout(
        template='plotly_dark',
        title=dict(text='Segmentación de Clientes (3D) - K=6', font=dict(size=16)),
        scene=dict(
            xaxis_title='Categorías Distintas',
            yaxis_title='Frecuencia de Compra',
            zaxis_title='Valor Monetario ($)'
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=450,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    return fig

def crear_grafica_clusters_clientes_pie():
    """Gráfica pie de distribución de segmentos de clientes con colores consistentes"""
    # Ordenar por el orden definido
    segmentos_ordenados = list(SEGMENTO_COLORS.keys())
    
    segmento_data = []
    colors_pie = []
    
    for segmento in segmentos_ordenados:
        count = len(cluster_clientes[cluster_clientes['Segmento'] == segmento])
        if count > 0:
            segmento_data.append({'Segmento': segmento, 'Count': count})
            colors_pie.append(SEGMENTO_COLORS[segmento])
    
    segmento_df = pd.DataFrame(segmento_data)
    
    fig = go.Figure(go.Pie(
        labels=segmento_df['Segmento'],
        values=segmento_df['Count'],
        hole=0.4,
        marker=dict(colors=colors_pie),
        textinfo='label+percent',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>Clientes: %{value}<br>Porcentaje: %{percent}<extra></extra>'
    ))
    fig.update_layout(
        template='plotly_dark',
        title=dict(text='Distribución de Clientes por Segmento', font=dict(size=16)),
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
        showlegend=False
    )
    return fig

def crear_grafica_radar_clientes():
    """Gráfica radar de perfil de segmentos de clientes con colores consistentes"""
    # Calcular medias por segmento
    segmentos_ordenados = list(SEGMENTO_COLORS.keys())
    
    cluster_means = cluster_clientes.groupby('Segmento')[['Categorias', 'Frequency', 'Monetary', 'Ticket_Promedio']].mean()
    
    # Normalizar todas las variables (mayor es mejor)
    cluster_means_norm = cluster_means.copy()
    for col in cluster_means_norm.columns:
        min_val = cluster_means[col].min()
        max_val = cluster_means[col].max()
        cluster_means_norm[col] = (cluster_means[col] - min_val) / (max_val - min_val + 0.001)
    
    categories = ['Variedad (Categorías)', 'Frecuencia', 'Valor Monetario', 'Ticket Promedio']
    
    fig = go.Figure()
    
    for segmento in segmentos_ordenados:
        if segmento not in cluster_means_norm.index:
            continue
        
        row = cluster_means_norm.loc[segmento]
        values = row.values.tolist()
        values += values[:1]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=segmento,
            line=dict(color=SEGMENTO_COLORS[segmento])
        ))
    
    fig.update_layout(
        template='plotly_dark',
        title=dict(text='Perfil de Segmentos de Clientes', font=dict(size=16)),
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        margin=dict(l=60, r=60, t=50, b=20),
        height=380
    )
    return fig

def crear_grafica_valor_cliente_segmento():
    """Gráfica de valor monetario por segmento de cliente con colores consistentes"""
    segmentos_ordenados = list(SEGMENTO_COLORS.keys())
    
    valor_segmento = cluster_clientes.groupby('Segmento').agg({
        'Monetary': ['sum', 'mean', 'count']
    }).reset_index()
    valor_segmento.columns = ['Segmento', 'Total', 'Promedio', 'Clientes']
    
    # Ordenar por total para la visualización pero mantener colores consistentes
    valor_segmento = valor_segmento.sort_values('Total', ascending=True)
    
    # Obtener colores en el orden correcto
    colors_bar = [SEGMENTO_COLORS[seg] for seg in valor_segmento['Segmento']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=valor_segmento['Segmento'],
        x=valor_segmento['Total'],
        orientation='h',
        marker=dict(color=colors_bar),
        text=[f'${x:,.0f}' for x in valor_segmento['Total']],
        textposition='inside',
        hovertemplate='<b>%{y}</b><br>Total: $%{x:,.2f}<br>Promedio: $%{customdata[0]:,.2f}<br>Clientes: %{customdata[1]}<extra></extra>',
        customdata=valor_segmento[['Promedio', 'Clientes']].values
    ))
    
    fig.update_layout(
        template='plotly_dark',
        title=dict(text='Valor Total por Segmento de Cliente', font=dict(size=16)),
        xaxis_title='Valor Total ($)',
        margin=dict(l=20, r=20, t=50, b=20),
        height=350
    )
    return fig

def crear_grafica_estacionalidad():
    """Gráfica de estacionalidad semanal"""
    ventas_dia = df.groupby('Dia_semana')['Cantidad_dinero'].agg(['sum', 'mean', 'count']).reset_index()
    ventas_dia.columns = ['Dia_semana', 'Total', 'Promedio', 'Transacciones']
    
    # Reordenar días
    dia_map = dict(zip(dias_orden, range(7)))
    ventas_dia['orden'] = ventas_dia['Dia_semana'].map(dia_map)
    ventas_dia = ventas_dia.sort_values('orden')
    ventas_dia['Dia_ES'] = [dias_espanol[i] for i in ventas_dia['orden']]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Ingresos Totales', 'Promedio por Transacción'])
    
    fig.add_trace(go.Bar(
        x=ventas_dia['Dia_ES'],
        y=ventas_dia['Total'],
        marker_color=COLORS['primary'],
        name='Total',
        hovertemplate='%{x}<br>Total: $%{y:,.2f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=ventas_dia['Dia_ES'],
        y=ventas_dia['Promedio'],
        marker_color=COLORS['secondary'],
        name='Promedio',
        hovertemplate='%{x}<br>Promedio: $%{y:,.2f}<extra></extra>'
    ), row=1, col=2)
    
    fig.update_layout(
        template='plotly_dark',
        title=dict(text='Patrón de Ventas Semanal', font=dict(size=16)),
        margin=dict(l=20, r=20, t=70, b=20),
        height=350,
        showlegend=False
    )
    return fig

def crear_grafica_heatmap():
    """Heatmap de ventas por día y mes"""
    heatmap_data = df.groupby(['Dia_semana', 'Mes'])['Cantidad_dinero'].sum().unstack(fill_value=0)
    
    # Reordenar días
    heatmap_data = heatmap_data.reindex(dias_orden).fillna(0)
    
    meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    fig = go.Figure(go.Heatmap(
        z=heatmap_data.values,
        x=[meses_nombres[m-1] for m in heatmap_data.columns],
        y=dias_espanol,
        colorscale='YlOrRd',
        hovertemplate='<b>%{y}, %{x}</b><br>Ingresos: $%{z:,.2f}<extra></extra>'
    ))
    fig.update_layout(
        template='plotly_dark',
        title=dict(text='Mapa de Calor: Ventas por Día y Mes', font=dict(size=16)),
        xaxis_title='Mes',
        yaxis_title='Día de la Semana',
        margin=dict(l=20, r=20, t=50, b=20),
        height=350
    )
    return fig

def crear_grafica_categorias():
    """Gráfica de ventas por categoría"""
    ventas_cat = df.groupby('Categoría').agg({
        'Cantidad_dinero': 'sum',
        'Cantidad': 'sum',
        'ID_Venta': 'count'
    }).reset_index()
    ventas_cat.columns = ['Categoría', 'Ingresos', 'Unidades', 'Transacciones']
    ventas_cat = ventas_cat.sort_values('Ingresos', ascending=False)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ventas_cat['Categoría'].astype(str),
        y=ventas_cat['Ingresos'],
        marker=dict(
            color=ventas_cat['Ingresos'],
            colorscale='Greens',
            showscale=True,
            colorbar=dict(title='Ingresos')
        ),
        text=[f'${x:,.0f}' for x in ventas_cat['Ingresos']],
        textposition='outside',
        hovertemplate='<b>Categoría %{x}</b><br>Ingresos: $%{y:,.2f}<br>Unidades: %{customdata[0]:,}<br>Transacciones: %{customdata[1]:,}<extra></extra>',
        customdata=ventas_cat[['Unidades', 'Transacciones']].values
    ))
    
    fig.update_layout(
        template='plotly_dark',
        title=dict(text='Ingresos por Categoría de Producto', font=dict(size=16)),
        xaxis_title='Categoría',
        yaxis_title='Ingresos ($)',
        margin=dict(l=20, r=20, t=50, b=20),
        height=350
    )
    return fig

def crear_grafica_metodos_pago():
    """Gráfica de métodos de pago con labels descriptivos"""
    # Mapeo de IDs a nombres de métodos de pago
    METODOS_PAGO_LABELS = {
        1: 'Efectivo',
        2: 'Tarjeta de Crédito',
        3: 'Tarjeta de Débito',
        4: 'Mercado Pago',
        5: 'Transferencia'
    }
    
    # Verificar qué columna existe
    if 'Método_Pago' in df.columns:
        metodo_stats = df.groupby('Método_Pago').agg({
            'Cantidad_dinero': 'sum',
            'ID_Venta': 'count'
        }).reset_index()
        metodo_stats.columns = ['Método', 'Ingresos', 'Transacciones']
        # Aplicar mapeo si es numérico
        if metodo_stats['Método'].dtype in ['int64', 'float64']:
            metodo_stats['Método'] = metodo_stats['Método'].map(METODOS_PAGO_LABELS).fillna(metodo_stats['Método'].astype(str))
    elif 'ID_Metodo' in df.columns:
        metodo_stats = df.groupby('ID_Metodo').agg({
            'Cantidad_dinero': 'sum',
            'ID_Venta': 'count'
        }).reset_index()
        metodo_stats.columns = ['Método', 'Ingresos', 'Transacciones']
        # Aplicar mapeo de IDs a nombres
        metodo_stats['Método'] = metodo_stats['Método'].map(METODOS_PAGO_LABELS).fillna(metodo_stats['Método'].astype(str))
    else:
        # Si no hay columna de método, crear datos vacíos
        metodo_stats = pd.DataFrame({'Método': ['Sin datos'], 'Ingresos': [0], 'Transacciones': [0]})
    
    fig = go.Figure(go.Pie(
        labels=metodo_stats['Método'],
        values=metodo_stats['Ingresos'],
        hole=0.3,
        textinfo='label+percent',
        textposition='outside',  # Labels afuera del círculo
        pull=[0.02] * len(metodo_stats),  # Separación ligera entre segmentos
        hovertemplate='<b>%{label}</b><br>Ingresos: $%{value:,.2f}<br>Porcentaje: %{percent}<extra></extra>'
    ))
    fig.update_layout(
        template='plotly_dark',
        title=dict(text='Distribución por Método de Pago', font=dict(size=16)),
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
        showlegend=False  # Ocultar leyenda ya que los labels están afuera
    )
    return fig

def crear_grafica_comparacion_modelos():
    """Gráfica de comparación de modelos predictivos"""
    comparison_df = pd.DataFrame({
        'Modelo': list(modelos.keys()),
        'R²': [modelos[m]['metrics']['R2'] for m in modelos],
        'RMSE': [modelos[m]['metrics']['RMSE'] for m in modelos],
        'MAE': [modelos[m]['metrics']['MAE'] for m in modelos]
    })
    
    fig = make_subplots(
        rows=1, cols=3, 
        subplot_titles=['R² Score', 'RMSE', 'MAE']
    )
    
    colors = [COLORS['primary'], COLORS['accent'], COLORS['secondary'], COLORS['purple']]
    
    fig.add_trace(go.Bar(
        x=comparison_df['Modelo'],
        y=comparison_df['R²'],
        marker_color=colors[:len(comparison_df)],
        text=[f'{x:.3f}' for x in comparison_df['R²']],
        textposition='outside'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=comparison_df['Modelo'],
        y=comparison_df['RMSE'],
        marker_color=colors[:len(comparison_df)],
        text=[f'{x:.0f}' for x in comparison_df['RMSE']],
        textposition='outside'
    ), row=1, col=2)
    
    fig.add_trace(go.Bar(
        x=comparison_df['Modelo'],
        y=comparison_df['MAE'],
        marker_color=colors[:len(comparison_df)],
        text=[f'{x:.0f}' for x in comparison_df['MAE']],
        textposition='outside'
    ), row=1, col=3)
    
    fig.update_layout(
        template='plotly_dark',
        title=dict(text='Comparación de Modelos Predictivos', font=dict(size=16)),
        showlegend=False,
        margin=dict(l=20, r=20, t=70, b=20),
        height=350
    )
    return fig

# ...existing code...

def crear_grafica_predicciones_mejor_modelo():
    """Gráfica de predicciones del mejor modelo"""
    y_pred = modelos[best_model]['predictions']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        marker=dict(color=COLORS['primary'], opacity=0.6, size=8),
        name='Predicciones',
        hovertemplate='<b>Real:</b> $%{x:,.2f}<br><b>Predicho:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    # Línea de referencia perfecta
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color=COLORS['accent'], dash='dash', width=2),
        name='Predicción Perfecta'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        title=dict(text=f'Predicciones vs Valores Reales ({best_model})', font=dict(size=16)),
        xaxis_title='Valor Real ($)',
        yaxis_title='Valor Predicho ($)',
        margin=dict(l=20, r=20, t=50, b=20),
        height=350
    )
    return fig


# Reemplazar la función crear_grafica_importancia_features() con esta:

def crear_grafica_top_predicciones():
    """Top productos mejor y peor predichos - mas util que feature importance"""
    y_pred_test = modelos[best_model]['predictions']
    
    df_pred = pd.DataFrame({
        'ID_Producto': ids_test,
        'Valor_Real': y_test.values,
        'Valor_Predicho': y_pred_test,
        'Error': y_pred_test - y_test.values,
        'Error_Abs': np.abs(y_pred_test - y_test.values),
        'Error_Pct': np.abs((y_pred_test - y_test.values) / (y_test.values + 0.01)) * 100
    })
    
    df_pred = df_pred.merge(productos_info[['ID_Producto', 'Categoria', 'Segmento']], 
                            on='ID_Producto', how='left')
    
    # Top 8 mejor predichos (menor error %)
    top_mejor = df_pred.nsmallest(5, 'Error_Pct')
    # Top 8 peor predichos (mayor error %)
    top_peor = df_pred.nlargest(5, 'Error_Pct')
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Mejor Predichos', 'Peores Predichos'],
        horizontal_spacing=0.15
    )
    
    # Mejor predichos
    fig.add_trace(go.Bar(
        y=[f"Prod {p}" for p in top_mejor['ID_Producto']],
        x=top_mejor['Error_Pct'],
        orientation='h',
        marker=dict(
            color='#2ecc71',
            line=dict(width=1, color='white')
        ),
        text=[f'{e:.1f}%' for e in top_mejor['Error_Pct']],
        textposition='outside',
        hovertemplate=(
            '<b>Producto %{customdata[0]}</b><br>'
            'Categoria: %{customdata[1]}<br>'
            'Cluster: %{customdata[2]}<br>'
            'Real: $%{customdata[3]:,.0f}<br>'
            'Predicho: $%{customdata[4]:,.0f}<br>'
            'Error: %{x:.1f}%'
            '<extra></extra>'
        ),
        customdata=top_mejor[['ID_Producto', 'Categoria', 'Segmento', 'Valor_Real', 'Valor_Predicho']].values,
        showlegend=False
    ), row=1, col=1)
    
    # Peor predichos
    fig.add_trace(go.Bar(
        y=[f"Prod {p}" for p in top_peor['ID_Producto']],
        x=top_peor['Error_Pct'],
        orientation='h',
        marker=dict(
            color='#e74c3c',
            line=dict(width=1, color='white')
        ),
        text=[f'{e:.1f}%' for e in top_peor['Error_Pct']],
        textposition='outside',
        hovertemplate=(
            '<b>Producto %{customdata[0]}</b><br>'
            'Categoria: %{customdata[1]}<br>'
            'Cluster: %{customdata[2]}<br>'
            'Real: $%{customdata[3]:,.0f}<br>'
            'Predicho: $%{customdata[4]:,.0f}<br>'
            'Error: %{x:.1f}%'
            '<extra></extra>'
        ),
        customdata=top_peor[['ID_Producto', 'Categoria', 'Segmento', 'Valor_Real', 'Valor_Predicho']].values,
        showlegend=False
    ), row=1, col=2)
    
    fig.update_xaxes(title_text='Error (%)', row=1, col=1)
    fig.update_xaxes(title_text='Error (%)', row=1, col=2)
    
    fig.update_layout(
        template='plotly_dark',
        title=dict(text='Productos Mejor y Peor Predichos por el Modelo', font=dict(size=14)),
        margin=dict(l=20, r=20, t=60, b=20),
        height=350
    )
    
    return fig
# Agregar después de crear_grafica_importancia_features()

# =============================================================================
# REEMPLAZAR crear_grafica_predicciones_con_productos()
# =============================================================================

def crear_grafica_predicciones_con_productos():
    """Grafica de predicciones del mejor modelo - SOLO CONJUNTO DE PRUEBA, coloreado por cluster"""
    # Usar SOLO predicciones del conjunto de prueba (evitar overfitting)
    y_pred_test = modelos[best_model]['predictions']
    
    # Crear DataFrame con informacion del conjunto de PRUEBA solamente
    df_pred = pd.DataFrame({
        'ID_Producto': ids_test,
        'Valor_Real': y_test.values,
        'Valor_Predicho': y_pred_test,
        'Error': y_test.values - y_pred_test,
        'Error_Abs': np.abs(y_test.values - y_pred_test),
        'Error_Pct': np.abs((y_test.values - y_pred_test) / (y_test.values + 0.01)) * 100
    })
    
    # Agregar info de productos (categoria y segmento/cluster)
    df_pred = df_pred.merge(productos_info[['ID_Producto', 'Categoria', 'Segmento', 'Precio']], 
                            on='ID_Producto', how='left')
    
    fig = go.Figure()
    
    # Colorear por CLUSTER DE PRODUCTO (Estrella, Nicho, etc.)
    for cluster_nombre in CLUSTER_PRODUCTO_COLORS.keys():
        subset = df_pred[df_pred['Segmento'] == cluster_nombre]
        if len(subset) == 0:
            continue
        
        color = CLUSTER_PRODUCTO_COLORS[cluster_nombre]
        
        fig.add_trace(go.Scatter(
            x=subset['Valor_Real'],
            y=subset['Valor_Predicho'],
            mode='markers',
            name=f'{cluster_nombre} ({len(subset)})',
            marker=dict(
                color=color,
                size=10,
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            customdata=subset[['ID_Producto', 'Categoria', 'Segmento', 'Error', 'Error_Pct', 'Precio']].values,
            hovertemplate=(
                '<b>Producto %{customdata[0]}</b><br>'
                'Categoria: %{customdata[1]}<br>'
                'Cluster: %{customdata[2]}<br>'
                'Precio Unit: $%{customdata[5]:,.2f}<br>'
                '<br>'
                'Real: $%{x:,.0f}<br>'
                'Predicho: $%{y:,.0f}<br>'
                'Error: $%{customdata[3]:,.0f} (%{customdata[4]:.1f}%)'
                '<extra></extra>'
            )
        ))
    
    # Linea de prediccion perfecta
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='white', dash='dash', width=2),
        name='Prediccion Perfecta',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        title=dict(
            text=f'Predicciones vs Valores Reales - Conjunto de Prueba ({best_model})',
            font=dict(size=16)
        ),
        xaxis_title='Ingreso Real ($)',
        yaxis_title='Ingreso Predicho ($)',
        margin=dict(l=20, r=20, t=50, b=20),
        height=450,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='closest'
    )
    return fig

# =============================================================================
# REEMPLAZAR crear_grafica_error_por_segmento()
# =============================================================================

def crear_grafica_error_por_segmento():
    """Analisis de precision del modelo por cluster - SOLO CONJUNTO DE PRUEBA"""
    y_pred_test = modelos[best_model]['predictions']
    
    df_error = pd.DataFrame({
        'ID_Producto': ids_test,
        'Valor_Real': y_test.values,
        'Valor_Predicho': y_pred_test,
        'Error_Abs': np.abs(y_test.values - y_pred_test),
        'Error_Pct': np.abs((y_test.values - y_pred_test) / (y_test.values + 0.01)) * 100
    })
    
    df_error = df_error.merge(productos_info[['ID_Producto', 'Categoria', 'Segmento']], 
                              on='ID_Producto', how='left')
    
    # Estadisticas por cluster de producto
    stats_segmento = df_error.groupby('Segmento').agg({
        'Error_Pct': 'mean',
        'Error_Abs': 'mean',
        'ID_Producto': 'count',
        'Valor_Real': 'mean'
    }).reset_index()
    stats_segmento.columns = ['Segmento', 'MAPE', 'MAE', 'N_Productos', 'Ingreso_Prom']
    stats_segmento['Precision'] = 100 - stats_segmento['MAPE']
    stats_segmento = stats_segmento.sort_values('Precision', ascending=True)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Precision por Cluster (%)', 'Error vs Ingreso Promedio'],
        specs=[[{'type': 'bar'}, {'type': 'scatter'}]]
    )
    
    # Colores por cluster de producto
    colors_seg = [CLUSTER_PRODUCTO_COLORS.get(seg, '#888') for seg in stats_segmento['Segmento']]
    
    # Barras de precision
    fig.add_trace(go.Bar(
        y=stats_segmento['Segmento'],
        x=stats_segmento['Precision'],
        orientation='h',
        marker=dict(color=colors_seg),
        text=[f'{p:.1f}%' for p in stats_segmento['Precision']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Precision: %{x:.1f}%<br>Productos: %{customdata}<extra></extra>',
        customdata=stats_segmento['N_Productos'],
        showlegend=False
    ), row=1, col=1)
    
    # Scatter error vs ingreso
    fig.add_trace(go.Scatter(
        x=stats_segmento['Ingreso_Prom'],
        y=stats_segmento['MAPE'],
        mode='markers+text',
        marker=dict(
            size=stats_segmento['N_Productos']*3 + 10,
            color=colors_seg,
            line=dict(width=2, color='white')
        ),
        text=stats_segmento['Segmento'],
        textposition='top center',
        textfont=dict(size=9),
        hovertemplate='<b>%{text}</b><br>Ingreso Prom: $%{x:,.0f}<br>Error: %{y:.1f}%<extra></extra>',
        showlegend=False
    ), row=1, col=2)
    
    fig.update_xaxes(title_text='Precision (%)', row=1, col=1)
    fig.update_xaxes(title_text='Ingreso Promedio ($)', row=1, col=2)
    fig.update_yaxes(title_text='Error Promedio (%)', row=1, col=2)
    
    fig.update_layout(
        template='plotly_dark',
        title=dict(text='Analisis de Precision por Cluster (Conjunto de Prueba)', font=dict(size=16)),
        margin=dict(l=20, r=20, t=70, b=20),
        height=380
    )
    return fig

def crear_tabla_predicciones_detalle():
    """Tabla con detalle de predicciones por producto"""
    y_pred = modelos[best_model]['predictions']
    
    df_tabla = pd.DataFrame({
        'ID_Producto': ids_test,
        'Real': y_test.values,
        'Predicho': y_pred,
        'Error': y_pred - y_test.values,
        'Error_Pct': ((y_pred - y_test.values) / (y_test.values + 0.01)) * 100
    })
    
    df_tabla = df_tabla.merge(productos_info[['ID_Producto', 'Categoria', 'Segmento', 'Precio']], 
                              on='ID_Producto', how='left')
    
    # Ordenar por error absoluto
    df_tabla['Error_Abs'] = np.abs(df_tabla['Error'])
    df_tabla = df_tabla.sort_values('Error_Abs')
    
    return df_tabla

def crear_grafica_top_productos():
    """Top 10 productos por ingresos"""
    top_productos = df.groupby('ID_Producto')['Cantidad_dinero'].sum().nlargest(10).reset_index()
    top_productos = top_productos.sort_values('Cantidad_dinero', ascending=True)
    
    fig = go.Figure(go.Bar(
        y=top_productos['ID_Producto'].astype(str),
        x=top_productos['Cantidad_dinero'],
        orientation='h',
        marker=dict(color=COLORS['primary']),
        text=[f'${x:,.0f}' for x in top_productos['Cantidad_dinero']],
        textposition='inside'
    ))
    fig.update_layout(
        template='plotly_dark',
        title=dict(text='Top 10 Productos por Ingresos', font=dict(size=16)),
        xaxis_title='Ingresos ($)',
        yaxis_title='ID Producto',
        margin=dict(l=20, r=20, t=50, b=20),
        height=350
    )
    return fig


def crear_grafica_top_clientes():
    """Top 10 clientes por valor"""
    top_clientes = df.groupby('ID_Cliente')['Cantidad_dinero'].sum().nlargest(10).reset_index()
    top_clientes = top_clientes.sort_values('Cantidad_dinero', ascending=True)
    
    fig = go.Figure(go.Bar(
        y=top_clientes['ID_Cliente'].astype(str),
        x=top_clientes['Cantidad_dinero'],
        orientation='h',
        marker=dict(color=COLORS['secondary']),
        text=[f'${x:,.0f}' for x in top_clientes['Cantidad_dinero']],
        textposition='inside'
    ))
    fig.update_layout(
        template='plotly_dark',
        title=dict(text='Top 10 Clientes por Valor', font=dict(size=16)),
        xaxis_title='Valor Total ($)',
        yaxis_title='ID Cliente',
        margin=dict(l=20, r=20, t=50, b=20),
        height=350
    )
    return fig


# =============================================================================
# CREAR FIGURAS ESTÁTICAS
# =============================================================================

print("Creando visualizaciones...")
fig_tendencia = crear_grafica_tendencia_global()
fig_regiones = crear_grafica_regiones()
fig_clusters_prod_3d = crear_grafica_clusters_productos_3d()
fig_clusters_prod_pie = crear_grafica_clusters_productos_pie()
fig_radar_productos = crear_grafica_radar_productos()
fig_clusters_cli_3d = crear_grafica_clusters_clientes_3d()
fig_clusters_cli_pie = crear_grafica_clusters_clientes_pie()
fig_radar_clientes = crear_grafica_radar_clientes()
fig_valor_segmento = crear_grafica_valor_cliente_segmento()
fig_estacionalidad = crear_grafica_estacionalidad()
fig_heatmap = crear_grafica_heatmap()
fig_categorias = crear_grafica_categorias()
fig_metodos_pago = crear_grafica_metodos_pago()
fig_comparacion_modelos = crear_grafica_comparacion_modelos()

# ...existing figures...
fig_predicciones = crear_grafica_predicciones_con_productos()  # Reemplaza la anterior
fig_error_segmento = crear_grafica_error_por_segmento()  # Nueva

fig_top_predicciones_modelo = crear_grafica_top_predicciones()
# ...rest of figures...
fig_top_productos = crear_grafica_top_productos()
fig_top_clientes = crear_grafica_top_clientes()

# =============================================================================
# INICIALIZAR APP
# =============================================================================

app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}]
)
app.title = "Dashboard Ejecutivo - Análisis de Ventas"

# ...existing code (todas las pestañas y callbacks)...

# =============================================================================
# LAYOUT - PESTAÑA 1: RESUMEN EJECUTIVO
# =============================================================================

tab_resumen = dbc.Container([
    # Título de sección
    dbc.Row([
        dbc.Col([
            html.H3("Resumen Ejecutivo", className="text-center mb-4 mt-3"),
            html.Hr(className="mb-4")
        ])
    ]),
    
    # KPIs Principales
    dbc.Row([
        dbc.Col([crear_kpi_card("Ingresos Totales", f"${total_ventas:,.2f}", "", "success")], width=2),
        dbc.Col([crear_kpi_card("Unidades Vendidas", f"{total_unidades:,}", "", "info")], width=2),
        dbc.Col([crear_kpi_card("Ticket Promedio", f"${ticket_promedio:.2f}", "", "warning")], width=2),
        dbc.Col([crear_kpi_card("Productos", f"{productos_unicos:,}", "", "primary")], width=2),
        dbc.Col([crear_kpi_card("Clientes", f"{clientes_unicos:,}", "", "danger")], width=2),
        dbc.Col([crear_kpi_card("Regiones", f"{regiones_activas:,}", "", "secondary")], width=2),
    ], className="mb-4 g-3"),
    
    # Gráficas principales
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=fig_tendencia, config={'displayModeBar': False})])
            ], className="shadow-sm border-0")
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=fig_regiones, config={'displayModeBar': False})])
            ], className="shadow-sm border-0")
        ], width=4),
    ], className="mb-4 g-3"),
    
    # Segunda fila
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=fig_categorias, config={'displayModeBar': False})])
            ], className="shadow-sm border-0")
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=fig_metodos_pago, config={'displayModeBar': False})])
            ], className="shadow-sm border-0")
        ], width=6),
    ], className="mb-4 g-3"),
    
    # Top productos y clientes
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=fig_top_productos, config={'displayModeBar': False})])
            ], className="shadow-sm border-0")
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=fig_top_clientes, config={'displayModeBar': False})])
            ], className="shadow-sm border-0")
        ], width=6),
    ], className="mb-4 g-3"),
    
], fluid=True)

# =============================================================================
# LAYOUT - PESTAÑA 2: SEGMENTACIÓN DE PRODUCTOS
# =============================================================================

tab_productos = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3("Segmentación de Productos", className="text-center mb-4 mt-3"),
            html.P("Stock, Cantidad Vendida, Stock Actual y Porcentaje de Rotación", 
                   className="text-center text-muted mb-4"),
            html.Hr()
        ])
    ]),
    
    # Resumen de clusters
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Segmentos Identificados", className="card-title mb-3"),
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Span("●", style={'color': COLORS['clusters'][i], 'fontSize': '1.5rem', 'marginRight': '10px'}),
                                    html.Strong(nombre),
                                    html.Br(),
                                    html.Small(f"{len(cluster_productos[cluster_productos['Cluster'] == i])} productos", className="text-muted")
                                ], className="mb-2")
                            ], width=3) for i, nombre in enumerate(['Bajo Rendimiento', 'Alto Stock', 'Estrella', 'Nicho'])
                        ])
                    ])
                ])
            ], className="shadow-sm border-0 mb-4")
        ])
    ]),
    
    # Gráficas de clustering
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=fig_clusters_prod_3d, config={'displayModeBar': True})])
            ], className="shadow-sm border-0")
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=fig_clusters_prod_pie, config={'displayModeBar': False})])
            ], className="shadow-sm border-0")
        ], width=4),
    ], className="mb-4 g-3"),
    
# En tab_productos, agregar después de la sección "Radar y estadísticas" (después del dbc.Row con fig_radar_productos y la tabla de estadísticas):

# ...existing code...

    # Radar y estadísticas
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=fig_radar_productos, config={'displayModeBar': False})])
            ], className="shadow-sm border-0")
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Estadísticas por Segmento", className="card-title mb-4"),
                    html.Div([
                        dbc.Table([
                            html.Thead([
                                html.Tr([
                                    html.Th("Segmento"),
                                    html.Th("Productos"),
                                    html.Th("Avg. Vendido"),
                                    html.Th("Avg. % Rotación")
                                ])
                            ]),
                            html.Tbody([
                                html.Tr([
                                    html.Td([
                                        html.Span("●", style={'color': COLORS['clusters'][cluster_productos[cluster_productos['Cluster_Nombre'] == nombre]['Cluster'].iloc[0]], 'marginRight': '5px'}),
                                        nombre
                                    ]),
                                    html.Td(f"{len(cluster_productos[cluster_productos['Cluster_Nombre'] == nombre])}"),
                                    html.Td(f"{cluster_productos[cluster_productos['Cluster_Nombre'] == nombre]['Cantidad_vendida'].mean():.0f}"),
                                    html.Td(f"{cluster_productos[cluster_productos['Cluster_Nombre'] == nombre]['Porcentaje_venta'].mean():.1f}%")
                                ]) for nombre in cluster_productos['Cluster_Nombre'].unique()
                            ])
                        ], bordered=True, hover=True, responsive=True, striped=True, className="table-dark")
                    ])
                ])
            ], className="shadow-sm border-0 h-100")
        ], width=6),
    ], className="mb-4 g-3"),
    
    # Descripción de segmentos de productos (NUEVA SECCIÓN)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Descripción de Segmentos de Productos", className="card-title mb-4"),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Span("●", style={'color': CLUSTER_PRODUCTO_COLORS['Estrella'], 'marginRight': '8px'}),
                                html.Strong("Estrella"),
                                html.P("Alta rotación y buenas ventas. Productos más rentables del inventario. Mantener stock óptimo y priorizar en promociones.", className="text-muted small mb-3")
                            ]),
                            html.Div([
                                html.Span("●", style={'color': CLUSTER_PRODUCTO_COLORS['Nicho'], 'marginRight': '8px'}),
                                html.Strong("Nicho"),
                                html.P("Stock limitado pero demanda específica. Productos especializados con mercado definido. Monitorear para evitar quiebres de stock.", className="text-muted small mb-3")
                            ]),
                        ], width=6),
                        dbc.Col([
                            html.Div([
                                html.Span("●", style={'color': CLUSTER_PRODUCTO_COLORS['Alto Stock'], 'marginRight': '8px'}),
                                html.Strong("Alto Stock"),
                                html.P("Mucho inventario con rotación moderada. Capital inmovilizado. Aplicar descuentos o promociones para acelerar ventas.", className="text-muted small mb-3")
                            ]),
                            html.Div([
                                html.Span("●", style={'color': CLUSTER_PRODUCTO_COLORS['Bajo Rendimiento'], 'marginRight': '8px'}),
                                html.Strong("Bajo Rendimiento"),
                                html.P("Baja rotación y pocas ventas. Evaluar discontinuar o liquidar. Liberar espacio para productos más rentables.", className="text-muted small mb-3")
                            ]),
                        ], width=6),
                    ])
                ])
            ], className="shadow-sm border-0")
        ])
    ], className="mb-4"),
    
], fluid=True)

# ...existing code...

# =============================================================================
# LAYOUT - PESTAÑA 3: SEGMENTACIÓN DE CLIENTES
# =============================================================================

tab_clientes = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3("Segmentación de Clientes", className="text-center mb-4 mt-3"),
            html.P("Frecuencia de compras, total comprado, gasto total, caterogias distintas, promedio de gasto de compra, región ", 
                   className="text-center text-muted mb-4"),
            html.Hr()
        ])
    ]),
    
    # KPIs de segmentos con colores consistentes
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Resumen de Segmentos", className="card-title mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Span("●", style={'color': SEGMENTO_COLORS.get(segmento, '#888'), 'fontSize': '1.5rem', 'marginRight': '10px'}),
                                html.Strong(segmento, style={'fontSize': '0.85rem'}),
                                html.Br(),
                                html.Small(f"{len(cluster_clientes[cluster_clientes['Segmento'] == segmento])} clientes", className="text-muted"),
                                html.Br(),
                                html.Small(f"${cluster_clientes[cluster_clientes['Segmento'] == segmento]['Monetary'].sum():,.0f}", className="text-success")
                            ], className="mb-2")
                        ], width=2) for segmento in SEGMENTO_COLORS.keys() if len(cluster_clientes[cluster_clientes['Segmento'] == segmento]) > 0
                    ])
                ])
            ], className="shadow-sm border-0 mb-4")
        ])
    ]),
    
    # Gráficas 3D y Pie
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=fig_clusters_cli_3d, config={'displayModeBar': True})])
            ], className="shadow-sm border-0")
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=fig_clusters_cli_pie, config={'displayModeBar': False})])
            ], className="shadow-sm border-0")
        ], width=4),
    ], className="mb-4 g-3"),
    
    # Radar y valor por segmento
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=fig_radar_clientes, config={'displayModeBar': False})])
            ], className="shadow-sm border-0")
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=fig_valor_segmento, config={'displayModeBar': False})])
            ], className="shadow-sm border-0")
        ], width=6),
    ], className="mb-4 g-3"),
    
    # Tabla de estadísticas de clientes con colores consistentes
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Estadísticas Detalladas por Segmento", className="card-title mb-4"),
                    dbc.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Segmento"),
                                html.Th("Clientes"),
                                html.Th("Categorías Prom."),
                                html.Th("Frecuencia Prom."),
                                html.Th("Ticket Prom."),
                                html.Th("Valor Total")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td([
                                    html.Span("●", style={'color': SEGMENTO_COLORS.get(segmento, '#888'), 'marginRight': '8px'}),
                                    html.Strong(segmento)
                                ]),
                                html.Td(f"{len(cluster_clientes[cluster_clientes['Segmento'] == segmento]):,}"),
                                html.Td(f"{cluster_clientes[cluster_clientes['Segmento'] == segmento]['Categorias'].mean():.1f}"),
                                html.Td(f"{cluster_clientes[cluster_clientes['Segmento'] == segmento]['Frequency'].mean():.1f}"),
                                html.Td(f"${cluster_clientes[cluster_clientes['Segmento'] == segmento]['Ticket_Promedio'].mean():,.0f}"),
                                html.Td(f"${cluster_clientes[cluster_clientes['Segmento'] == segmento]['Monetary'].sum():,.0f}", className="text-success fw-bold")
                            ]) for segmento in SEGMENTO_COLORS.keys() if len(cluster_clientes[cluster_clientes['Segmento'] == segmento]) > 0
                        ])
                    ], bordered=True, hover=True, responsive=True, striped=True, className="table-dark")
                ])
            ], className="shadow-sm border-0")
        ])
    ], className="mb-4"),
    
    # Descripción de segmentos
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Descripción de Segmentos", className="card-title mb-4"),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Span("●", style={'color': SEGMENTO_COLORS['Premium y Frecuentes'], 'marginRight': '8px'}),
                                html.Strong("Premium y Frecuentes"),
                                html.P("Clientes de alto valor, compran seguido y variado. Programas de lealtad premium.", className="text-muted small mb-2")
                            ]),
                            html.Div([
                                html.Span("●", style={'color': SEGMENTO_COLORS['Exploradores de Nicho'], 'marginRight': '8px'}),
                                html.Strong("Exploradores de Nicho"),
                                html.P("Compran menos pero enfocados en categorías particulares. Ofertas específicas.", className="text-muted small mb-2")
                            ]),
                            html.Div([
                                html.Span("●", style={'color': SEGMENTO_COLORS['Clientes Estables'], 'marginRight': '8px'}),
                                html.Strong("Clientes Estables"),
                                html.P("Regulares en gasto y frecuencia. Segmento medio. Recompensas por constancia.", className="text-muted small mb-2")
                            ]),
                        ], width=6),
                        dbc.Col([
                            html.Div([
                                html.Span("●", style={'color': SEGMENTO_COLORS['Ocasionales Económicos'], 'marginRight': '8px'}),
                                html.Strong("Ocasionales Económicos"),
                                html.P("Compran con poca frecuencia pero variado. Promociones estacionales.", className="text-muted small mb-2")
                            ]),
                            html.Div([
                                html.Span("●", style={'color': SEGMENTO_COLORS['Cazadores de Oferta'], 'marginRight': '8px'}),
                                html.Strong("Cazadores de Oferta"),
                                html.P("Compra orientada a bajo costo. Cupones y flash sales.", className="text-muted small mb-2")
                            ]),
                            html.Div([
                                html.Span("●", style={'color': SEGMENTO_COLORS['Nuevos o Dormidos'], 'marginRight': '8px'}),
                                html.Strong("Nuevos o Dormidos"),
                                html.P("Ticket alto pero poca variedad. Emails de bienvenida o reactivación.", className="text-muted small mb-2")
                            ]),
                        ], width=6),
                    ])
                ])
            ], className="shadow-sm border-0")
        ])
    ], className="mb-4"),
    
], fluid=True)

# =============================================================================
# LAYOUT - PESTAÑA 4: MODELOS PREDICTIVOS
# =============================================================================

# =============================================================================
# LAYOUT - PESTAÑA 4: MODELOS PREDICTIVOS (ACTUALIZADA)
# =============================================================================

# =============================================================================
# LAYOUT - PESTANA 4: MODELOS PREDICTIVOS (SIN EMOJIS)
# =============================================================================

tab_modelos = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3("Modelos Predictivos de Ingresos por Producto", className="text-center mb-4 mt-3"),
            #html.P("Predice el ingreso total esperado de un producto basandose en sus caracteristicas", 
                   #className="text-center text-muted mb-4"),
            html.Hr()
        ])
    ]),
    
    # KPIs del mejor modelo
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Mejor Modelo", className="card-title text-muted"),
                    html.H3(best_model, className="text-success mb-0")
                ])
            ], className="shadow-sm border-0 h-100")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("R2 Score", className="card-title text-muted"),
                    html.H3(f"{modelos[best_model]['metrics']['R2']:.4f}", className="text-info mb-0"),
                    html.Small(f"Explica {modelos[best_model]['metrics']['R2']*100:.1f}% de la varianza", className="text-muted")
                ])
            ], className="shadow-sm border-0 h-100")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("RMSE", className="card-title text-muted"),
                    html.H3(f"{modelos[best_model]['metrics']['RMSE']:,.0f}", className="text-warning mb-0"),
                    html.Small("Error cuadratico medio", className="text-muted")
                ])
            ], className="shadow-sm border-0 h-100")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("MAE", className="card-title text-muted"),
                    html.H3(f"{modelos[best_model]['metrics']['MAE']:,.0f}", className="text-danger mb-0"),
                    html.Small("Error absoluto medio", className="text-muted")
                ])
            ], className="shadow-sm border-0 h-100")
        ], width=3),
    ], className="mb-4 g-3"),
    
    # Comparacion de modelos
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=fig_comparacion_modelos, config={'displayModeBar': False})])
            ], className="shadow-sm border-0")
        ])
    ], className="mb-4"),
    
    # Grafica de predicciones con productos identificados
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Predicciones por Producto (Conjunto de Prueba)", className="card-title mb-2"),
                    #html.P("Coloreado por tipo de cluster. Pasa el cursor para ver detalles.", className="text-muted small"),
                    dcc.Graph(figure=fig_predicciones, config={'displayModeBar': False})
                ])
            ], className="shadow-sm border-0")
        ])
    ], className="mb-4"),
    
    # Analisis de error por segmento
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Precision del Modelo por Cluster", className="card-title mb-2"),
                    #html.P("Evalua en que tipos de productos el modelo predice mejor", className="text-muted small"),
                    dcc.Graph(figure=fig_error_segmento, config={'displayModeBar': False})
                ])
            ], className="shadow-sm border-0")
        ], width=7),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Productos Mejor y Peor Predichos", className="card-title mb-2"),
                    html.P("Identifica donde el modelo acierta y donde falla", className="text-muted small"),
                    dcc.Graph(figure=fig_top_predicciones_modelo, config={'displayModeBar': False})
                ])
            ], className="shadow-sm border-0")
        ], width=5),
    ], className="mb-4 g-3"),
    # Simulador de prediccion
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Consultar Prediccion de un Producto", className="card-title mb-3"),
                    #html.P("Selecciona un producto del conjunto de prueba para ver su prediccion", className="text-muted small mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Seleccionar Producto:", className="fw-bold"),
                            dcc.Dropdown(
                                id='dropdown-producto-prediccion',
                                options=[
                                    {'label': f'Producto {p} - Cat. {c} ({s})', 'value': p}
                                    for p, c, s in zip(
                                        productos_info[productos_info['ID_Producto'].isin(ids_test)]['ID_Producto'],
                                        productos_info[productos_info['ID_Producto'].isin(ids_test)]['Categoria'],
                                        productos_info[productos_info['ID_Producto'].isin(ids_test)]['Segmento']
                                    )
                                ],
                                placeholder="Selecciona un producto...",
                                style={'color': '#000'}
                            )
                        ], width=6),
                        dbc.Col([
                            html.Div(id='resultado-prediccion-producto', className="mt-2")
                        ], width=6),
                    ])
                ])
            ], className="shadow-sm border-0")
        ])
    ], className="mb-4"),
    
    # Explicacion del modelo
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Que predice este modelo?", className="card-title mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H6("Variable a Predecir (Y)", className="text-success"),
                                html.P("Ingresos totales por producto", className="mb-1"),
                                #html.Code("Cantidad_dinero = Suma(Precio x Cantidad)", className="small")
                            ], className="p-2 border-start border-success border-3")
                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.H6("Variables de Entrada (X)", className="text-info"),
                                html.Ul([
                                    html.Li("Precio Unitario", className="small"),
                                    html.Li("Stock inicial y actual", className="small"),
                                    html.Li("Porcentaje de Rotacion", className="small"),
                                    html.Li("Region y Categoria", className="small"),
                                ], className="mb-0 ps-3")
                            ], className="p-2 border-start border-info border-3")
                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.H6("Casos de Uso", className="text-warning"),
                                html.Ul([
                                    html.Li("Estimar ingresos de nuevos productos", className="small"),
                                    html.Li("Optimizar inventario", className="small"),
                                    html.Li("Planificar promociones", className="small"),
                                ], className="mb-0 ps-3")
                            ], className="p-2 border-start border-warning border-3")
                        ], width=4),
                    ])
                ])
            ], className="shadow-sm border-0 bg-dark")
        ])
    ], className="mb-4"),
    
    # Tabla comparativa de modelos
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Comparativa Detallada de Modelos", className="card-title mb-4"),
                    dbc.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Modelo"),
                                html.Th("R2 Score"),
                                html.Th("RMSE"),
                                html.Th("MAE"),
                                html.Th("Estado")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td(html.Strong(modelo) if modelo == best_model else modelo),
                                html.Td(f"{modelos[modelo]['metrics']['R2']:.4f}"),
                                html.Td(f"{modelos[modelo]['metrics']['RMSE']:,.0f}"),
                                html.Td(f"{modelos[modelo]['metrics']['MAE']:,.0f}"),
                                html.Td(
                                    dbc.Badge("Mejor", color="success") if modelo == best_model 
                                    else dbc.Badge("-", color="secondary")
                                )
                            ], className="table-success" if modelo == best_model else "") 
                            for modelo in modelos.keys()
                        ])
                    ], bordered=True, hover=True, responsive=True, striped=True, className="table-dark")
                ])
            ], className="shadow-sm border-0")
        ])
    ], className="mb-4"),
    
], fluid=True)
# =============================================================================
# LAYOUT - PESTAÑA 5: ANÁLISIS TEMPORAL
# =============================================================================

# ...existing code...

# =============================================================================
# LAYOUT - PESTAÑA 5: ANÁLISIS TEMPORAL
# =============================================================================

tab_temporal = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3("Análisis Temporal y Estacionalidad", className="text-center mb-4 mt-3"),
            #html.P("Patrones de ventas y pronósticos con Prophet (Facebook/Meta)", 
            #       className="text-center text-muted mb-4"),
            html.Hr()
        ])
    ]),
    
    # Estacionalidad y heatmap
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=fig_estacionalidad, config={'displayModeBar': False})])
            ], className="shadow-sm border-0")
        ], width=7),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=fig_heatmap, config={'displayModeBar': False})])
            ], className="shadow-sm border-0")
        ], width=5),
    ], className="mb-4 g-3"),
    
    # Serie temporal con forecast dinámico
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Pronóstico de Ventas con Prophet", className="card-title mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Días a pronosticar:", className="fw-bold"),
                            dcc.Slider(
                                id='slider-forecast',
                                min=7,
                                max=90,
                                step=7,
                                value=30,
                                marks={i: f'{i}d' for i in [7, 14, 30, 60, 90]},
                                className="mb-2"
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Región:", className="fw-bold"),
                            dcc.Dropdown(
                                id='dropdown-region-forecast',
                                options=[{'label': 'Todas las regiones', 'value': 'Todas'}] + 
                                        [{'label': r, 'value': r} for r in df['Región'].unique()],
                                value='Todas',
                                clearable=False,
                                style={'color': '#000'}
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Modelo:", className="fw-bold"),
                            dcc.Dropdown(
                                id='dropdown-modelo-forecast',
                                options=[
                                    {'label': 'Prophet (Aditivo)', 'value': 'prophet_default'},
                                    {'label': 'Prophet (Multiplicativo)', 'value': 'prophet_mult'},
                                    {'label': 'Prophet (Con Regresores)', 'value': 'prophet_reg'},
                                    {'label': 'ARIMA (Estadistico)', 'value': 'arima'},
                                ],
                                value='prophet_default',
                                clearable=False,
                                style={'color': '#000'}
                            )
                        ], width=4),
                    ], className="mb-3"),
                    dcc.Loading(
                        id="loading-forecast",
                        type="circle",
                        children=[dcc.Graph(id='grafico-forecast', config={'displayModeBar': False})]
                    ),
                    html.Div(id='info-modelo-forecast', className="mt-2")
                ])
            ], className="shadow-sm border-0")
        ])
    ], className="mb-4"),
    
], fluid=True)


# =============================================================================
# LAYOUT - PESTAÑA 6: CONCLUSIONES
# =============================================================================
# Calcular valor segmento Premium (con manejo de error si no existe)

# Calcular valor segmento Premium (con manejo de error si no existe)
valor_premium = 0
if 'Premium y Frecuentes' in cluster_clientes['Segmento'].values:
    valor_premium = cluster_clientes[cluster_clientes['Segmento'] == 'Premium y Frecuentes']['Monetary'].sum()

tab_conclusiones = dbc.Container([
    # Título principal
    dbc.Row([
        dbc.Col([
            html.H3("Conclusiones y Recomendaciones", className="text-center mb-4 mt-3"),
            html.Hr()
        ])
    ]),

    # Fila: Hallazgos a la izquierda, Oportunidades a la derecha
    dbc.Row([
        # Columna 1: Hallazgos
        dbc.Col([
            dbc.CardBody([
                html.H4("Hallazgos Principales", className="card-title text-danger mb-4"),

                dbc.Alert([
                    html.H5("Segmentación de Productos", className="alert-heading"),
                    html.P("Se identificaron 4 segmentos de productos. Los productos 'Estrella' representan las mejores oportunidades de negocio con alta rotación."),
                ], color="info", className="mb-3"),

                dbc.Alert([
                    html.H5("Segmentación de Clientes", className="alert-heading"),
                    html.P(f"El análisis reveló 6 segmentos de clientes. Los clientes 'Premium y Frecuentes' generan ${valor_premium:,.0f} en valor total."),
                ], color="success", className="mb-3"),

                dbc.Alert([
                    html.H5("Modelo Predictivo", className="alert-heading"),
                    html.P("El modelo Gradient Boosting logró un R² de 0.8476, permitiendo predecir ingresos con alta precisión."),
                ], color="warning", className="mb-3"),

                dbc.Alert([
                    html.H5("Ingresos Totales", className="alert-heading"),
                    html.P("El negocio generó $103,947.36 en ingresos totales, con 3,000 transacciones de 326 clientes únicos."),
                ], color="primary", className="mb-3"),
            ])
        ], width=6),

        # Columna 2: Áreas de Oportunidad
        dbc.Col([
            dbc.CardBody([
                html.H4("Áreas de Oportunidad", className="card-title text-success mb-4"),

                dbc.Alert([
                    html.H5("Encuestas a Clientes", className="alert-heading"),
                    html.P("Detectar razones de compra, satisfacción y promociones preferidas."),
                ], color="light", className="mb-3"),

                dbc.Alert([
                    html.H5("Integrar Variables Externas", className="alert-heading"),
                    html.P("Clima, días festivos, eventos o redes sociales podrían impactar ventas."),
                ], color="light", className="mb-3"),

                dbc.Alert([
                    html.H5("Canales de Feedback", className="alert-heading"),
                    html.P("Usar reseñas, atención al cliente y menciones en redes para conocer percepciones."),
                ], color="light", className="mb-3"),

                dbc.Alert([
                    html.H5("Datos de Recompra", className="alert-heading"),
                    html.P("Analizar tiempo entre compras y productos relacionados."),
                ], color="light", className="mb-3"),
            ])
        ], width=6),
    ], className="mb-4"),

    # Tabla: Estrategias por Cliente
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(" Estrategias por Segmento de Clientes", className="card-title mb-3"),
                    dbc.Table([
                        html.Thead([
                            html.Tr([html.Th("Segmento"), html.Th("Comportamiento"), html.Th("Estrategia")])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td("Cluster 0 – Premium y Frecuentes"),
                                html.Td("Compran mucho, seguido y en muchas categorías."),
                                html.Td("Programa de fidelización exclusivo y beneficios premium.")
                            ]),
                            html.Tr([
                                html.Td("Cluster 1 – Exploradores de Nicho"),
                                html.Td("Compran en categorías específicas."),
                                html.Td("Promociones focalizadas para explorar nuevas categorías.")
                            ]),
                            html.Tr([
                                html.Td("Cluster 2 – Clientes Estables"),
                                html.Td("Consistentes en gasto y frecuencia."),
                                html.Td("Recompensas por constancia y muestras para ampliar interés.")
                            ]),
                            html.Tr([
                                html.Td("Cluster 3 – Ocasionales Económicos"),
                                html.Td("Compran poco pero variado."),
                                html.Td("Campañas de recurrencia y packs por categoría.")
                            ]),
                            html.Tr([
                                html.Td("Cluster 4 – Cazadores de Oferta"),
                                html.Td("Muy sensibles al precio."),
                                html.Td("Cupones, descuentos flash y combos tipo “lleva más por menos”.")
                            ]),
                            html.Tr([
                                html.Td("Cluster 5 – Nuevos o Dormidos"),
                                html.Td("Compran poco o están inactivos."),
                                html.Td("Campañas de reactivación o bienvenida.")
                            ]),
                        ])
                    ], bordered=True, striped=True, hover=True, className="table-dark")
                ])
            ])
        ])
    ], className="mb-4"),

    # Tabla: Estrategias por Producto
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Estrategias por Segmento de Productos", className="card-title mb-3"),
                    dbc.Table([
                        html.Thead([
                            html.Tr([html.Th("Segmento"), html.Th("Comportamiento"), html.Th("Estrategia")])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td("Cluster 0 – Alto Stock y Baja Rotación"),
                                html.Td("Mucho inventario pero baja venta."),
                                html.Td("Aplicar promociones o liquidación para acelerar rotación.")
                            ]),
                            html.Tr([
                                html.Td("Cluster 1 – Bajo Stock y Venta Moderada"),
                                html.Td("Ventas aceptables pero inventario limitado."),
                                html.Td("Aumentar inventario ligeramente para evitar quiebres.")
                            ]),
                            html.Tr([
                                html.Td("Cluster 2 – Stock y Ventas Balanceadas"),
                                html.Td("Ventas y stock equilibrados."),
                                html.Td("Mantener niveles y promover cross-selling.")
                            ]),
                            html.Tr([
                                html.Td("Cluster 3 – Alta Venta y Bajo Stock"),
                                html.Td("Muy buena rotación, poco inventario."),
                                html.Td("Reabastecimiento inmediato e incentivar promoción.")
                            ]),
                        ])
                    ], bordered=True, striped=True, hover=True, className="table-dark")
                ])
            ])
        ])
    ], className="mb-4"),

    # Recomendaciones Generales
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Recomendaciones Generales", className="card-title text-success mb-4"),
                    html.Div([
                        html.Div([
                            html.H6("1. Optimización de Inventario", className="text-info"),
                            html.P("Reducir stock de productos de 'Bajo Rendimiento' y aumentar productos 'Estrella'.", className="text-muted small"),
                        ], className="mb-3 p-2 border-start border-info border-3"),

                        html.Div([
                            html.H6("2. Programa de Fidelización", className="text-success"),
                            html.P("Crear programa VIP para retener clientes 'Premium y Frecuentes' y recuperar 'Nuevos o Dormidos'.", className="text-muted small"),
                        ], className="mb-3 p-2 border-start border-success border-3"),

                        html.Div([
                            html.H6("3. Campañas Segmentadas", className="text-warning"),
                            html.P("Diseñar promociones específicas para cada segmento de cliente según su comportamiento.", className="text-muted small"),
                        ], className="mb-3 p-2 border-start border-warning border-3"),

                        html.Div([
                            html.H6("4. Expansión Regional", className="text-primary"),
                            html.P("Enfocar esfuerzos en las regiones de mayor potencial identificadas en el análisis.", className="text-muted small"),
                        ], className="mb-3 p-2 border-start border-primary border-3"),

                        html.Div([
                            html.H6("5. Predicción de Demanda", className="text-danger"),
                            html.P("Implementar el modelo predictivo para anticipar demanda y optimizar operaciones.", className="text-muted small"),
                        ], className="mb-3 p-2 border-start border-danger border-3"),
                    ])
                ])
            ], className="shadow-sm border-0")
        ])
    ], className="mb-4"),

    # Conclusión Final
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Conclusión Final", className="card-title mb-3"),
                    html.Div([
                        html.P("El análisis integral de patrones de ventas, apoyado en técnicas de EDA, clustering y modelos predictivos, permitió identificar comportamientos clave de productos y clientes, optimizar la gestión de inventario y mejorar la toma de decisiones comerciales. Los hallazgos obtenidos sirven como base para implementar estrategias de surtido, promociones y segmentación más efectivas. Se recomienda continuar con el monitoreo periódico del comportamiento del mercado, así como enriquecer el análisis con fuentes externas y cualitativas, como encuestas o datos de contexto, que permitan capturar aspectos no reflejados en las ventas históricas.",
                               className="text-light small")
                    ])
                ])
            ], className="shadow-sm border-0")
        ])
    ], className="mb-4"),

    # Métricas Finales
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Resumen de Métricas del Análisis", className="card-title mb-4"),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H2(f"{productos_unicos}", className="text-primary mb-0"),
                                html.P("Productos Analizados", className="text-muted")
                            ], className="text-center")
                        ], width=2),
                        dbc.Col([
                            html.Div([
                                html.H2(f"{clientes_unicos}", className="text-success mb-0"),
                                html.P("Clientes Segmentados", className="text-muted")
                            ], className="text-center")
                        ], width=2),
                        dbc.Col([
                            html.Div([
                                html.H2("4", className="text-info mb-0"),
                                html.P("Modelos Evaluados", className="text-muted")
                            ], className="text-center")
                        ], width=2),
                        dbc.Col([
                            html.Div([
                                html.H2(f"{regiones_activas}", className="text-warning mb-0"),
                                html.P("Regiones Analizadas", className="text-muted")
                            ], className="text-center")
                        ], width=2),
                        dbc.Col([
                            html.Div([
                                html.H2("10", className="text-danger mb-0"),
                                html.P("Segmentos Creados", className="text-muted")
                            ], className="text-center")
                        ], width=2),
                        dbc.Col([
                            html.Div([
                                html.H2(f"{len(df):,}", className="text-secondary mb-0"),
                                html.P("Registros Procesados", className="text-muted")
                            ], className="text-center")
                        ], width=2),
                    ])
                ])
            ], className="shadow-sm border-0")
        ])
    ], className="mb-4"),

], fluid=True)


# =============================================================================
# LAYOUT PRINCIPAL
# =============================================================================

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("Dashboard Ejecutivo", className="mb-0"),
                html.P("Análisis de Patrones de Ventas", className="text-muted mb-0")
            ], className="text-center py-4")
        ])
    ], className="bg-dark mb-3"),
    
    # Tabs
    dbc.Tabs([
        dbc.Tab(tab_resumen, label="Resumen", tab_id="tab-resumen"),
        dbc.Tab(tab_productos, label="Productos", tab_id="tab-productos"),
        dbc.Tab(tab_clientes, label="Clientes", tab_id="tab-clientes"),
        dbc.Tab(tab_modelos, label="Modelos", tab_id="tab-modelos"),
        dbc.Tab(tab_temporal, label="Temporal", tab_id="tab-temporal"),
        dbc.Tab(tab_conclusiones, label="Conclusiones", tab_id="tab-conclusiones"),
    ], id="tabs", active_tab="tab-resumen", className="mb-3"),
    
    
], fluid=True, className="px-4")

# =============================================================================
# CALLBACKS
# =============================================================================

# En la sección CALLBACKS, agregar:

# =============================================================================
# CALLBACK ACTUALIZADO (SIN EMOJIS, SOLO CONJUNTO DE PRUEBA)
# =============================================================================

@callback(
    Output('resultado-prediccion-producto', 'children'),
    Input('dropdown-producto-prediccion', 'value')
)
def predecir_producto(id_producto):
    """Muestra la prediccion para un producto del conjunto de prueba"""
    if id_producto is None:
        return html.Div([
            html.P("Selecciona un producto para ver su prediccion", className="text-muted")
        ])
    
    # Buscar el producto en el conjunto de PRUEBA
    mask_test = ids_test == id_producto
    
    if not mask_test.any():
        return dbc.Alert("Producto no encontrado en conjunto de prueba", color="warning")
    
    # Obtener indice del producto
    idx = np.where(mask_test)[0][0]
    valor_real = y_test.values[idx]
    valor_predicho = modelos[best_model]['predictions'][idx]
    error = valor_predicho - valor_real
    error_pct = (error / valor_real) * 100 if valor_real != 0 else 0
    
    # Info del producto
    info = productos_info[productos_info['ID_Producto'] == id_producto].iloc[0]
    cluster_color = CLUSTER_PRODUCTO_COLORS.get(info['Segmento'], '#888')
    
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5(f"Producto {id_producto}", className="text-primary mb-0"),
                ], width=8),
                dbc.Col([
                    dbc.Badge(info['Segmento'], style={'backgroundColor': cluster_color}, className="float-end")
                ], width=4),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Small("Categoria", className="text-muted d-block"),
                        html.Strong(str(info['Categoria']))
                    ])
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.Small("Cluster", className="text-muted d-block"),
                        html.Strong(info['Segmento'])
                    ])
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.Small("Precio Unit.", className="text-muted d-block"),
                        html.Strong(f"${info['Precio']:,.2f}")
                    ])
                ], width=4),
            ], className="mb-3"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4(f"${valor_real:,.0f}", className="text-success mb-0"),
                        html.Small("Ingreso Real", className="text-muted")
                    ], className="text-center")
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.H4(f"${valor_predicho:,.0f}", className="text-info mb-0"),
                        html.Small("Prediccion", className="text-muted")
                    ], className="text-center")
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.H4(
                            f"{'+' if error > 0 else ''}{error_pct:.1f}%", 
                            className=f"{'text-danger' if abs(error_pct) > 20 else 'text-success'} mb-0"
                        ),
                        html.Small("Error", className="text-muted")
                    ], className="text-center")
                ], width=4),
            ]),
            html.Div([
                html.Small(
                    "Datos del conjunto de prueba ", 
                    className="text-muted d-block mt-3 fst-italic"
                )
            ])
        ])
    ], className="border-0 bg-dark")
# =============================================================================
# CALLBACKS
# =============================================================================

@callback(
    [Output('grafico-forecast', 'figure'),
     Output('info-modelo-forecast', 'children')],
    [Input('slider-forecast', 'value'),
     Input('dropdown-region-forecast', 'value'),
     Input('dropdown-modelo-forecast', 'value')]
)
def actualizar_forecast(forecast_days, region, modelo_seleccionado):
    """Actualiza el gráfico de forecast según los parámetros"""
    df_filtrado = df.copy()
    if region != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['Región'] == region]
    
    resultado, error = entrenar_mejor_forecast(df_filtrado, forecast_days, modelo_seleccionado)
    
    if error:
        fig = go.Figure()
        fig.add_annotation(
            text=f"⚠️ {error}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="orange")
        )
        fig.update_layout(template='plotly_dark', height=400)
        return fig, dbc.Alert(f"Error: {error}", color="danger", className="mt-2")
    
    fig = go.Figure()
    
    # Histórico
    hist_col = resultado.get('historico_col', 'Cantidad_dinero')
    if hist_col == 'y':
        hist_values = resultado['historico']['y'] if 'y' in resultado['historico'].columns else resultado['historico'].iloc[:, 0]
    else:
        hist_values = resultado['historico'][hist_col]
    
    fig.add_trace(go.Scatter(
        x=resultado['historico'].index,
        y=hist_values,
        mode='lines',
        name='Histórico',
        line=dict(color=COLORS['primary'], width=1.5),
        opacity=0.8
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=resultado['forecast_index'],
        y=resultado['forecast'],
        mode='lines',
        name='Pronóstico',
        line=dict(color=COLORS['accent'], width=2.5)
    ))
    
    # Intervalo de confianza
    fig.add_trace(go.Scatter(
        x=list(resultado['forecast_index']) + list(resultado['forecast_index'][::-1]),
        y=list(resultado['conf_int_upper']) + list(resultado['conf_int_lower'][::-1]),
        fill='toself',
        fillcolor='rgba(231, 76, 60, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Intervalo 95%',
        hoverinfo='skip'
    ))
    
    model_name = resultado.get('model_name', 'Modelo')
    fig.update_layout(
        template='plotly_dark',
        title=dict(text=f'Pronóstico {model_name} - {forecast_days} días', font=dict(size=16)),
        xaxis_title='Fecha',
        yaxis_title='Ingresos ($)',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='x unified'
    )
    
    # Info del modelo
    forecast_mean = np.mean(resultado['forecast'])
    forecast_total = np.sum(resultado['forecast'])
    
    info_items = [dbc.Badge(f"📊 {model_name}", color="primary", className="me-2")]
    
    if 'order' in resultado:
        info_items.append(dbc.Badge(f"Order: {resultado['order']}", color="info", className="me-2"))
    if 'aic' in resultado:
        info_items.append(dbc.Badge(f"AIC: {resultado['aic']:.1f}", color="secondary", className="me-2"))
    
    info_content = dbc.Card([
        dbc.CardBody([
            html.Div(info_items, className="mb-2"),
            dbc.Row([
                dbc.Col([
                    html.Small("Promedio diario:", className="text-muted"),
                    html.Span(f" ${forecast_mean:,.2f}", className="fw-bold text-success")
                ], width=6),
                dbc.Col([
                    html.Small(f"Total ({forecast_days}d):", className="text-muted"),
                    html.Span(f" ${forecast_total:,.2f}", className="fw-bold text-warning")
                ], width=6),
            ])
        ])
    ], className="mt-2 border-0 bg-dark")
    
    return fig, info_content


# =============================================================================
# EJECUTAR APP
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Dashboard de Análisis Multivariado de Ventas")
    print("=" * 60)
    print(f"📊 Datos cargados: {len(df):,} registros")
    print(f"📦 Productos: {productos_unicos:,}")
    print(f"👥 Clientes: {clientes_unicos:,}")
    print(f"🌍 Regiones: {regiones_activas}")
    print(f"🎯 Clusters Productos: {cluster_productos['Cluster'].nunique()} grupos")
    print(f"🎯 Segmentos Clientes: {cluster_clientes['Segmento'].nunique()} grupos")
    print(f"🤖 Mejor modelo: {best_model} (R²={modelos[best_model]['metrics']['R2']:.4f})")
    print("=" * 60)
    print("\n🌐 Abriendo en http://127.0.0.1:8050")
    print("⏹️  Presiona Ctrl+C para detener el servidor")
    print("=" * 60)
    app.run(debug=True, port=8050)