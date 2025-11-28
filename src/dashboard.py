"""
Dashboard Interactivo de Análisis de Ventas
Basado en el análisis multivariado del notebook principal
"""

import dash
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from math import pi
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =============================================================================
# CARGA Y PREPARACION DE DATOS
# =============================================================================

def cargar_datos():
    """Carga y prepara todos los datos necesarios"""
    # Cargar datasets
    categorias = pd.read_csv("../data_raw/categorias.csv")
    clientes = pd.read_csv("../data_raw/clientes.csv")
    productos = pd.read_csv("../data_raw/productos.csv")
    ventas = pd.read_csv("../data_raw/ventas.csv")
    metodos_pago = pd.read_csv("../data_raw/metodos_pago.csv")
    
    # Limpiar precio unitario
    productos["Precio_Unitario"] = productos["Precio_Unitario"].str.replace(",", ".").astype(float)
    
    # Crear DataFrame principal
    df = ventas.merge(clientes, on="ID_Cliente", how="inner")
    df.drop(columns=["Email"], inplace=True)
    df = df.merge(productos[["ID_Producto", "Precio_Unitario"]], on="ID_Producto", how="inner")
    df["Cantidad_dinero"] = df["Cantidad"] * df["Precio_Unitario"]
    df = df.merge(productos[["ID_Producto", "Stock"]], on="ID_Producto", how="inner")
    df = df.merge(productos[["ID_Producto", "Categoría"]], on="ID_Producto", how="left")
    
    # Calcular stock actual
    dict_stock = productos.set_index("ID_Producto")["Stock"].to_dict()
    cantidad_de_producto_actual = []
    for cantidad, id_producto in zip(df["Cantidad"], df["ID_Producto"]):
        dict_stock[id_producto] -= cantidad
        cantidad_de_producto_actual.append(dict_stock[id_producto])
    df["Producto_actual_stock"] = cantidad_de_producto_actual
    
    # Convertir fecha
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d/%m/%Y")
    df["Dia_semana"] = df["Fecha"].dt.day_name()
    df["Mes"] = df["Fecha"].dt.month
    
    # Productos vendidos
    productos_vendidos = df.groupby("ID_Producto")["Producto_actual_stock"].min().reset_index()
    productos_vendidos = productos_vendidos.merge(productos[["ID_Producto", "Stock"]], on="ID_Producto", how="inner")
    productos_vendidos["Cantidad_vendida"] = productos_vendidos["Stock"] - productos_vendidos["Producto_actual_stock"]
    
    return df, productos, productos_vendidos, categorias

def preparar_clusters(df, productos_vendidos):
    """Prepara datos de clustering"""
    cluster_data = productos_vendidos[['ID_Producto', 'Cantidad_vendida', 'Stock', 'Producto_actual_stock']].copy()
    cluster_data["Porcentaje_venta"] = (cluster_data["Cantidad_vendida"] / cluster_data["Stock"]) * 100
    
    # Agregar categoria al cluster_data
    producto_categoria = df.groupby('ID_Producto')['Categoría'].first().reset_index()
    cluster_data = cluster_data.merge(producto_categoria, on='ID_Producto', how='left')
    
    # Agregar region mas frecuente por producto
    producto_region = df.groupby('ID_Producto')['Región'].agg(lambda x: x.value_counts().index[0]).reset_index()
    producto_region.columns = ['ID_Producto', 'Region_principal']
    cluster_data = cluster_data.merge(producto_region, on='ID_Producto', how='left')
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_data[['Cantidad_vendida', 'Stock', 'Producto_actual_stock', 'Porcentaje_venta']])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_data['Cluster'] = kmeans.fit_predict(X_scaled)
    
    return cluster_data, kmeans, scaler

def entrenar_modelos(df, cluster_data):
    """Entrena modelos de boosting y otros"""
    df_regression = df.groupby('ID_Producto').agg({
        'Cantidad': 'sum',
        'Cantidad_dinero': 'sum',
        'Precio_Unitario': 'first',
        'Región': lambda x: x.value_counts().index[0],
        'Categoría': 'first'
    }).reset_index()
    
    df_regression = df_regression.merge(
        cluster_data[['ID_Producto', 'Cluster', 'Stock', 'Cantidad_vendida', 'Porcentaje_venta', 'Producto_actual_stock']], 
        on='ID_Producto'
    )
    
    df_regression = pd.get_dummies(df_regression, columns=['Región', 'Categoría'], drop_first=True)
    
    feature_cols = [col for col in df_regression.columns if col not in 
                   ['ID_Producto', 'Cluster', 'Cantidad_dinero', 'Cantidad_vendida', 'Cantidad']]
    
    X = df_regression[feature_cols]
    y = df_regression['Cantidad_dinero']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    modelos = {}
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    modelos['Gradient Boosting'] = {
        'model': gb,
        'predictions': y_pred_gb,
        'metrics': {
            'R2': r2_score(y_test, y_pred_gb),
            'RMSE': mean_squared_error(y_test, y_pred_gb) ** 0.5,
            'MAE': mean_absolute_error(y_test, y_pred_gb)
        }
    }
    
    # AdaBoost
    ada = AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    ada.fit(X_train, y_train)
    y_pred_ada = ada.predict(X_test)
    modelos['AdaBoost'] = {
        'model': ada,
        'predictions': y_pred_ada,
        'metrics': {
            'R2': r2_score(y_test, y_pred_ada),
            'RMSE': mean_squared_error(y_test, y_pred_ada) ** 0.5,
            'MAE': mean_absolute_error(y_test, y_pred_ada)
        }
    }
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    modelos['Random Forest'] = {
        'model': rf,
        'predictions': y_pred_rf,
        'metrics': {
            'R2': r2_score(y_test, y_pred_rf),
            'RMSE': mean_squared_error(y_test, y_pred_rf) ** 0.5,
            'MAE': mean_absolute_error(y_test, y_pred_rf)
        }
    }
    
    # SVR (Support Vector Regression)
    # Escalar datos para SVR
    scaler_svr = StandardScaler()
    X_train_scaled = scaler_svr.fit_transform(X_train)
    X_test_scaled = scaler_svr.transform(X_test)
    
    svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
    svr.fit(X_train_scaled, y_train)
    y_pred_svr = svr.predict(X_test_scaled)
    modelos['SVR'] = {
        'model': svr,
        'predictions': y_pred_svr,
        'metrics': {
            'R2': r2_score(y_test, y_pred_svr),
            'RMSE': mean_squared_error(y_test, y_pred_svr) ** 0.5,
            'MAE': mean_absolute_error(y_test, y_pred_svr)
        },
        'scaler': scaler_svr
    }
    
    return modelos, feature_cols, X_test, y_test

# Cargar datos
df, productos, productos_vendidos, categorias = cargar_datos()
cluster_data, kmeans, scaler = preparar_clusters(df, productos_vendidos)
modelos, feature_cols, X_test, y_test = entrenar_modelos(df, cluster_data)

# =============================================================================
# CALCULOS GLOBALES
# =============================================================================

total_ventas = df['Cantidad_dinero'].sum()
total_unidades = df['Cantidad'].sum()
ticket_promedio = df['Cantidad_dinero'].mean()
productos_unicos = df['ID_Producto'].nunique()
clientes_unicos = df['ID_Cliente'].nunique()
regiones_activas = df['Región'].nunique()

# Ventas por region
ventas_region = df.groupby('Región').agg({
    'Cantidad_dinero': 'sum',
    'Cantidad': 'sum'
}).reset_index()
ventas_region.columns = ['Región', 'Ingresos', 'Unidades']
ventas_region = ventas_region.sort_values('Ingresos', ascending=False)

# Ventas por categoria
ventas_categoria = df.groupby('Categoría').agg({
    'Cantidad_dinero': 'sum',
    'Cantidad': 'sum'
}).reset_index()
ventas_categoria.columns = ['Categoría', 'Ingresos', 'Unidades']
ventas_categoria = ventas_categoria.sort_values('Ingresos', ascending=False)

dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dias_espanol = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo']
ventas_dia = df.groupby('Dia_semana')['Cantidad_dinero'].sum().reindex(dias_orden)

# Calcular medias por cluster para radar chart
cluster_means = cluster_data.groupby('Cluster')[['Cantidad_vendida', 'Stock', 'Producto_actual_stock', 'Porcentaje_venta']].mean()
cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

best_model = max(modelos.keys(), key=lambda x: modelos[x]['metrics']['R2'])

# =============================================================================
# FUNCIONES AUXILIARES PARA FILTRADO
# =============================================================================

def filtrar_datos(df_original, region, categoria):
    """Filtra el dataframe segun region y categoria"""
    df_filtrado = df_original.copy()
    if region != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['Región'] == region]
    if categoria != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['Categoría'] == categoria]
    return df_filtrado

def filtrar_clusters(cluster_original, region, categoria):
    """Filtra los datos de cluster segun region y categoria"""
    cluster_filtrado = cluster_original.copy()
    if region != 'Todas':
        cluster_filtrado = cluster_filtrado[cluster_filtrado['Region_principal'] == region]
    if categoria != 'Todas':
        cluster_filtrado = cluster_filtrado[cluster_filtrado['Categoría'] == categoria]
    return cluster_filtrado

def calcular_kpis(df_filtrado):
    """Calcula KPIs del dataframe filtrado"""
    return {
        'total_ventas': df_filtrado['Cantidad_dinero'].sum(),
        'total_unidades': df_filtrado['Cantidad'].sum(),
        'ticket_promedio': df_filtrado['Cantidad_dinero'].mean() if len(df_filtrado) > 0 else 0,
        'productos_unicos': df_filtrado['ID_Producto'].nunique(),
        'clientes_unicos': df_filtrado['ID_Cliente'].nunique(),
        'regiones_activas': df_filtrado['Región'].nunique()
    }

# =============================================================================
# INICIALIZAR APP DASH
# =============================================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Dashboard de Analisis de Ventas"

# =============================================================================
# LAYOUT
# =============================================================================

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Dashboard de Analisis Multivariado de Ventas", className="text-center my-4"),
            html.Hr()
        ])
    ]),
    
    # Filtros
    dbc.Row([
        dbc.Col([
            html.Label("Seleccionar Region:", className="text-light"),
            dcc.Dropdown(
                id='filtro-region',
                options=[{'label': 'Todas', 'value': 'Todas'}] + 
                        [{'label': r, 'value': r} for r in sorted(df['Región'].unique())],
                value='Todas',
                className="mb-3",
                style={'color': 'black'}
            )
        ], width=4),
        dbc.Col([
            html.Label("Seleccionar Categoria:", className="text-light"),
            dcc.Dropdown(
                id='filtro-categoria',
                options=[{'label': 'Todas', 'value': 'Todas'}] + 
                        [{'label': str(c), 'value': c} for c in sorted(df['Categoría'].dropna().unique())],
                value='Todas',
                className="mb-3",
                style={'color': 'black'}
            )
        ], width=4),
        dbc.Col([
            html.Label("Seleccionar Modelo:", className="text-light"),
            dcc.Dropdown(
                id='filtro-modelo',
                options=[{'label': m, 'value': m} for m in modelos.keys()],
                value=best_model,
                className="mb-3",
                style={'color': 'black'}
            )
        ], width=4),
    ], className="mb-4"),
    
    # KPIs
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Ingresos Totales", className="text-muted mb-2"),
                    html.H4(id='kpi-ingresos', className="text-primary mb-0")
                ])
            ], className="mb-3 shadow-sm")
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Unidades Vendidas", className="text-muted mb-2"),
                    html.H4(id='kpi-unidades', className="text-primary mb-0")
                ])
            ], className="mb-3 shadow-sm")
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Ticket Promedio", className="text-muted mb-2"),
                    html.H4(id='kpi-ticket', className="text-primary mb-0")
                ])
            ], className="mb-3 shadow-sm")
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Productos Unicos", className="text-muted mb-2"),
                    html.H4(id='kpi-productos', className="text-primary mb-0")
                ])
            ], className="mb-3 shadow-sm")
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Clientes", className="text-muted mb-2"),
                    html.H4(id='kpi-clientes', className="text-primary mb-0")
                ])
            ], className="mb-3 shadow-sm")
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Regiones", className="text-muted mb-2"),
                    html.H4(id='kpi-regiones', className="text-primary mb-0")
                ])
            ], className="mb-3 shadow-sm")
        ], width=2),
    ], className="mb-4"),
    
    # Graficos Fila 1
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Tendencia de Ventas"),
                dbc.CardBody([dcc.Graph(id='grafico-tendencia')])
            ])
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Top Regiones"),
                dbc.CardBody([dcc.Graph(id='grafico-regiones')])
            ])
        ], width=4),
    ], className="mb-4"),
    
    # Graficos Fila 2 - Clusters
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Mapa de Clusters 3D"),
                dbc.CardBody([dcc.Graph(id='grafico-clusters-3d')])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Distribucion por Cluster"),
                dbc.CardBody([dcc.Graph(id='grafico-clusters-pie')])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Perfil de Clusters (Radar)"),
                dbc.CardBody([dcc.Graph(id='grafico-radar-clusters')])
            ])
        ], width=4),
    ], className="mb-4"),
    
    # Graficos Fila 3 - Estacionalidad y Stock
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Estacionalidad Semanal"),
                dbc.CardBody([dcc.Graph(id='grafico-estacionalidad')])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Stock Total vs Stock Restante"),
                dbc.CardBody([dcc.Graph(id='grafico-stock')])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Cantidad Vendida por Producto"),
                dbc.CardBody([dcc.Graph(id='grafico-cantidad-vendida')])
            ])
        ], width=4),
    ], className="mb-4"),
    
    # Graficos Fila 4 - Modelos
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Comparacion de Modelos"),
                dbc.CardBody([dcc.Graph(id='grafico-comparacion-modelos')])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Importancia de Variables"),
                dbc.CardBody([dcc.Graph(id='grafico-importancia')])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Predicciones vs Real"),
                dbc.CardBody([dcc.Graph(id='grafico-predicciones')])
            ])
        ], width=4),
    ], className="mb-4"),
    
    # Graficos Fila 5 - Heatmap y Categorias
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Heatmap: Ventas por Dia y Mes"),
                dbc.CardBody([dcc.Graph(id='grafico-heatmap')])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Ventas por Categoria"),
                dbc.CardBody([dcc.Graph(id='grafico-categorias')])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Metricas del Modelo"),
                dbc.CardBody([
                    html.Div(id='metricas-modelo')
                ])
            ])
        ], width=3),
    ], className="mb-4"),
    
    # Graficos Fila 6 - Distribucion y Correlacion
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Distribucion de Ingresos por Venta"),
                dbc.CardBody([dcc.Graph(id='grafico-distribucion')])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Matriz de Correlacion"),
                dbc.CardBody([dcc.Graph(id='grafico-correlacion')])
            ])
        ], width=6),
    ], className="mb-4"),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("Dashboard desarrollado para Analisis Multivariado de Ventas", 
                  className="text-center text-muted")
        ])
    ])
    
], fluid=True, className="bg-dark")

# =============================================================================
# CALLBACKS
# =============================================================================

# Callback para actualizar KPIs
@callback(
    [Output('kpi-ingresos', 'children'),
     Output('kpi-unidades', 'children'),
     Output('kpi-ticket', 'children'),
     Output('kpi-productos', 'children'),
     Output('kpi-clientes', 'children'),
     Output('kpi-regiones', 'children')],
    [Input('filtro-region', 'value'),
     Input('filtro-categoria', 'value')]
)
def actualizar_kpis(region, categoria):
    df_filtrado = filtrar_datos(df, region, categoria)
    kpis = calcular_kpis(df_filtrado)
    
    return (
        f"${kpis['total_ventas']:,.2f}",
        f"{kpis['total_unidades']:,}",
        f"${kpis['ticket_promedio']:.2f}",
        f"{kpis['productos_unicos']:,}",
        f"{kpis['clientes_unicos']:,}",
        f"{kpis['regiones_activas']:,}"
    )

# Callback para grafico de tendencia
@callback(
    Output('grafico-tendencia', 'figure'),
    [Input('filtro-region', 'value'),
     Input('filtro-categoria', 'value')]
)
def actualizar_tendencia(region, categoria):
    df_filtrado = filtrar_datos(df, region, categoria)
    
    if len(df_filtrado) == 0:
        return go.Figure().update_layout(template='plotly_dark', title='Sin datos')
    
    ventas_temp = df_filtrado.groupby('Fecha')['Cantidad_dinero'].sum().reset_index()
    ventas_temp = ventas_temp.sort_values('Fecha')
    ventas_temp['MA7'] = ventas_temp['Cantidad_dinero'].rolling(window=7, min_periods=1).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ventas_temp['Fecha'], 
        y=ventas_temp['Cantidad_dinero'],
        mode='lines+markers',
        name='Ingresos Diarios',
        line=dict(color='#00bc8c', width=2),
        marker=dict(size=4)
    ))
    fig.add_trace(go.Scatter(
        x=ventas_temp['Fecha'], 
        y=ventas_temp['MA7'],
        mode='lines',
        name='Media Movil 7 dias',
        line=dict(color='#e74c3c', width=2, dash='dash')
    ))
    fig.update_layout(
        template='plotly_dark',
        xaxis_title='Fecha',
        yaxis_title='Ingresos ($)',
        margin=dict(l=20, r=20, t=20, b=20),
        height=300,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    return fig

# Callback para grafico de regiones
@callback(
    Output('grafico-regiones', 'figure'),
    [Input('filtro-region', 'value'),
     Input('filtro-categoria', 'value')]
)
def actualizar_regiones(region, categoria):
    df_filtrado = filtrar_datos(df, region, categoria)
    
    if len(df_filtrado) == 0:
        return go.Figure().update_layout(template='plotly_dark', title='Sin datos')
    
    ventas_temp = df_filtrado.groupby('Región')['Cantidad_dinero'].sum().sort_values(ascending=True).tail(10)
    
    fig = go.Figure(go.Bar(
        y=ventas_temp.index,
        x=ventas_temp.values,
        orientation='h',
        marker_color='#3498db'
    ))
    fig.update_layout(
        template='plotly_dark',
        xaxis_title='Ingresos ($)',
        margin=dict(l=20, r=20, t=20, b=20),
        height=300
    )
    return fig

# Callback para clusters 3D
@callback(
    Output('grafico-clusters-3d', 'figure'),
    [Input('filtro-region', 'value'),
     Input('filtro-categoria', 'value')]
)
def actualizar_clusters_3d(region, categoria):
    cluster_filtrado = filtrar_clusters(cluster_data, region, categoria)
    
    if len(cluster_filtrado) == 0:
        return go.Figure().update_layout(template='plotly_dark', title='Sin datos')
    
    colors_cluster = ['#440154', '#31688e', '#35b779', '#fde725']
    
    fig = go.Figure()
    
    for cluster_id in sorted(cluster_filtrado['Cluster'].unique()):
        cluster_subset = cluster_filtrado[cluster_filtrado['Cluster'] == cluster_id]
        
        fig.add_trace(go.Scatter3d(
            x=cluster_subset['Stock'],
            y=cluster_subset['Cantidad_vendida'],
            z=cluster_subset['Porcentaje_venta'],
            mode='markers',
            name=f'Cluster {cluster_id}',
            marker=dict(size=6, color=colors_cluster[cluster_id % len(colors_cluster)], opacity=0.8),
            text=[f'ID: {id}<br>Stock: {s}<br>Vendido: {v}' 
                  for id, s, v in zip(cluster_subset['ID_Producto'], 
                                      cluster_subset['Stock'],
                                      cluster_subset['Cantidad_vendida'])],
            hoverinfo='text'
        ))
    
    fig.update_layout(
        template='plotly_dark',
        scene=dict(
            xaxis_title='Stock',
            yaxis_title='Cant. Vendida',
            zaxis_title='% Venta'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=350
    )
    return fig

# Callback para clusters pie
@callback(
    Output('grafico-clusters-pie', 'figure'),
    [Input('filtro-region', 'value'),
     Input('filtro-categoria', 'value')]
)
def actualizar_clusters_pie(region, categoria):
    cluster_filtrado = filtrar_clusters(cluster_data, region, categoria)
    
    if len(cluster_filtrado) == 0:
        return go.Figure().update_layout(template='plotly_dark', title='Sin datos')
    
    cluster_dist = cluster_filtrado['Cluster'].value_counts().sort_index()
    
    fig = go.Figure(go.Pie(
        labels=[f'Cluster {i}' for i in cluster_dist.index],
        values=cluster_dist.values,
        hole=0.4,
        marker=dict(colors=['#440154', '#31688e', '#35b779', '#fde725'])
    ))
    fig.update_layout(
        template='plotly_dark',
        margin=dict(l=20, r=20, t=20, b=20),
        height=350
    )
    return fig

# Callback para radar de clusters
@callback(
    Output('grafico-radar-clusters', 'figure'),
    [Input('filtro-region', 'value'),
     Input('filtro-categoria', 'value')]
)
def actualizar_radar_clusters(region, categoria):
    cluster_filtrado = filtrar_clusters(cluster_data, region, categoria)
    
    if len(cluster_filtrado) == 0:
        return go.Figure().update_layout(template='plotly_dark', title='Sin datos')
    
    # Calcular medias por cluster filtrado
    cluster_means_filt = cluster_filtrado.groupby('Cluster')[['Cantidad_vendida', 'Stock', 'Producto_actual_stock', 'Porcentaje_venta']].mean()
    
    if len(cluster_means_filt) == 0:
        return go.Figure().update_layout(template='plotly_dark', title='Sin datos')
    
    # Normalizar
    cluster_means_norm_filt = (cluster_means_filt - cluster_means_filt.min()) / (cluster_means_filt.max() - cluster_means_filt.min() + 0.001)
    
    categories = list(cluster_means_norm_filt.columns)
    colors_cluster = ['#440154', '#31688e', '#35b779', '#fde725']
    
    fig = go.Figure()
    
    for idx, (cluster, row) in enumerate(cluster_means_norm_filt.iterrows()):
        values = row.values.tolist()
        values += values[:1]  # Cerrar el poligono
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=f'Cluster {cluster}',
            line=dict(color=colors_cluster[idx % len(colors_cluster)])
        ))
    
    fig.update_layout(
        template='plotly_dark',
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        margin=dict(l=40, r=40, t=20, b=20),
        height=350
    )
    return fig

# Callback para estacionalidad
@callback(
    Output('grafico-estacionalidad', 'figure'),
    [Input('filtro-region', 'value'),
     Input('filtro-categoria', 'value')]
)
def actualizar_estacionalidad(region, categoria):
    df_filtrado = filtrar_datos(df, region, categoria)
    
    if len(df_filtrado) == 0:
        return go.Figure().update_layout(template='plotly_dark', title='Sin datos')
    
    ventas_temp = df_filtrado.groupby('Dia_semana')['Cantidad_dinero'].sum()
    ventas_temp = ventas_temp.reindex(dias_orden).fillna(0)
    
    fig = go.Figure(go.Bar(
        x=dias_espanol,
        y=ventas_temp.values,
        marker_color='#e74c3c'
    ))
    fig.update_layout(
        template='plotly_dark',
        xaxis_title='Dia',
        yaxis_title='Ingresos ($)',
        margin=dict(l=20, r=20, t=20, b=20),
        height=300
    )
    return fig

# Callback para stock
@callback(
    Output('grafico-stock', 'figure'),
    [Input('filtro-region', 'value'),
     Input('filtro-categoria', 'value')]
)
def actualizar_stock(region, categoria):
    cluster_filtrado = filtrar_clusters(cluster_data, region, categoria)
    
    if len(cluster_filtrado) == 0:
        return go.Figure().update_layout(template='plotly_dark', title='Sin datos')
    
    productos_sorted = cluster_filtrado.sort_values('Cantidad_vendida', ascending=False).head(15)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=productos_sorted['ID_Producto'].astype(str),
        y=productos_sorted['Stock'],
        name='Stock Total',
        marker_color='#3498db'
    ))
    fig.add_trace(go.Bar(
        x=productos_sorted['ID_Producto'].astype(str),
        y=productos_sorted['Producto_actual_stock'],
        name='Stock Restante',
        marker_color='#e74c3c'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        barmode='overlay',
        xaxis_title='ID Producto',
        yaxis_title='Cantidad',
        margin=dict(l=20, r=20, t=20, b=20),
        height=300,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    return fig

# Callback para cantidad vendida
@callback(
    Output('grafico-cantidad-vendida', 'figure'),
    [Input('filtro-region', 'value'),
     Input('filtro-categoria', 'value')]
)
def actualizar_cantidad_vendida(region, categoria):
    cluster_filtrado = filtrar_clusters(cluster_data, region, categoria)
    
    if len(cluster_filtrado) == 0:
        return go.Figure().update_layout(template='plotly_dark', title='Sin datos')
    
    productos_sorted = cluster_filtrado.sort_values('Cantidad_vendida', ascending=False).head(15)
    
    fig = go.Figure(go.Bar(
        x=productos_sorted['ID_Producto'].astype(str),
        y=productos_sorted['Cantidad_vendida'],
        marker=dict(
            color=productos_sorted['Cantidad_vendida'],
            colorscale='Inferno',
            showscale=True
        )
    ))
    
    fig.update_layout(
        template='plotly_dark',
        xaxis_title='ID Producto',
        yaxis_title='Cantidad Vendida',
        margin=dict(l=20, r=20, t=20, b=20),
        height=300
    )
    return fig

# Callback para comparacion de modelos
@callback(
    Output('grafico-comparacion-modelos', 'figure'),
    Input('filtro-modelo', 'value')
)
def actualizar_comparacion(_):
    comparison_df = pd.DataFrame({
        'Modelo': list(modelos.keys()),
        'R2': [modelos[m]['metrics']['R2'] for m in modelos],
        'RMSE': [modelos[m]['metrics']['RMSE'] for m in modelos]
    })
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=['R2 Score', 'RMSE'])
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    
    fig.add_trace(go.Bar(
        x=comparison_df['Modelo'],
        y=comparison_df['R2'],
        marker_color=colors[:len(comparison_df)],
        name='R2'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=comparison_df['Modelo'],
        y=comparison_df['RMSE'],
        marker_color=colors[:len(comparison_df)],
        name='RMSE'
    ), row=1, col=2)
    
    fig.update_layout(
        template='plotly_dark',
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        height=300
    )
    return fig

# Callback para importancia de variables
@callback(
    Output('grafico-importancia', 'figure'),
    Input('filtro-modelo', 'value')
)
def actualizar_importancia(modelo_seleccionado):
    # SVR no tiene feature_importances_
    if modelo_seleccionado == 'SVR':
        fig = go.Figure()
        fig.add_annotation(
            text="SVR no proporciona importancia de variables directamente",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="white")
        )
        fig.update_layout(
            template='plotly_dark',
            margin=dict(l=20, r=20, t=20, b=20),
            height=300
        )
        return fig
    
    model = modelos[modelo_seleccionado]['model']
    importances = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=True).tail(10)
    
    fig = go.Figure(go.Bar(
        y=importance_df['Feature'],
        x=importance_df['Importance'],
        orientation='h',
        marker_color='#9b59b6'
    ))
    fig.update_layout(
        template='plotly_dark',
        xaxis_title='Importancia',
        margin=dict(l=20, r=20, t=20, b=20),
        height=300
    )
    return fig

# Callback para predicciones
@callback(
    Output('grafico-predicciones', 'figure'),
    Input('filtro-modelo', 'value')
)
def actualizar_predicciones(modelo_seleccionado):
    y_pred = modelos[modelo_seleccionado]['predictions']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        marker=dict(color='#2ecc71', opacity=0.6),
        name='Predicciones'
    ))
    
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Linea Ideal'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        xaxis_title='Valores Reales',
        yaxis_title='Predicciones',
        margin=dict(l=20, r=20, t=20, b=20),
        height=300,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    return fig

# Callback para heatmap
@callback(
    Output('grafico-heatmap', 'figure'),
    [Input('filtro-region', 'value'),
     Input('filtro-categoria', 'value')]
)
def actualizar_heatmap(region, categoria):
    df_filtrado = filtrar_datos(df, region, categoria)
    
    if len(df_filtrado) == 0:
        return go.Figure().update_layout(template='plotly_dark', title='Sin datos')
    
    heatmap_data = df_filtrado.groupby(['Dia_semana', 'Mes'])['Cantidad_dinero'].sum().unstack(fill_value=0)
    heatmap_data = heatmap_data.reindex(dias_orden).fillna(0)
    
    meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    fig = go.Figure(go.Heatmap(
        z=heatmap_data.values,
        x=[meses_nombres[m-1] for m in heatmap_data.columns],
        y=dias_espanol,
        colorscale='YlOrRd',
        hovertemplate='Dia: %{y}<br>Mes: %{x}<br>Ingresos: $%{z:,.2f}<extra></extra>'
    ))
    fig.update_layout(
        template='plotly_dark',
        xaxis_title='Mes',
        yaxis_title='Dia de la Semana',
        margin=dict(l=20, r=20, t=20, b=20),
        height=300
    )
    return fig

# Callback para categorias
@callback(
    Output('grafico-categorias', 'figure'),
    [Input('filtro-region', 'value'),
     Input('filtro-categoria', 'value')]
)
def actualizar_categorias(region, categoria):
    df_filtrado = filtrar_datos(df, region, categoria)
    
    if len(df_filtrado) == 0:
        return go.Figure().update_layout(template='plotly_dark', title='Sin datos')
    
    ventas_cat = df_filtrado.groupby('Categoría')['Cantidad_dinero'].sum().sort_values(ascending=False).head(10)
    
    fig = go.Figure(go.Bar(
        x=ventas_cat.index.astype(str),
        y=ventas_cat.values,
        marker_color='#f39c12'
    ))
    fig.update_layout(
        template='plotly_dark',
        xaxis_title='Categoria',
        yaxis_title='Ingresos ($)',
        margin=dict(l=20, r=20, t=20, b=20),
        height=300
    )
    return fig

# Callback para distribucion de ingresos
@callback(
    Output('grafico-distribucion', 'figure'),
    [Input('filtro-region', 'value'),
     Input('filtro-categoria', 'value')]
)
def actualizar_distribucion(region, categoria):
    df_filtrado = filtrar_datos(df, region, categoria)
    
    if len(df_filtrado) == 0:
        return go.Figure().update_layout(template='plotly_dark', title='Sin datos')
    
    fig = go.Figure(go.Histogram(
        x=df_filtrado['Cantidad_dinero'],
        nbinsx=30,
        marker_color='#3498db',
        opacity=0.7
    ))
    fig.update_layout(
        template='plotly_dark',
        xaxis_title='Cantidad Dinero ($)',
        yaxis_title='Frecuencia',
        margin=dict(l=20, r=20, t=20, b=20),
        height=300
    )
    return fig

# Callback para matriz de correlacion
@callback(
    Output('grafico-correlacion', 'figure'),
    [Input('filtro-region', 'value'),
     Input('filtro-categoria', 'value')]
)
def actualizar_correlacion(region, categoria):
    df_filtrado = filtrar_datos(df, region, categoria)
    
    if len(df_filtrado) == 0:
        return go.Figure().update_layout(template='plotly_dark', title='Sin datos')
    
    corr_matrix = df_filtrado[['Cantidad', 'Precio_Unitario', 'Cantidad_dinero']].corr()
    
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 12},
        hovertemplate='%{x} vs %{y}: %{z:.2f}<extra></extra>'
    ))
    fig.update_layout(
        template='plotly_dark',
        margin=dict(l=20, r=20, t=20, b=20),
        height=300
    )
    return fig

# Callback para metricas del modelo
@callback(
    Output('metricas-modelo', 'children'),
    Input('filtro-modelo', 'value')
)
def actualizar_metricas(modelo_seleccionado):
    metrics = modelos[modelo_seleccionado]['metrics']
    
    return html.Div([
        html.H5(f"{modelo_seleccionado}", className="text-primary mb-4"),
        html.Div([
            html.Div([
                html.P("R2 Score", className="text-muted mb-1"),
                html.H4(f"{metrics['R2']:.4f}", className="text-success")
            ], className="mb-3"),
            html.Div([
                html.P("RMSE", className="text-muted mb-1"),
                html.H4(f"{metrics['RMSE']:.2f}", className="text-warning")
            ], className="mb-3"),
            html.Div([
                html.P("MAE", className="text-muted mb-1"),
                html.H4(f"{metrics['MAE']:.2f}", className="text-info")
            ])
        ])
    ])

# =============================================================================
# EJECUTAR APP
# =============================================================================

if __name__ == '__main__':
    print("Iniciando Dashboard...")
    print(f"Datos cargados: {len(df)} registros")
    print(f"Clusters: {cluster_data['Cluster'].nunique()} grupos")
    print(f"Modelos disponibles: {list(modelos.keys())}")
    print(f"Mejor modelo: {best_model}")
    print("\nAbriendo en http://127.0.0.1:8050")
    app.run(debug=True, port=8050)