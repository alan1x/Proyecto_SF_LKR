### Proyecto Final

# Análisis Multivariado y Predictivo de Ventas

Este proyecto se centra en el análisis exploratorio de datos (EDA), la preparación de datos, la segmentación de productos mediante técnicas de *clustering* (KMeans) y la construcción de modelos de regresión para predecir los ingresos totales por producto (*Cantidad_dinero*).

## 1. Requisitos Técnicos y Dependencias

Para ejecutar el análisis, se requiere tener instalado Python y las siguientes librerías, las cuales fueron utilizadas a lo largo del proceso:

| Librería | Propósito principal |
| :--- | :--- |
| **pandas** | Manipulación y carga de datos. |
| **matplotlib** | Generación de gráficos estáticos (e.g., Boxplot, Método del Codo). |
| **seaborn** | Visualizaciones estadísticas (e.g., Matriz de correlación, Boxplots). |
| **plotly.express/graph_objects** | Gráficos interactivos y Dashboards (e.g., Cluster 3D). |
| **sklearn** | Normalización (StandardScaler), Clustering (KMeans), Métrica (silhouette_score), Modelos de Regresión y métricas (LinearRegression, train_test_split, mean_squared_error, r2_score). |
| **statsmodels** | Regresión OLS para obtener estadísticas detalladas y R² ajustado. |
| **Ensemble Models** | Modelos de predicción global (GradientBoostingRegressor, AdaBoostRegressor, XGBRegressor). |
| **scipy** | Funcionalidades para jerarquías de clústeres y otras utilidades (Dendrograma). |
| **numpy** | Soporte para operaciones numéricas avanzadas. |

## 2. Estructura de Archivos

Los archivos de datos brutos (`.csv`) deben estar ubicados en la ruta relativa `../data_raw/` para que el script pueda leerlos correctamente.
```
data_raw/
   ├── categorias.csv
   ├── clientes.csv
   ├── metodos_pago.csv
   ├── productos.csv
   └── ventas.csv
```
*   `categorias.csv`
*   `clientes.csv`
*   `productos.csv` (Contiene `Precio_Unitario` y `Stock` inicial)
*   `ventas.csv` (Contiene `Cantidad` vendida y `ID_Venta`)
*   `metodos_pago.csv`

## 3. Instrucciones de Ejecución

El análisis sigue una secuencia lógica de preparación y modelado. Para replicar los resultados:

1.  **Carga y Limpieza Inicial:** Cargar todos los archivos `.csv`. La columna `Precio_Unitario` del *dataframe* `productos` debe ser convertida a tipo flotante, reemplazando las comas (",") por puntos (".").
2.  **Unión de Datos (*Merge*):** Fusionar los *dataframes* `ventas`, `clientes` y `productos` para crear el *dataframe* principal (`df`). Se calcula la variable objetivo `Cantidad_dinero` como `Cantidad * Precio_Unitario`.
3.  **Preparación de Variables:** Calcular el stock actual y la `Cantidad_vendida` por producto, agrupando los datos.
4.  **Segmentación (Clustering):**
    *   Preparar los datos de clúster utilizando `Cantidad_vendida`, `Stock`, `Producto_actual_stock`, y `Porcentaje_venta`.
    *   Normalizar los datos utilizando `StandardScaler`.
    *   Aplicar KMeans con $k=4$ (seleccionado como el mejor equilibrio entre el Método del Codo y el Coeficiente de Silhouette).
5.  **Modelado Predictivo:**
    *   Ejecutar Regresión Múltiple (OLS) de forma segmentada (por cada uno de los 4 clústeres) para evaluar la predicción de `Cantidad_dinero`.
    *   Ejecutar Modelos de Boosting de forma global (Gradient Boosting, AdaBoost y XGBoost) para comparar el rendimiento de predicción general.
6.  **Visualización:** Generar los gráficos de resumen, la matriz de correlación, la interpretación de clústeres, la comparación de modelos de Boosting y el Dashboard Consolidado Final.

