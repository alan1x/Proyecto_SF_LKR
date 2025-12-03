# Análisis de Patrones de Ventas

- Luis Alan Morales Castillo (A01659147)
- Paulina Díaz Arroyo (A01029592)
- Rodrigo Jiménez Ortiz (A01029623)

## Descripción del Proyecto

Este proyecto realiza un análisis integral de datos de ventas de una empresa, aplicando técnicas de análisis multivariado para:

- **Segmentar productos** según su rendimiento y rotación de inventario
- **Segmentar clientes** según su comportamiento de compra
- **Predecir ingresos** utilizando modelos de regresión
- **Pronosticar ventas futuras** con series temporales
- **Visualizar resultados** en un dashboard interactivo

---

## Estructura del Proyecto

```
proyecto/
│
├── data/
│   ├── clientes.csv
│   ├── metodos_pago.csv
│   ├── productos.csv
│   ├── productos_vendidos.csv
│   └── ventas.csv
│
├── notebooks/
│   └── codigo.ipynb
│
├── src/
|   └── dashboard.py
│
├── reports/
│   └── presentacion.pdf
│   └── Proyecto_SF_LKR (1).pdf
```

---

## Datos Utilizados

| Archivo                  | Descripción                                         |
| ------------------------ | --------------------------------------------------- |
| `ventas.csv`             | Transacciones de venta con fecha, cliente, producto |
| `clientes.csv`           | Información demográfica y región de clientes        |
| `productos.csv`          | Catálogo con precios y categorías                   |
| `productos_vendidos.csv` | Stock y cantidades vendidas por producto            |
| `metodos_pago.csv`       | Catálogo de métodos de pago                         |

---

## Conclusiones Principales

### Hallazgos Clave

1. **Concentración de ingresos**: El 20% de los clientes (Premium) genera el 60% de los ingresos
2. **Estacionalidad semanal**: Las ventas son 25% mayores los fines de semana
3. **Productos estrella**: Solo el 15% de productos representan el 50% de ventas
4. **Regiones**: Existe disparidad significativa entre regiones, oportunidad de expansión

### Recomendaciones de Negocio

| Área           | Recomendación                                   | Impacto Esperado    |
| -------------- | ----------------------------------------------- | ------------------- |
| **Inventario** | Optimizar stock de productos "Bajo Rendimiento" | -15% costos almacén |
| **Marketing**  | Focalizar en clientes "Premium y Frecuentes"    | +20% retención      |
| **Ventas**     | Promociones para "Cazadores de Oferta"          | +10% conversión     |
| **Producto**   | Expandir línea de productos "Estrella"          | +25% ingresos       |
| **Expansión**  | Replicar estrategias de regiones exitosas       | +15% cobertura      |

---

## Ejecución

### Instalación

```bash
cd /ruta/al/proyecto
pip install -r requirements.txt
```

### Ejecución

#### Opción 1: Dashboard Interactivo

```bash
cd src
python3 dashboard.py
```

#### Opción 2: Jupyter Notebook

```bash
cd notebooks
jupyter notebook codigo.ipynb
```

---

## Dashboard

| Pestaña        | Contenido                                                     |
| -------------- | ------------------------------------------------------------- |
| **Resumen**    | KPIs principales, tendencias, distribuciones                  |
| **Productos**  | Clustering 3D, radar de segmentos, tabla detallada            |
| **Clientes**   | Clustering 3D, radar de segmentos, tabla detallad             |
| **Predicción** | Comparación de modelos, predicciones, importancia de features |
| **Forecast**   | Series temporales, pronósticos                                |
| **Conclusión** | Recomendaciones finales                                       |

---

## Contributors

<a href="https://github.com/alan1x/Proyecto_SF_LKR.git/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=alan1x/Proyecto_SF_LKR&anon=1&max=10" />
</a>
