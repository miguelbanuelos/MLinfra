import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pmdarima as pm
import psycopg2
import os
from dotenv import load_dotenv
import urllib.parse
import warnings

# Suppress math warnings caused by flatline data
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------------------------------------------------------------------------
# 1. Cargar Credenciales
# -------------------------------------------------------------------------
load_dotenv()

DB_USER = os.getenv("POSTGRES_USER", "admin")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
DB_DB = os.getenv("POSTGRES_DB", "homelab_metrics")
DB_PORT = os.getenv("TIMESCALEDB_PORT", "5432")
DB_HOST = os.getenv("TIMESCALEDB_HOST", "192.168.3.155")

safe_user = urllib.parse.quote_plus(DB_USER)
safe_password = urllib.parse.quote_plus(DB_PASSWORD)
DB_URL = f"postgresql://{safe_user}:{safe_password}@{DB_HOST}:{DB_PORT}/{DB_DB}?sslmode=disable"

# -------------------------------------------------------------------------
# 2. Motor SQL
# -------------------------------------------------------------------------
def load_sql(filename):
    filepath = os.path.join(os.path.dirname(__file__), 'sql', filename)
    with open(filepath, 'r') as file:
        return file.read()

def execute_query(sql_or_filename, params=None, is_select=True, is_file=True):
    """Ejecuta SQL. Puede recibir un archivo .sql o una cadena de texto directa."""
    query = load_sql(sql_or_filename) if is_file else sql_or_filename
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    if is_select:
        cur.execute(query, params)
        col_names = [desc[0] for desc in cur.description]
        data = cur.fetchall()
        df = pd.DataFrame(data, columns=col_names)
        cur.close()
        conn.close()
        return df
    else:
        cur.execute(query, params)
        conn.commit()
        cur.close()
        conn.close()

# -------------------------------------------------------------------------
# Inicialización de Dash
# -------------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
app.title = "HomeLab Analytics Platform"
server = app.server

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Forecast & Analytics", href="/")),
        dbc.NavItem(dbc.NavLink("Exclusions Management", href="/exclusions")),
    ],
    brand="⚙️ HomeLab Analytics Platform",
    brand_href="/",
    color="dark", dark=True, className="mb-4"
)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    dbc.Container(id='page-content', fluid=True)
])

# =========================================================================
# HELPER: POPULAR MENÚS
# =========================================================================
def get_host_options():
    df_hosts = execute_query('query_get_hosts.sql')
    if df_hosts is not None and not df_hosts.empty:
        opts = [{'label': h, 'value': h} for h in df_hosts['host']]
        return opts, df_hosts['host'].iloc[0]
    return [], None

# =========================================================================
# VISTA 1: FORECAST 
# =========================================================================
def layout_analytics():
    opts, default_val = get_host_options()
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Operation Filters", className="card-title text-info mb-4"),
                    
                    html.Label("1. Select Server:"),
                    dcc.Dropdown(id='dropdown-host', options=opts, value=default_val, className="text-dark mb-3"),
                    
                    html.Label("2. Target Metric:"),
                    dcc.Dropdown(id='dropdown-metric', options=[
                        {'label': 'CPU Usage (%)', 'value': 'cpu_percent_max'},
                        {'label': 'Memory Usage (%)', 'value': 'mem_percent_max'}
                    ], value='cpu_percent_max', className="text-dark mb-3", clearable=False),
                    
                    html.Label("3. Historical Data (Training):"),
                    dcc.Dropdown(id='dropdown-past-days', options=[
                        {'label': f'{d} Days in Past', 'value': d} for d in [10, 20, 30, 45]
                    ], value=10, className="text-dark mb-3", clearable=False),
                    
                    html.Label("4. Forecast Window (Future):"),
                    dcc.Dropdown(id='dropdown-future-days', options=[
                        {'label': f'{d} Days Ahead', 'value': d} for d in [5, 10, 15, 20, 30, 45]
                    ], value=5, className="text-dark mb-4", clearable=False),
                    
                    # Botón reactivo
                    dbc.Button("🚀 Generate Forecast", id='btn-forecast', color="primary", className="w-100 fw-bold")
                ])
            ], className="mb-4 shadow-sm border-primary")
        ], width=3),
        
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Capacity Prediction (SARIMAX)"),
                        dbc.CardBody([
                            # Spinner para que el usuario sepa que está cargando
                            dcc.Loading(
                                id="loading-forecast",
                                type="default",
                                children=dcc.Graph(id='graph-forecast')
                            ),
                            html.H6("Reference Metrics", className="mt-3 text-warning"),
                            html.Div(id='table-stats-container')
                        ])
                    ], className="mb-4 shadow-sm")
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Heatmap: Max Values per Hour and Day"),
                        dbc.CardBody([dcc.Graph(id='graph-heatmap')])
                    ], className="shadow-sm")
                ])
            ])
        ], width=9)
    ])

# =========================================================================
# VISTA 2: EXCLUSIONES
# =========================================================================
def layout_exclusions():
    opts, default_val = get_host_options()
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Register New Time Filter"),
                dbc.CardBody([
                    html.Label("Target Server:"),
                    dcc.Dropdown(id='excl-dropdown-host', options=opts, value=default_val, className="text-dark mb-2"),
                    html.Label("Start (YYYY-MM-DD HH:MM:SS):"),
                    dbc.Input(id='excl-start', placeholder="2026-05-29 02:00:00", className="mb-2"),
                    html.Label("End (YYYY-MM-DD HH:MM:SS):"),
                    dbc.Input(id='excl-end', placeholder="2026-05-29 03:30:00", className="mb-2"),
                    html.Label("Category:"),
                    dbc.Input(id='excl-cat', placeholder="Replica / Anomaly", className="mb-2"),
                    html.Label("Description:"),
                    dbc.Input(id='excl-desc', placeholder="Nightly backup", className="mb-3"),
                    dbc.Button("Save Exclusion", id='btn-add-excl', color="success", className="w-100"),
                    html.Div(id='add-excl-status', className="mt-2 text-center")
                ])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Active Exclusions Catalog"),
                dbc.CardBody([
                    html.Div(id='table-exclusions-container'),
                    html.Hr(className="my-4"),
                    html.H5("Remove Filter", className="text-danger"),
                    dbc.Row([
                        dbc.Col([dbc.Input(id='input-delete-id', type="number", placeholder="ID", className="me-2")], width=8),
                        dbc.Col([dbc.Button("Delete", id='btn-delete-excl', color="danger", className="w-100")], width=4)
                    ]),
                    html.Div(id='delete-excl-status', className="mt-2 text-center")
                ])
            ])
        ], width=8)
    ])

# -------------------------------------------------------------------------
# CALLBACKS
# -------------------------------------------------------------------------
@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    return layout_exclusions() if pathname == '/exclusions' else layout_analytics()

@app.callback(
    [Output('table-exclusions-container', 'children'), Output('add-excl-status', 'children'), Output('delete-excl-status', 'children')],
    [Input('btn-add-excl', 'n_clicks'), Input('btn-delete-excl', 'n_clicks'), Input('excl-dropdown-host', 'value')],
    [State('excl-start', 'value'), State('excl-end', 'value'), State('excl-cat', 'value'), State('excl-desc', 'value'), State('input-delete-id', 'value')]
)
def manage_exclusions(n_add, n_del, host, start, end, cat, desc, delete_id):
    ctx = dash.callback_context
    msg_add, msg_del = "", ""
    
    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'btn-add-excl' and all([host, start, end]):
            execute_query('query_insert_exclusion.sql', (host, start, end, cat, desc), is_select=False)
            msg_add = html.Span("Filter saved!", className="text-success")
        elif trigger_id == 'btn-delete-excl' and delete_id:
            execute_query('query_delete_exclusion.sql', (int(delete_id),), is_select=False)
            msg_del = html.Span(f"ID {delete_id} deleted.", className="text-warning")

    if host:
        df_excl = execute_query('query_exclusions.sql', (host,))
    else:
        df_excl = execute_query('query_all_exclusions.sql')
    
    if df_excl is None or df_excl.empty:
        return html.Div("No active exclusions."), msg_add, msg_del
        
    table = dash_table.DataTable(
        data=df_excl.to_dict('records'),
        columns=[{"name": i.upper(), "id": i} for i in df_excl.columns],
        style_header={'backgroundColor': '#222', 'color': '#0df', 'fontWeight': 'bold'},
        style_cell={'backgroundColor': '#333', 'color': 'white', 'textAlign': 'left'},
        page_size=6
    )
    return table, msg_add, msg_del


@app.callback(
    [Output('graph-forecast', 'figure'), Output('table-stats-container', 'children'), Output('graph-heatmap', 'figure')],
    [Input('btn-forecast', 'n_clicks')],
    [State('dropdown-host', 'value'), State('dropdown-metric', 'value'), State('dropdown-past-days', 'value'), State('dropdown-future-days', 'value')]
)
def process_analytics(n_clicks, host, metric, past_days, future_days):
    # Evita que el dashboard procese todo al abrir la página. Espera al clic.
    if not n_clicks:
        fig_empty = go.Figure().update_layout(template="plotly_dark", title="Ready. Adjust parameters and click Forecast.")
        return fig_empty, html.Div("Waiting for execution..."), fig_empty

    if not host:
        return go.Figure(), "Select a host", go.Figure()

    metric_label = "CPU" if "cpu" in metric else "Memory"

    # --- CONSULTA DINÁMICA A LA NUEVA TABLA "Servers" ---
    # Esto reemplaza al antiguo query_telemetry.sql
    query_telemetry = f"""
    SELECT timestamp AS "Time", "{metric}" AS "Usage"
    FROM "Servers"
    WHERE "ServerName" = %s AND timestamp >= NOW() - INTERVAL '{past_days} days'
    ORDER BY timestamp ASC;
    """
    
    df_telemetry = execute_query(query_telemetry, (host,), is_select=True, is_file=False)
    df_excl = execute_query('query_exclusions.sql', (host,))

    # 1. ALINEACIÓN DE REJILLA (Intervalos de 5 mins = 288 por día)
    intervalos_pasados = int(past_days) * 288 
    pasos_futuros = int(future_days) * 288
    
    now_floored = pd.Timestamp.now(tz='America/Mexico_City').floor('5min').tz_localize(None)
    esqueleto_tiempo = pd.date_range(end=now_floored, periods=intervalos_pasados, freq='5min')

    if df_telemetry is None or df_telemetry.empty:
        df_telemetry = pd.DataFrame(0, index=esqueleto_tiempo, columns=['Usage', 'Clean'])
        df_telemetry.index.name = 'Time'
    else:
        df_telemetry['Time'] = pd.to_datetime(df_telemetry['Time'])
        df_telemetry.set_index('Time', inplace=True)
        
        if df_telemetry.index.tz is None:
            df_telemetry.index = df_telemetry.index.tz_localize('UTC')
        
        df_telemetry.index = df_telemetry.index.tz_convert('America/Mexico_City').tz_localize(None)
        df_telemetry = df_telemetry[~df_telemetry.index.duplicated(keep='last')]
        df_telemetry = df_telemetry.reindex(esqueleto_tiempo)
        
        df_telemetry['Usage'] = df_telemetry['Usage'].ffill(limit=3).fillna(0)
        df_telemetry['Clean'] = df_telemetry['Usage'].copy()

    # 2. APLICAR EXCLUSIONES (Limpieza de anomalías)
    if df_excl is not None and not df_excl.empty:
        for _, row in df_excl.iterrows():
            start_dt = pd.to_datetime(row['start_time'])
            end_dt = pd.to_datetime(row['end_time'])
            df_telemetry.loc[start_dt:end_dt, 'Clean'] = np.nan
            
        df_telemetry['Clean'] = df_telemetry['Clean'].fillna(df_telemetry['Clean'].shift(1).rolling('3h', min_periods=1).mean())
        df_telemetry['Clean'] = df_telemetry['Clean'].bfill().fillna(0)

    
    # 3. HEATMAP
    df_heatmap_prep = df_telemetry.copy()
    df_heatmap_prep['Date'] = df_heatmap_prep.index.strftime('%Y-%m-%d')
    df_heatmap_prep['Hour'] = df_heatmap_prep.index.hour
    pivot_df = df_heatmap_prep.groupby(['Hour', 'Date'])['Usage'].max().unstack(level=1).fillna(0)
    
    # Escala de colores dinámica anclada a porcentajes
    custom_colorscale = [
        [0.0, '#00bc8c'],  
        [0.5, '#f1c40f'],  
        [0.8, '#ff4d4d'],  
        [1.0, '#8b0000']   
    ]
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=pivot_df.values, 
        x=pivot_df.columns, 
        y=pivot_df.index, 
        colorscale=custom_colorscale,
        zmin=0,   # Mantiene el verde anclado al 0% real
        zmax=100  # Mantiene el rojo oscuro anclado al 100% real
    ))
    
    fig_heatmap.update_layout(
        template="plotly_dark", 
        yaxis=dict(tickmode='linear', dtick=1), 
        margin=dict(l=40, r=20, t=20, b=40), 
        height=300
    )
# 4. ENTRENAMIENTO SARIMAX (OPTIMIZADO CON RESAMPLE)
    y_train = df_telemetry['Clean'].resample('1h').mean().fillna(0)
    
    # Preparamos las variables exógenas con el nuevo índice por hora
    X_train = pd.DataFrame(index=y_train.index)
    X_train['is_ai_running'] = 0.0
    X_train['is_offline_schedule'] = (X_train.index.hour < 11).astype(int)
    
    # Recalculamos los pasos futuros basados en horas (24 por día)
    pasos_futuros_horas = int(future_days) * 24
        
    fechas_futuras = pd.date_range(start=y_train.index[-1], periods=pasos_futuros_horas + 1, freq='1h')[1:]
    
    X_future = pd.DataFrame(index=fechas_futuras)
    X_future['is_ai_running'] = 0.0
    X_future['is_offline_schedule'] = (X_future.index.hour < 11).astype(int)
    
    try:
        model = pm.auto_arima(y_train, X=X_train, start_p=0, start_q=0, max_p=3, max_q=3, seasonal=False, error_action='ignore', suppress_warnings=True)
        
        # Predecimos usando los pasos calculados por hora
        prediccion, conf_int = model.predict(n_periods=pasos_futuros_horas, X=X_future, return_conf_int=True)
        
        ganador_arima = str(model.summary().tables[0].data[1][1]).strip()
        aic_val = f"{model.aic():.2f}"
        
    except Exception as e:
        fig_err = go.Figure()
        fig_err.update_layout(template="plotly_dark", title=f"Calibrating model... Error: {e}")
        return fig_err, "Insufficient data to converge", fig_heatmap

    # 5. DIBUJAR GRÁFICO PRINCIPAL
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=df_telemetry.index, y=df_telemetry['Usage'], name='Raw Unfiltered', line=dict(color='gray', width=1), opacity=0.5))
    fig_forecast.add_trace(go.Scatter(x=df_telemetry.index, y=df_telemetry['Clean'], name='Normalized Time Series', line=dict(color='#00bc8c', width=2)))
    
    fig_forecast.add_trace(go.Scatter(x=list(fechas_futuras), y=list(prediccion), name='Forecast', line=dict(color='#3498db', dash='dash', width=2)))
    
    fig_forecast.add_trace(go.Scatter(
        x=list(fechas_futuras) + list(fechas_futuras)[::-1], 
        y=list(conf_int[:, 1]) + list(conf_int[:, 0])[::-1], 
        fill='toself', fillcolor='rgba(52, 152, 219, 0.15)', line=dict(color='rgba(255,255,255,0)'), showlegend=True, name="95% Confidence"
    ))

    # LÍNEA DE UMBRAL CRÍTICO
    fig_forecast.add_hline(
        y=80, line_dash="dot", line_color="#ff4d4d", line_width=2,
        annotation_text="⚠️ Critical Threshold (80%)", annotation_position="top left", annotation_font=dict(color="#ff4d4d")
    )
    
    fig_forecast.update_layout(
        title=f"SARIMAX: {past_days} Days Past → {future_days} Days Future ({metric_label})",
        template="plotly_dark", margin=dict(l=40, r=20, t=40, b=40), height=350, legend=dict(orientation="h", y=1.02),
        yaxis=dict(range=[0, 105])
    )
    
    # 6. ESTADÍSTICAS FINALES
    stats_data = [
        {"Metric": "Target", "Value": metric_label.upper()},
        {"Metric": "ARIMA Winner", "Value": ganador_arima},
        {"Metric": "Akaike Info (AIC)", "Value": aic_val},
        {"Metric": "Average (Window)", "Value": f"{df_telemetry['Usage'].mean():.2f}%"}
    ]
    
    table_stats = dbc.Table([html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])), html.Tbody([html.Tr([html.Td(st["Metric"]), html.Td(st["Value"])]) for st in stats_data])], bordered=True, hover=True, striped=True, size="sm", className="text-white bg-dark")

    return fig_forecast, table_stats, fig_heatmap

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8099, debug=False)