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

# Suppress math warnings caused by flatline data (e.g., servers turned off)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------------------------------------------------------------------------
# 1. Load Credentials from .env and Sanitize URL
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
# 2. Secure SQL Execution Engine
# -------------------------------------------------------------------------
def load_sql(filename):
    """Loads a SQL file from the /sql folder"""
    filepath = os.path.join(os.path.dirname(__file__), 'sql', filename)
    with open(filepath, 'r') as file:
        return file.read()

def execute_query(sql_filename, params=None, is_select=True):
    """Executes a native query with psycopg2 and converts it cleanly to Pandas."""
    query = load_sql(sql_filename)
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
# Dash Initialization
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
# HELPER FUNCTION TO POPULATE MENUS DYNAMICALLY
# =========================================================================
def get_host_options():
    df_hosts = execute_query('query_get_hosts.sql')
    if df_hosts is not None and not df_hosts.empty:
        opts = [{'label': h, 'value': h} for h in df_hosts['host']]
        return opts, df_hosts['host'].iloc[0]
    return [], None

# =========================================================================
# VIEW 1: FORECAST AND HEATMAP LAYOUT
# =========================================================================
def layout_analytics():
    opts, default_val = get_host_options()
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Operation Filters", className="card-title text-info"),
                    html.Label("Select Server/Host:"),
                    dcc.Dropdown(id='dropdown-host', options=opts, value=default_val, className="text-dark mb-3"),
                    html.Label("Training Strategy (Past -> Future):"),
                    dcc.Dropdown(
                        id='dropdown-ventana',
                        options=[
                            {'label': '360 Intervals (30 hrs) -> Predicts 180', 'value': 360},
                            {'label': '720 Intervals (60 hrs) -> Predicts 360', 'value': 720}
                        ],
                        value=360, className="text-dark"
                    ),
                ])
            ], className="mb-4")
        ], width=3),
        
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Capacity Prediction with SARIMAX Scenario"),
                        dbc.CardBody([
                            dcc.Graph(id='graph-forecast'),
                            html.H6("Reference Metrics", className="mt-3 text-warning"),
                            html.Div(id='table-stats-container')
                        ])
                    ], className="mb-4")
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Heatmap: Max CPU Values per Hour and Day"),
                        dbc.CardBody([dcc.Graph(id='graph-heatmap')])
                    ])
                ])
            ])
        ], width=9)
    ])

# =========================================================================
# VIEW 2: EXCLUSIONS MANAGEMENT
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
    
    # 1. Process button clicks (Save or Delete)
    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'btn-add-excl' and all([host, start, end]):
            execute_query('query_insert_exclusion.sql', (host, start, end, cat, desc), is_select=False)
            msg_add = html.Span("Filter saved!", className="text-success")
        elif trigger_id == 'btn-delete-excl' and delete_id:
            execute_query('query_delete_exclusion.sql', (int(delete_id),), is_select=False)
            msg_del = html.Span(f"ID {delete_id} deleted.", className="text-warning")

    # 2. Dynamic Table Logic: Filter by selected host, or show all if none selected
    if host:
        df_excl = execute_query('query_exclusions.sql', (host,))
    else:
        df_excl = execute_query('query_all_exclusions.sql')
    
    # 3. Build the visual table
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
    [Input('dropdown-host', 'value'), Input('dropdown-ventana', 'value')]
)
def process_analytics(host, ventana_intervalos):
    if not host:
        return go.Figure(), "Select a host", go.Figure()

    # Calculate history hours (e.g. 360 * 5 / 60 = 30 hours)
    horas_historial = (int(ventana_intervalos) * 5) / 60
    df_telemetry = execute_query('query_telemetry.sql', (host, horas_historial))
    df_excl = execute_query('query_exclusions.sql', (host,))

    # 1. THE MAGIC OF ZERO: Build the perfect time skeleton anchored to CST
    now_floored = pd.Timestamp.now(tz='America/Mexico_City').floor('5min').tz_localize(None)
    esqueleto_tiempo = pd.date_range(end=now_floored, periods=int(ventana_intervalos), freq='5min')

    if df_telemetry.empty:
        df_telemetry = pd.DataFrame(0, index=esqueleto_tiempo, columns=['CPU_Usage', 'CPU_Clean'])
        df_telemetry.index.name = 'Time'
    else:
        df_telemetry['Time'] = pd.to_datetime(df_telemetry['Time'])
        df_telemetry.set_index('Time', inplace=True)
        
        # Ensure UTC timezone is declared, then cast to CST (America/Mexico_City)
        if df_telemetry.index.tz is None:
            df_telemetry.index = df_telemetry.index.tz_localize('UTC')
        
        # Shift DB data to match your local timezone, then strip tz for ARIMA compatibility
        df_telemetry.index = df_telemetry.index.tz_convert('America/Mexico_City').tz_localize(None)
        
        # Clean duplicates after shifting timezone
        df_telemetry = df_telemetry[~df_telemetry.index.duplicated(keep='last')]
        
        # Paste the localized DB data onto the perfect CST skeleton
        df_telemetry = df_telemetry.reindex(esqueleto_tiempo)
        
        # Fill micro-cuts and force long drops to zero
        df_telemetry['CPU_Usage'] = df_telemetry['CPU_Usage'].ffill(limit=3).fillna(0)
        df_telemetry['CPU_Clean'] = df_telemetry['CPU_Usage'].copy()

    # 2. Apply exclusions safely
    # Note: Since exclusions are entered manually as naive CST strings, they will perfectly match the CST index
    if not df_excl.empty:
        for _, row in df_excl.iterrows():
            start_dt = pd.to_datetime(row['start_time'])
            end_dt = pd.to_datetime(row['end_time'])
            df_telemetry.loc[start_dt:end_dt, 'CPU_Clean'] = np.nan
            
        df_telemetry['CPU_Clean'] = df_telemetry['CPU_Clean'].fillna(df_telemetry['CPU_Clean'].shift(1).rolling('3h', min_periods=1).mean())
        df_telemetry['CPU_Clean'] = df_telemetry['CPU_Clean'].bfill().fillna(0)

    # 3. Heatmap
    df_heatmap_prep = df_telemetry.copy()
    df_heatmap_prep['Date'] = df_heatmap_prep.index.strftime('%Y-%m-%d')
    df_heatmap_prep['Hour'] = df_heatmap_prep.index.hour
    pivot_df = df_heatmap_prep.groupby(['Hour', 'Date'])['CPU_Usage'].max().unstack(level=1).fillna(0)
    
    fig_heatmap = go.Figure(data=go.Heatmap(z=pivot_df.values, x=pivot_df.columns, y=pivot_df.index, colorscale='Viridis'))
    fig_heatmap.update_layout(template="plotly_dark", yaxis=dict(tickmode='linear', dtick=1), margin=dict(l=40, r=20, t=20, b=40), height=300)

    # 4. Auto-ARIMA Preparation with Exogenous Variables (Offline Schedule)
    y_train = df_telemetry['CPU_Clean']
    
    X_train = pd.DataFrame(index=y_train.index)
    X_train['is_ai_running'] = 0.0
    # Mathematically notify that from 00:00 to 10:59 local time the equipment sleeps
    X_train['is_offline_schedule'] = (X_train.index.hour < 11).astype(int)
    
    pasos_futuros = int(int(ventana_intervalos) / 2)
    fechas_futuras = pd.date_range(start=esqueleto_tiempo[-1], periods=pasos_futuros + 1, freq='5min')[1:]
    
    X_future = pd.DataFrame(index=fechas_futuras)
    X_future['is_ai_running'] = 0.0
    X_future['is_offline_schedule'] = (X_future.index.hour < 11).astype(int)
    
    try:
        # Training
        model = pm.auto_arima(y_train, X=X_train, start_p=0, start_q=0, max_p=3, max_q=3, seasonal=False, error_action='ignore', suppress_warnings=True)
        prediccion, conf_int = model.predict(n_periods=pasos_futuros, X=X_future, return_conf_int=True)
        
        ganador_arima = str(model.summary().tables[0].data[1][1]).strip()
        aic_val = f"{model.aic():.2f}"
        
    except Exception as e:
        fig_err = go.Figure()
        fig_err.update_layout(template="plotly_dark", title=f"Calibrating model (Waiting for more clean data)... Detail: {e}")
        return fig_err, "Insufficient data to converge", fig_heatmap

    # 5. Draw Main Graph
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=df_telemetry.index, y=df_telemetry['CPU_Usage'], name='Raw (DB)', line=dict(color='gray', width=1), opacity=0.5))
    fig_forecast.add_trace(go.Scatter(x=df_telemetry.index, y=df_telemetry['CPU_Clean'], name='Filtered (Exclusions)', line=dict(color='#00bc8c', width=2)))
    
    fig_forecast.add_trace(go.Scatter(x=list(fechas_futuras), y=list(prediccion), name='Forecast', line=dict(color='#3498db', dash='dash', width=2)))
    
    fig_forecast.add_trace(go.Scatter(
        x=list(fechas_futuras) + list(fechas_futuras)[::-1], 
        y=list(conf_int[:, 1]) + list(conf_int[:, 0])[::-1], 
        fill='toself', fillcolor='rgba(52, 152, 219, 0.15)', line=dict(color='rgba(255,255,255,0)'), showlegend=True, name="95% Confidence"
    ))

    # --- THRESHOLD LINE (80%) ---
    fig_forecast.add_hline(
        y=80, 
        line_dash="dot", 
        line_color="#ff4d4d",
        line_width=2,
        annotation_text="⚠️ Critical Threshold (80%)", 
        annotation_position="top left",
        annotation_font=dict(color="#ff4d4d")
    )
    
    fig_forecast.update_layout(
        title=f"SARIMAX: {int(ventana_intervalos)} Past Intervals → {pasos_futuros} Future",
        template="plotly_dark", margin=dict(l=40, r=20, t=40, b=40), height=350, legend=dict(orientation="h", y=1.02),
        yaxis=dict(range=[0, 105]) # Keeps the Y-axis fixed up to 100% so the threshold is always visible
    )
    
    # 6. Final Statistics
    stats_data = [
        {"Metric": "ARIMA Winner", "Value": ganador_arima},
        {"Metric": "Akaike Info Criterion (AIC)", "Value": aic_val},
        {"Metric": "Average CPU (Window)", "VALUE": f"{df_telemetry['CPU_Usage'].mean():.2f}%"}
    ]
    
    table_stats = dbc.Table([html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])), html.Tbody([html.Tr([html.Td(st["Metric"]), html.Td(st.get("Value") or st.get("VALUE"))]) for st in stats_data])], bordered=True, hover=True, striped=True, size="sm", className="text-white bg-dark")

    return fig_forecast, table_stats, fig_heatmap

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8099, debug=True)