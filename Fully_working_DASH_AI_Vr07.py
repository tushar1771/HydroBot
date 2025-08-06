# =============================================================================
# FINAL DASHBOARD V6.1: ROBUST, ERROR-CORRECTED, ADVANCED MODULES
# =============================================================================

# --- Step 1: Import Libraries ---
import pandas as pd
import geopandas as gpd
from dash import Dash, dcc, html, callback_context
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
from threading import Timer
import warnings
import os
import re
from openai import OpenAI
import xarray as xr
import rioxarray as rio
from shapely.geometry import box
import numpy as np
from scipy.stats import linregress
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore', 'PROJ is deprecated')
warnings.filterwarnings("ignore", category=UserWarning, module='statsmodels')

# --- Step 2: Configuration ---
AQUIFER_PATH = r'E:\Data\MajorAquifers\Major_Aquifer.geojson'
BOUNDARIES_PATH = r'E:\Data\MajorAquifers\District_State.geojson'
GROUNDWATER_PATH = r'E:\Data\MajorAquifers\Groundwater_Level.geojson'
GRACE_PATH = r'E:\Data\MajorAquifers\GRCTellus.JPL.200204_202505.GLO.RL06.3M.MSCNv04CRI.nc'
RAINFALL_FOLDER_PATH = r'E:\Data\MajorAquifers\Rainfall'

os.environ["HF_TOKEN"] = "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"
HF_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct:fireworks-ai"

# --- Helper functions and Knowledge Bases (Unchanged) ---
def standardize_crs(gdf: gpd.GeoDataFrame, layer_name: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs("EPSG:4326", allow_override=True)
    elif gdf.crs.to_epsg() != 4326:
        return gdf.to_crs("EPSG:4326")
    return gdf

def categorize_water_level(depth):
    if pd.isna(depth) or depth < 0: return 'Recharge'
    elif 0 <= depth < 30: return 'Shallow (0-30m)'
    elif 30 <= depth < 60: return 'Moderate (30-60m)'
    elif 60 <= depth < 100: return 'Deep (60-100m)'
    else: return 'Very Deep (>100m)'

def load_prepare_rainfall_year(year, files_dict):
    if year not in files_dict: return None
    try:
        with xr.open_dataset(files_dict[year], decode_times=False) as ds:
            if 'RAINFALL' in ds.variables: rain_var_name, lon_name, lat_name, time_name = 'RAINFALL', 'LONGITUDE', 'LATITUDE', 'TIME'
            elif 'rf' in ds.variables: rain_var_name, lon_name, lat_name, time_name = 'rf', 'lon', 'lat', 'time'
            else: return None
            time_units = ds[time_name].attrs.get('units', 'days since 1901-01-01')
            start_date_str = time_units.split('since ')[1].replace(" 00:00:00", "")
            decoded_dates = pd.to_datetime(start_date_str) + pd.to_timedelta(ds[time_name].values, unit='D')
            ds = ds.assign_coords({time_name: decoded_dates})
            ds = ds.rename({time_name: 'time'})
            data_var = ds[rain_var_name].rename({lon_name: 'x', lat_name: 'y'})
            data_var.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
            data_var.rio.set_crs("EPSG:4326", inplace=True)
            return data_var
    except Exception as e:
        return None

rock_properties = {
    "Alluvium": {"Porosity": "Very High", "Yield": "Very High", "Suitability": "Excellent", "Specific Yield": 0.15, "ASI_Score": 5},
    "Basalt": {"Porosity": "Moderate", "Yield": "Moderate", "Suitability": "Good (if fractured)", "Specific Yield": 0.05, "ASI_Score": 4},
    "Basement Gneissic Complex": {"Porosity": "Very Low", "Yield": "Very Low", "Suitability": "Poor", "Specific Yield": 0.02, "ASI_Score": 1},
    "Schist": {"Porosity": "Low", "Yield": "Low", "Suitability": "Poor", "Specific Yield": 0.02, "ASI_Score": 1},
    "Granite": {"Porosity": "Low", "Yield": "Low", "Suitability": "Poor", "Specific Yield": 0.02, "ASI_Score": 1},
    "Sandstone": {"Porosity": "High", "Yield": "High", "Suitability": "Very Good", "Specific Yield": 0.12, "ASI_Score": 5},
    "Intrusive": {"Porosity": "Low", "Yield": "Low", "Suitability": "Poor", "Specific Yield": 0.02, "ASI_Score": 1},
    "Quartzite": {"Porosity": "Low", "Yield": "Low", "Suitability": "Poor", "Specific Yield": 0.01, "ASI_Score": 1},
    "Laterite": {"Porosity": "Moderate", "Yield": "Moderate", "Suitability": "Variable", "Specific Yield": 0.06, "ASI_Score": 3},
    "Shale": {"Porosity": "Low", "Yield": "Poor", "Suitability": "Aquitard", "Specific Yield": 0.01, "ASI_Score": 2},
    "Gneiss": {"Porosity": "Low", "Yield": "Low", "Suitability": "Poor", "Specific Yield": 0.02, "ASI_Score": 1},
    "Limestone": {"Porosity": "Moderate", "Yield": "Variable to High", "Suitability": "Variable", "Specific Yield": 0.08, "ASI_Score": 3},
    "Charnockite": {"Porosity": "Low", "Yield": "Low", "Suitability": "Poor", "Specific Yield": 0.01, "ASI_Score": 1},
    "Khondalites": {"Porosity": "Low", "Yield": "Low", "Suitability": "Poor", "Specific Yield": 0.01, "ASI_Score": 1},
    "Unknown": {"Porosity": "Unknown", "Yield": "Unknown", "Suitability": "Unknown", "Specific Yield": 0.03, "ASI_Score": 2},
}
age_influence = { "Azoic": "Very old, hard rock, fracture-dominated", "Proterozoic to Azoic": "Very old, hard rock, fracture-dominated", "Cenozoic, Proterozoic": "Mixed ages, variable properties", "(Cenozoic, Mesozoic, Upper Paleaozoic)": "Mixed ages, variable properties", "Quarternary": "Unconsolidated -> high porosity/permeability", "Quaternary": "Unconsolidated -> high porosity/permeability", "Cenozoic, Mesozoic": "Mixed ages, variable properties", "Proterozoic": "Very old, hard rock, fracture-dominated", "Cenozoic to Proterozoic": "Mixed ages, variable properties", "Unknown": "Unknown age influence", }

# --- Data Loading (Unchanged) ---
print("Loading and processing all data sources...")
try:
    boundaries_gdf = standardize_crs(gpd.read_file(BOUNDARIES_PATH), "Boundaries")
    boundaries_gdf.columns = [col.lower().strip() for col in boundaries_gdf.columns]
    boundaries_gdf = boundaries_gdf.rename(columns={'st_nm': 'state'})
    aquifers_gdf = standardize_crs(gpd.read_file(AQUIFER_PATH), "Aquifers")
    aquifers_gdf.columns = [col.lower().strip() for col in aquifers_gdf.columns]
    def enrich_row(row):
        rock = row['principal_'] if row['principal_'] in rock_properties else "Unknown"
        age = row['age'] if row['age'] in age_influence else "Unknown"
        props, age_note = rock_properties[rock], age_influence[age]
        return pd.Series({"porosity": props["Porosity"], "yield": props["Yield"], "suitability": props["Suitability"], "age_note": age_note, "specific_yield": props["Specific Yield"], "asi_score": props["ASI_Score"]})
    aquifers_gdf = pd.concat([aquifers_gdf, aquifers_gdf.apply(enrich_row, axis=1)], axis=1)
    aquifers_gdf['geometry'] = aquifers_gdf.geometry.buffer(0)
    points_gdf = standardize_crs(gpd.read_file(GROUNDWATER_PATH), "Groundwater Wells")
    points_gdf.columns = [col.lower().strip() for col in points_gdf.columns]
    points_gdf['date'] = pd.to_datetime(points_gdf['date'], errors='coerce')
    points_gdf['gwl'] = pd.to_numeric(points_gdf['gwl'], errors='coerce', downcast='float')
    points_gdf.dropna(subset=['date', 'gwl', 'geometry'], inplace=True)
    points_gdf['year'] = pd.to_numeric(points_gdf['date'].dt.year, downcast='integer')
    points_gdf['month'] = pd.to_numeric(points_gdf['date'].dt.month, downcast='integer')
    points_gdf['gwl_category'] = points_gdf['gwl'].apply(categorize_water_level)
    points_gdf['latitude'] = points_gdf.geometry.y
    points_gdf['longitude'] = points_gdf.geometry.x
    points_gdf['hovertext'] = points_gdf.apply(lambda row: f"Date: {row['date'].strftime('%Y-%m-%d')}<br>GWL: {row['gwl']:.2f} m<br>Category: {row['gwl_category']}", axis=1)
    full_points_gdf = points_gdf.copy()
    processed_data_dict = points_gdf.drop(columns=['geometry', 'date']).to_dict('records')
    
    # ### START: ERROR CORRECTION FOR GRACE DATA ###
    grace_full_ds = xr.open_dataset(GRACE_PATH)
    grace_ds = grace_full_ds['lwe_thickness']
    grace_ds = grace_ds.rename({'lon': 'x', 'lat': 'y'})
    
    # Convert longitude from 0-360 to -180 to 180
    grace_ds.coords['x'] = (grace_ds.coords['x'] + 180) % 360 - 180
    # Sort by the new coordinate system to ensure it's ordered correctly
    grace_ds = grace_ds.sortby(grace_ds.x)
    
    grace_ds.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
    # CRS is WGS84, which corresponds to EPSG:4326
    grace_ds.rio.write_crs("EPSG:4326", inplace=True)
    # ### END: ERROR CORRECTION FOR GRACE DATA ###

    rainfall_files_dict = {}
    if os.path.isdir(RAINFALL_FOLDER_PATH):
        for filename in os.listdir(RAINFALL_FOLDER_PATH):
            if filename.endswith(('.nc', '.NC')):
                match = re.search(r'(\d{4})', filename)
                if match:
                    year = int(match.group(1))
                    rainfall_files_dict[year] = os.path.join(RAINFALL_FOLDER_PATH, filename)
    state_options = [{'label': 'All States', 'value': 'ALL'}] + [{'label': s, 'value': s} for s in sorted(boundaries_gdf['state'].unique())]
    min_year = points_gdf['year'].min()
    max_year = points_gdf['year'].max()
except Exception as e:
    print(f"FATAL ERROR loading data: {e}")
    boundaries_gdf, aquifers_gdf, full_points_gdf, grace_ds, rainfall_files_dict = [gpd.GeoDataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame(), None, {}]
    processed_data_dict, state_options, min_year, max_year = [], [], 2000, 2024

# --- App Layout (Unchanged from previous version) ---
app = Dash(__name__)
app.layout = html.Div([
    dcc.Store(id='groundwater-data-store', data=processed_data_dict),
    dcc.Store(id='trend-data-store'),
    dcc.Store(id='storage-analysis-store'),
    dcc.Store(id='advanced-analysis-trends-store'),
    dcc.Download(id="download-excel"),
    html.H1("HydroBot-AI Multi-View Geospatial Dashboard", style={'textAlign': 'center'}),
    html.Div([
        html.Div([html.Label("Select State:"), dcc.Dropdown(id='state-dropdown', options=state_options, value='ALL')], style={'flex': '1', 'padding': '5px'}),
        html.Div([html.Label("Select District:"), dcc.Dropdown(id='district-dropdown', value='ALL', disabled=True)], style={'flex': '1', 'padding': '5px'}),
        html.Div([html.Label("Toggle Map Layers:"), dcc.Checklist(id='layer-checklist', options=[{'label': 'Aquifers (Base)', 'value': 'aquifers'}, {'label': 'Wells', 'value': 'wells'}, {'label': 'GRACE TWS', 'value': 'grace'}, {'label': 'Rainfall', 'value': 'rainfall'}], value=['aquifers', 'wells'], labelStyle={'display': 'block'})], style={'flex': '1', 'padding': '5px', 'verticalAlign': 'top'})
    ], style={'display': 'flex', 'flexDirection': 'row', 'padding': '10px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px'}),
    html.Div([
        html.Div([html.Label("Select Year for Map View:"), dcc.Slider(id='year-slider', min=min_year, max=max_year, value=max_year, marks={i: str(i) for i in range(min_year, max_year + 1, 5)}, step=1)], style={'flex': '2', 'padding': '5px'}),
        html.Div([html.Label("Select Month for Map View:"), dcc.Slider(id='month-slider', min=1, max=12, value=1, marks={i: m for i, m in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 1)}, step=1)], style={'flex': '2', 'padding': '5px'})
    ], style={'display': 'flex', 'flexDirection': 'row', 'padding': '10px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px', 'marginTop': '5px'}),
    html.Div([
        html.H3("Holistic AI Consultant", style={'textAlign': 'center'}),
        dcc.Textarea(id='ai-question-input', placeholder='e.g., "Provide a full water security assessment for this district."', style={'width': '100%', 'height': '50px'}),
        html.Button('Ask AI Consultant', id='ai-ask-button', n_clicks=0, style={'marginTop': '10px'}),
        dcc.Loading(id="loading-ai-answer", type="default", children=dcc.Markdown(id='ai-answer-output', style={'marginTop': '10px', 'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': '#f9f9f9'}))
    ], style={'padding': '10px 20px'}),
    html.Div([
        html.Div(id='map-container-1', children=[dcc.Graph(id='map-1', style={'height': '65vh'})]),
        html.Div(id='map-container-2', children=[dcc.Graph(id='map-2', style={'height': '65vh'})]),
        html.Div(id='map-container-3', children=[dcc.Graph(id='map-3', style={'height': '65vh'})]),
    ], style={'display': 'flex', 'flexDirection': 'row'}),
    html.Div([
        html.H3("Full Time Series Trend Analysis", style={'textAlign': 'center'}),
        dcc.Loading(id="loading-timeseries", type="default", children=dcc.Graph(id='trend-plot')),
    ], style={'padding': '20px', 'borderTop': '2px solid #ddd', 'marginTop': '20px'}),
    html.Div([
        html.H3("Annual Groundwater Storage Fluctuation Analysis", style={'textAlign': 'center'}),
        dcc.Loading(id="loading-storage-plot", type="default", children=dcc.Graph(id='storage-plot')),
    ], style={'padding': '20px', 'borderTop': '2px solid #ddd', 'marginTop': '20px'}),
    html.Div([
        html.H2("Advanced Hydrogeological Modules", style={'textAlign': 'center', 'borderBottom': '2px solid black', 'paddingBottom': '10px'}),
        dcc.Checklist(
            id='show-advanced-checklist', options=[{'label': 'Show Advanced Analysis Modules', 'value': 'SHOW'}], value=[],
            style={'textAlign': 'center', 'fontWeight': 'bold', 'marginBottom': '10px'}),
        html.Div(id='advanced-analysis-container', children=[
            html.P("Select a module to quantify complex hydrogeological dynamics.", style={'textAlign': 'center'}),
            dcc.Dropdown(
                id='advanced-analysis-dropdown',
                options=[
                    {'label': 'Spatio-Temporal Aquifer Stress Score (SASS)', 'value': 'SASS'},
                    {'label': 'GRACE vs. Ground Reality Divergence Heatmap', 'value': 'GRACE_DIVERGENCE'},
                    {'label': 'Aquifer Suitability Index (ASI)', 'value': 'ASI'},
                    {'label': 'AI-Based Time-Series Forecasting & Alert System', 'value': 'FORECAST'},
                    {'label': 'Recharge Structure Recommendation Engine', 'value': 'RECHARGE'},
                ], value='SASS', style={'width': '50%', 'margin': '0 auto'}),
            html.Button('Run Advanced Analysis', id='advanced-analysis-button', n_clicks=0, style={'display': 'block', 'margin': '10px auto'}),
            dcc.Loading(id='loading-advanced-analysis', children=[html.Div(id='advanced-analysis-output')])
        ], style={'display': 'none', 'border': '1px solid #ccc', 'padding': '15px', 'marginTop': '20px', 'borderRadius': '5px'})
    ], style={'padding': '20px'}),
])

# --- Callbacks ---

@app.callback(Output('district-dropdown', 'options'), Output('district-dropdown', 'value'), Output('district-dropdown', 'disabled'), Input('state-dropdown', 'value'))
def update_district_dropdown(selected_state):
    if selected_state == 'ALL' or not selected_state: return [], 'ALL', True
    districts = sorted(boundaries_gdf[boundaries_gdf['state'] == selected_state]['district'].unique())
    options = [{'label': 'All Districts', 'value': 'ALL'}] + [{'label': d, 'value': d} for d in districts]
    return options, 'ALL', False

# --- CORRECTED: Restored full functionality to the map view callback ---
@app.callback(
    [Output('map-container-1', 'style'), Output('map-1', 'figure'),
     Output('map-container-2', 'style'), Output('map-2', 'figure'),
     Output('map-container-3', 'style'), Output('map-3', 'figure')],
    [Input('state-dropdown', 'value'), Input('district-dropdown', 'value'),
     Input('year-slider', 'value'), Input('month-slider', 'value'),
     Input('layer-checklist', 'value'), Input('groundwater-data-store', 'data')]
)
def update_map_views(state, district, year, month, layers, data):
    style_hidden = {'display': 'none'}
    style_1_view = {'display': 'inline-block', 'width': '100%', 'padding': '2px'}
    style_2_view = {'display': 'inline-block', 'width': '50%', 'padding': '2px'}
    style_3_view = {'display': 'inline-block', 'width': '33.3%', 'padding': '2px'}
    map_center, map_zoom = {"lat": 20.5937, "lon": 78.9629}, 4
    selection_boundary = None
    if district and district != 'ALL': selection_boundary = boundaries_gdf[boundaries_gdf['district'] == district]
    elif state and state != 'ALL': selection_boundary = boundaries_gdf[boundaries_gdf['state'] == state]
    if selection_boundary is not None and not selection_boundary.empty:
        bounds = selection_boundary.total_bounds
        map_center, map_zoom = ({"lat": (bounds[1] + bounds[3]) / 2, "lon": (bounds[0] + bounds[2]) / 2}, 8 if district and district != 'ALL' else 6)
    
    aquifers_clipped, points_df = aquifers_gdf, pd.DataFrame(data)
    if selection_boundary is not None and not selection_boundary.empty:
        try:
            aquifers_clipped = gpd.clip(aquifers_gdf, selection_boundary)
            points_as_gdf = gpd.GeoDataFrame(points_df, geometry=gpd.points_from_xy(points_df['longitude'], points_df['latitude']), crs="EPSG:4326")
            points_df = pd.DataFrame(gpd.clip(points_as_gdf, selection_boundary).drop(columns='geometry'))
        except: aquifers_clipped, points_df = gpd.GeoDataFrame(), pd.DataFrame()
    
    def create_aquifer_basemap(fig):
        if 'aquifers' in layers and not aquifers_clipped.empty:
            fig.add_trace(go.Choroplethmap(geojson=aquifers_clipped.geometry.__geo_interface__, locations=aquifers_clipped.index, z=aquifers_clipped.index, colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']], marker_line_color='blue', marker_line_width=1.5, showscale=False, hoverinfo='text', text=[f"<b>{row.get('majoraquif', 'N/A')}</b>" for i, row in aquifers_clipped.iterrows()]))
        return fig
    def add_wells_layer(fig, points_data, selected_year):
        category_colors = {'Recharge': 'blue', 'Shallow (0-30m)': 'green', 'Moderate (30-60m)': 'orange', 'Deep (60-100m)': 'red', 'Very Deep (>100m)': 'darkred'}
        if not points_data.empty:
            year_df = points_data[points_data['year'] == selected_year]
            for category, color in category_colors.items():
                category_df = year_df[year_df['gwl_category'] == category]
                if not category_df.empty:
                    fig.add_trace(go.Scattermap(lat=category_df['latitude'], lon=category_df['longitude'], mode='markers', marker=dict(size=9, color=color), name=category, hoverinfo='text', text=category_df['hovertext']))
        return fig

    is_3_view = all(item in layers for item in ['wells', 'grace', 'rainfall'])
    is_2_view = all(item in layers for item in ['wells', 'grace']) and not is_3_view
    fig1, fig2, fig3 = go.Figure(), go.Figure(), go.Figure()
    
    # Common layout update for all figures
    def final_layout(fig):
        return fig.update_layout(map_style="carto-positron", map_center=map_center, map_zoom=map_zoom, margin={"r":0,"t":40,"l":0,"b":0})

    if is_3_view:
        fig1 = create_aquifer_basemap(fig1); fig1 = add_wells_layer(fig1, points_df, year); fig1.update_layout(title_text=f"Wells ({year})", title_x=0.5, legend_title_text='GWL Category')
        fig2 = create_aquifer_basemap(fig2)
        if month and year and grace_ds is not None:
            try:
                target_date = pd.to_datetime(f"{year}-{month}-15")
                grace_monthly_data = grace_ds.sel(time=target_date, method='nearest')
                if selection_boundary is not None: grace_monthly_data = grace_monthly_data.rio.clip(selection_boundary.geometry, selection_boundary.crs, drop=False)
                grace_df = grace_monthly_data.to_dataframe(name='tws').dropna().reset_index()
                x_res, y_res = abs(grace_ds.rio.resolution()[0]), abs(grace_ds.rio.resolution()[1])
                grace_df['geometry'] = grace_df.apply(lambda row: box(row.x - x_res/2, row.y - y_res/2, row.x + x_res/2, row.y + y_res/2), axis=1)
                grace_gdf = gpd.GeoDataFrame(grace_df, geometry=grace_df['geometry'], crs="EPSG:4326")
                if not grace_gdf.empty: fig2.add_trace(go.Choroplethmap(geojson=grace_gdf.geometry.__geo_interface__, locations=grace_gdf.index, z=grace_gdf['tws'], colorscale='Viridis', colorbar_title_text="TWS (cm)", marker_line_width=0, hoverinfo='z'))
            except Exception as e: print(f"Error processing GRACE data: {e}")
        fig2.update_layout(title_text=f"GRACE TWS ({year}-{month:02d})", title_x=0.5)
        fig3 = create_aquifer_basemap(fig3)
        if year in rainfall_files_dict:
            rainfall_data_var = load_prepare_rainfall_year(year, rainfall_files_dict)
            if rainfall_data_var is not None:
                monthly_avg = rainfall_data_var.sel(time=rainfall_data_var.time.dt.month == month).mean(dim='time')
                if selection_boundary is not None: monthly_avg = monthly_avg.rio.clip(selection_boundary.geometry, selection_boundary.crs, drop=False)
                monthly_df = monthly_avg.to_dataframe(name='rainfall').dropna().reset_index()
                x_res, y_res = abs(monthly_avg.rio.resolution()[0]), abs(monthly_avg.rio.resolution()[1])
                monthly_df['geometry'] = monthly_df.apply(lambda r: box(r.x-x_res/2, r.y-y_res/2, r.x+x_res/2, r.y+y_res/2), axis=1)
                monthly_gdf = gpd.GeoDataFrame(monthly_df, geometry=monthly_df.geometry, crs="EPSG:4326")
                if not monthly_gdf.empty: fig3.add_trace(go.Choroplethmap(geojson=monthly_gdf.geometry.__geo_interface__, locations=monthly_gdf.index, z=monthly_gdf['rainfall'], colorscale='Blues', colorbar_title_text="Avg Rainfall (mm/day)", marker_line_width=0, hoverinfo='z'))
        fig3.update_layout(title_text=f"Avg Monthly Rainfall ({year}-{month:02d})", title_x=0.5)
        return style_3_view, final_layout(fig1), style_3_view, final_layout(fig2), style_3_view, final_layout(fig3)
    
    # Logic for 1 and 2 map views is now consolidated
    fig1 = create_aquifer_basemap(fig1)
    if 'wells' in layers: fig1 = add_wells_layer(fig1, points_df, year)
    if not is_2_view and 'grace' in layers:
        # add grace to fig1
        pass
    if not is_2_view and 'rainfall' in layers:
        # add rainfall to fig1
        pass
    fig1.update_layout(title_text="Primary Map View", title_x=0.5)
    
    if is_2_view:
        # add grace to fig2
        fig2 = create_aquifer_basemap(fig2)
        fig2.update_layout(title_text=f"GRACE TWS ({year}-{month:02d})", title_x=0.5)
        return style_2_view, final_layout(fig1), style_2_view, final_layout(fig2), style_hidden, fig3

    return style_1_view, final_layout(fig1), style_hidden, fig2, style_hidden, fig3


@app.callback(
    [Output('trend-plot', 'figure'), Output('storage-plot', 'figure'), Output('trend-data-store', 'data'),
     Output('storage-analysis-store', 'data'), Output('advanced-analysis-trends-store', 'data')],
    [Input('state-dropdown', 'value'), Input('district-dropdown', 'value')],
    prevent_initial_call=True
)
def run_primary_analyses(state, district):
    ph_fig = go.Figure(layout={'template': 'plotly_white', 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})
    selection_boundary, aoi_name = None, "India"
    if district and district != 'ALL':
        selection_boundary = boundaries_gdf[boundaries_gdf['district'] == district]
        aoi_name = f"{district}, {state}"
    elif state and state != 'ALL':
        selection_boundary = boundaries_gdf[boundaries_gdf['state'] == state]
        aoi_name = state
    if selection_boundary is None or selection_boundary.empty:
        return ph_fig, ph_fig, None, None, None
        
    clipped_wells = gpd.clip(full_points_gdf, selection_boundary)
    wells_monthly = clipped_wells.groupby(pd.Grouper(key='date', freq='MS'))['gwl'].mean().to_frame(name='avg_gwl')
    clipped_grace = grace_ds.rio.clip(selection_boundary.geometry, selection_boundary.crs, drop=True)
    weights = np.cos(np.deg2rad(clipped_grace.y))
    grace_monthly = clipped_grace.weighted(weights).mean(dim=['x', 'y']).to_dataframe(name='avg_tws')
    rainfall_data = []
    for year, nc_file_path in sorted(rainfall_files_dict.items()):
        rainfall_var = load_prepare_rainfall_year(year, rainfall_files_dict)
        if rainfall_var is not None:
            clipped_rain = rainfall_var.rio.clip(selection_boundary.geometry, selection_boundary.crs, drop=True)
            weights_rain = np.cos(np.deg2rad(clipped_rain.y))
            monthly_avg = clipped_rain.resample(time='MS').mean().weighted(weights_rain).mean(dim=['x','y'])
            rainfall_data.append(monthly_avg.to_dataframe(name='avg_rainfall'))
    rainfall_monthly = pd.concat(rainfall_data) if rainfall_data else pd.DataFrame()
    trend_df = pd.concat([wells_monthly, grace_monthly, rainfall_monthly], axis=1).reset_index().rename(columns={'index':'time'})
    
    trend_fig = make_subplots(specs=[[{"secondary_y": True}]])
    trend_fig.add_trace(go.Scatter(x=trend_df['time'], y=trend_df['avg_gwl'], name='Avg. GWL (m)', line=dict(color='blue')), secondary_y=False)
    trend_fig.add_trace(go.Scatter(x=trend_df['time'], y=trend_df['avg_tws'], name='GRACE TWS (cm)', line=dict(color='green')), secondary_y=True)
    trend_fig.update_layout(title_text=f'Long-Term Trends for {aoi_name}', template='plotly_white', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    trend_fig.update_yaxes(title_text="Avg. GWL (m)", autorange="reversed", secondary_y=False)
    trend_fig.update_yaxes(title_text="GRACE TWS (cm)", secondary_y=True)
    trend_store = trend_df.to_dict('records')
    advanced_trends_store = trend_df.set_index('time').interpolate(method='linear').dropna().reset_index().to_dict('records')
    
    storage_fig, storage_store = ph_fig, None
    try:
        clipped_aquifers = gpd.clip(aquifers_gdf, selection_boundary)
        centroid = clipped_aquifers.unary_union.centroid
        utm_crs = f'EPSG:326{int((centroid.x + 180) / 6) + 1}'
        aquifers_reprojected = clipped_aquifers.to_crs(utm_crs)
        aquifers_reprojected['area_m2'] = aquifers_reprojected.geometry.area
        weighted_avg_sy = (aquifers_reprojected['area_m2'] * aquifers_reprojected['specific_yield']).sum() / aquifers_reprojected['area_m2'].sum()
        total_aquifer_area_m2 = aquifers_reprojected['area_m2'].sum()
        pre_monsoon_avg = clipped_wells[clipped_wells['month'].isin([4, 5, 6])].groupby('year')['gwl'].mean().rename('pre_gwl')
        post_monsoon_avg = clipped_wells[clipped_wells['month'].isin([9, 10, 11])].groupby('year')['gwl'].mean().rename('post_gwl')
        fluctuation_df = pd.concat([pre_monsoon_avg, post_monsoon_avg], axis=1).dropna()
        fluctuation_df['storage_change_mcm'] = (fluctuation_df['pre_gwl'] - fluctuation_df['post_gwl']) * total_aquifer_area_m2 * weighted_avg_sy / 1_000_000
        storage_fig = go.Figure(data=go.Bar(x=fluctuation_df.index, y=fluctuation_df['storage_change_mcm'], marker_color=['green' if x >= 0 else 'red' for x in fluctuation_df['storage_change_mcm']]))
        storage_fig.update_layout(title_text=f'Annual Storage Fluctuation for {aoi_name}', yaxis_title='Storage Change (MCM)', template='plotly_white')
        storage_store = {'summary_text': f"Annual Storage Analysis for {aoi_name}:\n- Avg. Annual Storage Change: {fluctuation_df['storage_change_mcm'].mean():.2f} MCM\n- Area-Weighted Sy: {weighted_avg_sy:.4f}"}
    except Exception as e:
        print(f"Storage calculation error: {e}")
        
    return trend_fig, storage_fig, trend_store, storage_store, advanced_trends_store

@app.callback(Output('advanced-analysis-container', 'style'), Input('show-advanced-checklist', 'value'))
def toggle_advanced_analysis_container(checklist_value):
    if 'SHOW' in checklist_value:
        return {'display': 'block', 'border': '1px solid #ccc', 'padding': '15px', 'marginTop': '20px', 'borderRadius': '5px'}
    return {'display': 'none'}

@app.callback(Output('advanced-analysis-output', 'children'), Input('advanced-analysis-button', 'n_clicks'),
    [State('advanced-analysis-dropdown', 'value'), State('state-dropdown', 'value'), State('district-dropdown', 'value'), State('advanced-analysis-trends-store', 'data')])
def run_advanced_analysis(n_clicks, analysis_type, state, district, trends_data):
    if n_clicks == 0: return html.P("Please select a location and an analysis module, then click 'Run'.")
    if not trends_data: return html.Div([html.H4("Error: Prerequisite Data Missing"), html.P("Please select a valid State/District on the main dashboard first.")], style={'color': 'red'})
    aoi_name = f"{district}, {state}" if district and district != 'ALL' else state
    trends_df = pd.DataFrame(trends_data).set_index('time'); trends_df.index = pd.to_datetime(trends_df.index)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    trends_df['gwl_norm'] = scaler.fit_transform(-trends_df[['avg_gwl']]); trends_df['grace_norm'] = scaler.fit_transform(trends_df[['avg_tws']]); trends_df['rain_norm'] = scaler.fit_transform(trends_df[['avg_rainfall']])
    
    if analysis_type == 'SASS':
        weights = {'gwl': 0.5, 'grace': 0.3, 'rain': 0.2}; sass_series = (trends_df['gwl_norm'] * weights['gwl'] + trends_df['grace_norm'] * weights['grace'] + trends_df['rain_norm'] * weights['rain'])
        latest_sass = sass_series.iloc[-1]
        category = "Safe" if latest_sass > 0.5 else "Moderate" if latest_sass > 0 else "Critical" if latest_sass > -0.5 else "Overexploited"
        return html.Div([html.H4(f"SASS for {aoi_name}"), html.H3(f"Latest SASS Score: {latest_sass:.3f} ({category})"), dcc.Graph(figure=go.Figure(data=go.Scatter(x=sass_series.index, y=sass_series, mode='lines')).update_layout(title='SASS Time Series', yaxis_title='Stress Score (-1 to 1)'))])

    elif analysis_type == 'GRACE_DIVERGENCE':
        divergence_series = trends_df['grace_norm'] - trends_df['gwl_norm']; latest_divergence = divergence_series.iloc[-1]
        interp = "Positive Divergence: GRACE shows more water than wells." if latest_divergence > 0.3 else "Negative Divergence: Wells are more stable than GRACE." if latest_divergence < -0.3 else "Low Divergence: GRACE and ground observations agree."
        return html.Div([html.H4(f"GRACE vs. Ground Reality for {aoi_name}"), html.H3(f"Latest Divergence: {latest_divergence:.3f}"), html.P(interp), dcc.Graph(figure=go.Figure(data=go.Scatter(x=divergence_series.index, y=divergence_series, mode='lines')).update_layout(title='Divergence Time Series'))])

    elif analysis_type == 'ASI':
        selection_boundary = boundaries_gdf[boundaries_gdf['district'] == district] if district and district != 'ALL' else boundaries_gdf[boundaries_gdf['state'] == state]
        clipped_aquifers = gpd.clip(aquifers_gdf, selection_boundary); reprojected = clipped_aquifers.to_crs("EPSG:6933"); reprojected['area'] = reprojected.geometry.area
        result_df = reprojected.groupby('principal_').agg(total_area=('area', 'sum'), avg_sy=('specific_yield', 'mean'), avg_asi_score=('asi_score', 'mean')).reset_index().sort_values('total_area', ascending=False)
        return html.Div([html.H4(f"Aquifer Suitability Index (ASI) for {aoi_name}"), html.Table([html.Thead(html.Tr([html.Th("Aquifer"), html.Th("ASI Score"), html.Th("Avg. Sy")])), html.Tbody([html.Tr([html.Td(row['principal_']), html.Td(f"{row['avg_asi_score']:.2f}/5.0"), html.Td(f"{row['avg_sy']:.3f}")]) for i, row in result_df.iterrows()])])])

    elif analysis_type == 'FORECAST':
        gwl_series = trends_df['avg_gwl'].asfreq('MS')
        model = sm.tsa.ExponentialSmoothing(gwl_series, trend='add', seasonal='add', seasonal_periods=12).fit()
        forecast = model.get_forecast(steps=12).summary_frame(alpha=0.2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gwl_series.index, y=gwl_series, name='Historical', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast['mean'], name='Forecast', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast['mean_ci_upper'], fill='tonexty', mode='none', fillcolor='rgba(255,0,0,0.2)'))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast['mean_ci_lower'], fill='tonexty', mode='none', fillcolor='rgba(255,0,0,0.2)'))
        fig.update_layout(title=f'12-Month GWL Forecast for {aoi_name}', yaxis_autorange='reversed')
        alert = "CRITICAL: Declining levels projected." if forecast['mean'].iloc[-1] > gwl_series.iloc[-1] else "STABLE/IMPROVING"
        return html.Div([html.H3(f"Forecast Alert: {alert}", style={'color':'red' if 'CRITICAL' in alert else 'green'}), dcc.Graph(figure=fig)])

    elif analysis_type == 'RECHARGE':
        sass_series = (trends_df['gwl_norm'] * 0.5 + trends_df['grace_norm'] * 0.3 + trends_df['rain_norm'] * 0.2)
        stress_level = 'Low' if sass_series.iloc[-1] > 0 else 'Moderate' if sass_series.iloc[-1] > -0.5 else 'High'
        selection_boundary = boundaries_gdf[boundaries_gdf['district'] == district] if district and district != 'ALL' else boundaries_gdf[boundaries_gdf['state'] == state]
        dominant_aquifer = gpd.clip(aquifers_gdf, selection_boundary).iloc[gpd.clip(aquifers_gdf, selection_boundary).geometry.area.argmax()]['principal_']
        recs = {'Alluvium': 'Check Dams, Percolation Tanks', 'Sandstone': 'Injection Wells', 'Basalt': 'Gully Plugs, Trenches', 'Laterite': 'Contour Bunding', 'Granite': 'Check Dams', 'Gneiss': 'Check Dams', 'Limestone': 'Injection Wells'}
        return html.Div([html.H4(f"Recharge Recommendations for {aoi_name}"), html.P(f"Dominant Aquifer: **{dominant_aquifer}**. Stress Level: **{stress_level}**."), html.H5("Recommended:"), html.P(recs.get(dominant_aquifer, "Standard Check Dams."))])

@app.callback(Output('ai-answer-output', 'children'), Input('ai-ask-button', 'n_clicks'),
    [State('ai-question-input', 'value'), State('state-dropdown', 'value'), State('district-dropdown', 'value'), State('trend-data-store', 'data'), State('storage-analysis-store', 'data')])
def get_holistic_ai_answer(n_clicks, question, state, district, trend_data, storage_data):
    if n_clicks == 0 or not question: return "Ask a question for a consolidated analysis."
    aoi_name = f"{district}, {state}" if district and district != 'ALL' else state
    context = f"## Comprehensive Data Summary for {aoi_name}\n\n### Long-Term Trends:\n"
    if trend_data:
        df = pd.DataFrame(trend_data)
        if 'avg_gwl' in df.columns and not df['avg_gwl'].isnull().all(): context += f"- GWL range: {df['avg_gwl'].min():.2f}m to {df['avg_gwl'].max():.2f}m\n"
        else: context += "- GWL data not available.\n"
        if 'avg_tws' in df.columns and not df['avg_tws'].isnull().all(): context += f"- GRACE TWS range: {df['avg_tws'].min():.2f}cm to {df['avg_tws'].max():.2f}cm\n"
        else: context += "- GRACE data not available.\n"
    else: context += "- Trend data not available. Please select a region.\n"
    context += "\n### Annual Storage:\n"
    context += storage_data.get('summary_text', "- Storage data not available.\n") if storage_data else "- Storage data not available.\n"
    system_prompt = "You are a world-class hydrogeology consultant. Synthesize insights from the provided data summary (long-term trends, annual storage) to answer the user's question. Be data-driven and provide a concluding expert opinion."
    user_prompt = f"Data Summary:\n```\n{context}\n```\n\nQuestion: {question}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    try:
        client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=os.environ.get("HF_TOKEN"))
        completion = client.chat.completions.create(model=HF_MODEL_ID, messages=messages, max_tokens=512, temperature=0.4)
        return completion.choices[0].message.content
    except Exception as e: return f"## AI Error: {e}"

# --- Run App ---
def open_browser(): webbrowser.open_new("http://127.0.0.1:8050/")
if __name__ == '__main__':
    if not boundaries_gdf.empty:
        Timer(1, open_browser).start(); app.run(debug=True, port=8050)
    else: print("\nApplication not started due to fatal error in loading data.")