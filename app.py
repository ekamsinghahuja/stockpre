import dash
# from jinja2 import escape
from dash import dcc, html
from dash.dependencies import Input, Output, State
from datetime import datetime as dt
from datetime import datetime as dt, timedelta
import yfinance as yf
import pandas as pd
import plotly.express as px
from model import model_evaluation_callback

app = dash.Dash(__name__)
app = dash.Dash(__name__, external_stylesheets=['assets/styles.css'])

def get_ema_plot(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df,
                     x='Date',
                     y='EWA_20',
                     title="Exponential Moving Average vs Date")
    fig.update_traces(mode='lines')  
    return fig


def get_stock_price_fig(df):
    fig = px.line(df,
                  x='Date',
                  y=['Open', 'Close'],
                  title="Closing and Opening Price vs Date")
    return fig


submit_button = html.Button('Submit', id='submit-button', n_clicks=0)
app.layout = html.Div([
    # Navigation Bar
    html.Div([
        html.P("Welcome to the Stock Dash App!", id="start"),
        html.Div([
            # Stock code input
            dcc.Input(id='stock-code', type='text', placeholder='Enter stock code...'),
            submit_button
        ]),
        html.Div([
            # Date range picker input
            dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=dt(2020, 1, 1),
                max_date_allowed=dt.today(),
                initial_visible_month=dt.today(),
                start_date_placeholder_text="Start Date",
                end_date_placeholder_text="End Date"
            )
        ]),
        html.Div([
            html.Button('Stock Price', id='stock-price-button'),
            html.Button('Indicators', id='indicators-button'),
            dcc.Input(id='forecast-days', type='number', placeholder='Enter forecast days...', min=2, max=30),
            html.Button('Forecast', id='forecast-button')
        ])
    ], className="nav"),

    # Content Section
    html.Div([
        html.Div([
            # Company Name
            html.H1(id='company-name')
        ], className="header"),
        html.Div([
            # Description
            html.P(id='description', className="decription_ticker")
        ]),
        html.Div([
            dcc.Graph(id='stock-price-plot',figure={},style={'display': 'none'})
        ], id="stock-price-content"),
        html.Div([
            # Indicator plot
            dcc.Graph(id='indicator-plot',figure={},style={'display': 'none'})
        ], id="indicator-content"),
        html.Div([
            # Forecast plot
            dcc.Graph(id='forecast-plot',figure={},style={'display': 'none'})
        ], id="forecast-content")
    ], className="content")
])

# Callbacks to update components based on user input

@app.callback(
    [Output('company-name', 'children'),
     Output('description', 'children')],
    [Input('submit-button', 'n_clicks')],
    [State('stock-code', 'value')]
)
def update_company_info(n_clicks, stock_code):
    if n_clicks > 0 and stock_code:
        try:
            ticker = yf.Ticker(str(stock_code))
            inf = ticker.info
            df = pd.DataFrame().from_dict(inf, orient="index").T
            company_name = df['longName'][0]
            description = df['longBusinessSummary'][0]
            return company_name, description
        except Exception as e:
            return f"Error fetching company info: {stock_code}", "Error fetching company info: {stock_code}"
    else:
        return "", ""
    

# Callback to update stock price plot based on user input
@app.callback(
    Output('stock-price-plot', 'figure'),
    [Input('stock-price-button', 'n_clicks')],
    [State('stock-code', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date')]
)
def update_stock_price_plot(n_clicks, stock_code, start_date, end_date):
    if n_clicks is not None and n_clicks > 0 and stock_code and start_date and end_date:
        try:
            df = yf.download(stock_code, start=start_date, end=end_date)
            df.reset_index(inplace=True)
            fig = get_stock_price_fig(df)
            print(start_date)
            print(end_date)
            return fig
        except Exception as e:
            print(e) 
            return {}
    else:
        return {}
    
@app.callback(
    Output('indicator-plot', 'figure'),
    [Input('indicators-button', 'n_clicks')],
    [State('stock-code', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date')]
)
def update_indicator_plot(n_clicks, stock_code, start_date, end_date):
    if n_clicks is not None and n_clicks > 0 and stock_code and start_date and end_date:
        try:
            df = yf.download(stock_code, start=start_date, end=end_date)
            df.reset_index(inplace=True)
            fig = get_ema_plot(df)
            return fig
        except Exception as e:
            print(e)  # Print the error for debugging
            return {}
    else:
        return {}

@app.callback(
    Output('forecast-plot', 'figure'),
    [Input('forecast-button', 'n_clicks')],
    [State('stock-code', 'value'),
     State('forecast-days', 'value')]
)
def update_forecast_plot(n_clicks, stock_code, forecast_days):
    if n_clicks is not None and n_clicks > 0 and stock_code and forecast_days:
        try:
           
            predicted_data = model_evaluation_callback(stock_code, forecast_days)
            tomorrow = (dt.today() + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            dates = pd.date_range(start=tomorrow, periods=forecast_days)
            df = pd.DataFrame({'Date': dates, 'Predicted': predicted_data})
            fig = px.line(df, x='Date', y='Predicted', title=f'Forecasted Prices for {forecast_days} Days')
            fig.update_traces(mode='lines+markers')
            return fig
        except Exception as e:
            print(e)  
            return {}
    else:
        return {}
    
@app.callback(
    Output('stock-price-plot', 'style'),
    [Input('stock-price-button', 'n_clicks')],
    [State('stock-code', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date')]
)
def show_stock_price_plot(n_clicks, stock_code, start_date, end_date):
    if n_clicks is not None and n_clicks > 0 and stock_code and start_date and end_date:
        return {'display': 'block'} 
    else:
        return {'display': 'none'}  
@app.callback(
    Output('indicator-plot', 'style'),
    [Input('indicators-button', 'n_clicks')],
    [State('stock-code', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date')]
)
def show_indicator_plot(n_clicks, stock_code, start_date, end_date):
    if n_clicks is not None and n_clicks > 0 and stock_code and start_date and end_date:
        return {'display': 'block'} 
    else:
        return {'display': 'none'}  

@app.callback(
    Output('forecast-plot', 'style'),
    [Input('forecast-button', 'n_clicks')],
    [State('stock-code', 'value'),
     State('forecast-days', 'value')]
)
def show_forecast_plot(n_clicks, stock_code, forecast_days):
    if n_clicks is not None and n_clicks > 0 and stock_code and forecast_days:
        return {'display': 'block'} 
    else:
        return {'display': 'none'}


if __name__ == '__main__':
    app.run_server(debug=True)
