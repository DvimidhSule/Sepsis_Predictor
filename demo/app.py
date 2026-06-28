"""
Interactive Dash Bedside Monitor Dashboard.
Streams patient data hour-by-hour, visualizes sepsis risk, and shows
SHAP value drivers using the trained server model.
This is a retrospective simulation on recorded ICU data.

Run: python demo/app.py and open http://127.0.0.1:8050
"""
import json
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx

PAT = pd.read_csv('demo/demo_patient.csv')
META = json.load(open('demo/demo_meta.json'))
THR = META['alarm_threshold']
WARN_H = META['first_warning_hour']        # SepsisLabel turns 1 (6h pre clinical onset)
ONSET_H = META['clinical_onset_hour']
ALARM_H = META['first_alarm_hour']
NH = len(PAT)
DISPLAY = ['HR', 'Temp', 'Resp', 'SBP', 'MAP', 'O2Sat']

# normal ranges for colour coding (low_ok, high_ok)
RANGES = {'HR': (60, 100), 'Temp': (36.0, 38.0), 'Resp': (12, 20),
          'SBP': (100, 140), 'MAP': (70, 100), 'O2Sat': (95, 100)}
UNITS = {'HR': 'bpm', 'Temp': '°C', 'Resp': '/min', 'SBP': 'mmHg', 'MAP': 'mmHg', 'O2Sat': '%'}

BG, PANEL, LINE, TXT, MUT = '#0d0f14', '#161a22', '#2a3140', '#e6e9ef', '#8a93a6'
GREEN, AMBER, RED, BLUE = '#46c98b', '#f5b945', '#ef5e6a', '#5b9dff'

app = Dash(__name__)
app.title = 'Sepsis Early-Warning Monitor'


def vital_color(v, val):
    if pd.isna(val):
        return MUT
    lo, hi = RANGES[v]
    return RED if (val < lo or val > hi) else GREEN


def card(label, value, unit, color):
    disp = '—' if pd.isna(value) else (f'{value:.1f}' if value % 1 else f'{int(value)}')
    return html.Div(style={'background': PANEL, 'border': f'1px solid {LINE}',
                           'borderRadius': '10px', 'padding': '12px 14px', 'flex': '1', 'minWidth': '92px'},
                    children=[
                        html.Div(label, style={'color': MUT, 'fontSize': '12px',
                                               'textTransform': 'uppercase', 'letterSpacing': '.05em'}),
                        html.Div([html.Span(disp, style={'fontSize': '26px', 'fontWeight': '700', 'color': color}),
                                  html.Span(' ' + unit, style={'fontSize': '12px', 'color': MUT})])
                    ])


app.layout = html.Div(style={'background': BG, 'color': TXT, 'minHeight': '100vh',
                             'fontFamily': '-apple-system,Segoe UI,Roboto,sans-serif', 'padding': '18px 22px'},
    children=[
        dcc.Interval(id='tick', interval=900, disabled=True),
        dcc.Store(id='hour', data=0),

        # disclaimer + title
        html.Div(style={'background': '#3a2a14', 'border': f'1px solid {AMBER}', 'borderRadius': '8px',
                        'padding': '6px 12px', 'fontSize': '12px', 'color': AMBER, 'marginBottom': '12px'},
                 children='⚠ RETROSPECTIVE SIMULATION on recorded ICU data — not a live or certified medical device.'),
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'baseline'},
                 children=[
                     html.H2('ICU Sepsis Early-Warning Monitor', style={'margin': '0'}),
                     html.Div(f"Patient #{META['patient_id']}  ·  nurse-station model (40-feature, ROC-AUC {META['server_auc']:.3f})",
                              style={'color': MUT, 'fontSize': '13px'})
                 ]),

        # alarm banner
        html.Div(id='alarm-banner', style={'margin': '14px 0'}),

        # vital cards
        html.Div(id='vitals', style={'display': 'flex', 'gap': '10px', 'marginBottom': '14px'}),

        # drivers
        html.Div(id='drivers', style={'marginBottom': '14px'}),

        # charts
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '14px'},
                 children=[dcc.Graph(id='risk-chart', config={'displayModeBar': False}),
                           dcc.Graph(id='vitals-chart', config={'displayModeBar': False})]),

        # controls
        html.Div(style={'display': 'flex', 'gap': '14px', 'alignItems': 'center', 'marginTop': '10px'},
                 children=[
                     html.Button('▶ Play', id='play', n_clicks=0,
                                 style={'background': PANEL, 'color': TXT, 'border': f'1px solid {LINE}',
                                        'borderRadius': '8px', 'padding': '8px 18px', 'cursor': 'pointer', 'fontWeight': '600'}),
                     html.Div(style={'flex': '1'},
                              children=dcc.Slider(id='slider', min=0, max=NH - 1, step=1, value=0,
                                                  marks={0: 'h0', WARN_H: '6h-pre-onset', ONSET_H: 'onset', NH - 1: f'h{NH-1}'},
                                                  tooltip={'placement': 'bottom'}))
                 ]),
    ])


# play/pause toggles interval
@app.callback(Output('tick', 'disabled'), Output('play', 'children'),
              Input('play', 'n_clicks'), prevent_initial_call=True)
def toggle(n):
    playing = (n % 2) == 1
    return (not playing), ('⏸ Pause' if playing else '▶ Play')


# advance hour on tick or slider
@app.callback(Output('hour', 'data'), Output('slider', 'value'),
              Input('tick', 'n_intervals'), Input('slider', 'value'), State('hour', 'data'))
def advance(_, slider_val, cur):
    if ctx.triggered_id == 'slider':
        return slider_val, slider_val
    nxt = min(cur + 1, NH - 1)
    return nxt, nxt


# render everything for the current hour
@app.callback(Output('alarm-banner', 'children'), Output('vitals', 'children'),
              Output('drivers', 'children'), Output('risk-chart', 'figure'),
              Output('vitals-chart', 'figure'), Input('hour', 'data'))
def render(h):
    row = PAT.iloc[h]
    upto = PAT.iloc[:h + 1]
    alarmed = (upto['risk'] >= THR).any()

    # alarm banner (latched)
    if alarmed:
        lead = max(0, WARN_H - ALARM_H)
        banner = html.Div(style={'background': '#3a1620', 'border': f'2px solid {RED}', 'borderRadius': '10px',
                                 'padding': '14px 18px', 'fontSize': '18px', 'fontWeight': '700', 'color': RED},
                          children=f'🚨 SEPSIS ALERT — risk crossed threshold at hour {ALARM_H} '
                                   f'({lead}h before the 6h-pre-onset mark, {ONSET_H - ALARM_H}h before clinical onset)')
    else:
        banner = html.Div(style={'background': PANEL, 'border': f'1px solid {GREEN}', 'borderRadius': '10px',
                                 'padding': '14px 18px', 'fontSize': '16px', 'fontWeight': '600', 'color': GREEN},
                          children=f'✓ Monitoring — risk {row["risk"]*100:.1f}%  (alarm at {THR*100:.1f}%)')

    vitals = [card(v, row[v], UNITS[v], vital_color(v, row[v])) for v in DISPLAY]

    # SHAP drivers (positive contributors = pushing risk up)
    drv = sorted([(row[f'shap_{v}'], v) for v in DISPLAY], reverse=True)
    chips = []
    for val, v in drv:
        if val > 0.02:
            chips.append(html.Span(f'↑ {v}', style={'background': '#3a1620', 'color': RED, 'padding': '4px 12px',
                                                     'borderRadius': '16px', 'marginRight': '8px', 'fontSize': '13px',
                                                     'fontWeight': '600', 'border': f'1px solid {RED}'}))
    drivers = html.Div([html.Span('Risk drivers (SHAP): ', style={'color': MUT, 'fontSize': '13px'})] +
                       (chips if chips else [html.Span('none elevated', style={'color': MUT})]))

    # risk chart
    rf = go.Figure()
    rf.add_trace(go.Scatter(x=upto['hour'], y=upto['risk'] * 100, mode='lines+markers',
                            line=dict(color=RED if alarmed else BLUE, width=2.5), name='risk'))
    rf.add_hline(y=THR * 100, line=dict(color=AMBER, dash='dash'),
                 annotation_text='alarm threshold', annotation_font_color=AMBER)
    rf.add_vline(x=WARN_H, line=dict(color=MUT, dash='dot'), annotation_text='6h-pre-onset', annotation_font_color=MUT)
    rf.add_vline(x=ONSET_H, line=dict(color='#aa5555', dash='dot'), annotation_text='clinical onset', annotation_font_color='#aa5555')
    rf.update_layout(title='Sepsis risk — P(onset ≤ 6h)', template='plotly_dark', paper_bgcolor=BG,
                     plot_bgcolor=PANEL, height=320, margin=dict(l=40, r=20, t=40, b=30),
                     xaxis=dict(range=[0, NH - 1], title='ICU hour'), yaxis=dict(title='risk %', rangemode='tozero'))

    # vitals chart
    vf = go.Figure()
    for v, col in [('HR', RED), ('Resp', AMBER), ('SBP', BLUE), ('MAP', GREEN)]:
        vf.add_trace(go.Scatter(x=upto['hour'], y=upto[v], mode='lines', name=v, line=dict(width=2)))
    vf.add_vline(x=ONSET_H, line=dict(color='#aa5555', dash='dot'))
    vf.update_layout(title='Vitals', template='plotly_dark', paper_bgcolor=BG, plot_bgcolor=PANEL,
                     height=320, margin=dict(l=40, r=20, t=40, b=30),
                     xaxis=dict(range=[0, NH - 1], title='ICU hour'), legend=dict(orientation='h', y=1.12))
    return banner, vitals, drivers, rf, vf


if __name__ == '__main__':
    print(f'Sepsis monitor: patient {META["patient_id"]}, {NH}h, alarm at h{ALARM_H}, onset h{ONSET_H}')
    app.run(debug=False, port=8050)
