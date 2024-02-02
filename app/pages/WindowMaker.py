""" Restructured frontend for proteogyver app"""
import os
import dash_bootstrap_components as dbc
from dash import html, callback, clientside_callback, dcc, register_page, no_update, ctx,ALL, dash_table
from components.parsing import parse_parameters
from dash.dependencies import Input, Output, State
import logging
from element_styles import CONTENT_STYLE
from components import ui_components as ui
from plotly import express as px
import pandas as pd
import numpy as np
import multiprocessing
import random
from uuid import uuid4
from plotly import graph_objects as go
from components import windowmaker_utils as wu
from datetime import datetime

random.seed(123)

slim_testing_factor = False#20
register_page(__name__, path='/windowmaker')
logger = logging.getLogger(__name__)
logger.warning(f'{__name__} loading')

offered_equations = [
    (equation, 'PREMADE-'+ wu.get_eq_id_str(equation)) for equation in [
        '0.75 * x + 10',
        '0.75 * x + 20',
        '0.5 * x - 20',
        '0.5 * x - 10'
    ]
]

parameters = parse_parameters('parameters.json')
db_file: str = os.path.join(*parameters['Data paths']['Database file'])
notification_music = '/assets/sound/bmm_shortest.mp3'
wmstyle = {
    #'margin-top': ,
    'margin-right': '2%',
    'paddingTop': 72,
    'paddingBottom': '1%',
    #'padding': '5% 1%',
    'width': '100%',
  #  'display': 'inline-block',
    'overflow': 'auto',
}
pasef_method_table_columns = [
    'MS Type','Cycle Id','1/K0 Begin [Vs/cm2]','1/K0 End [Vs/cm2]','Start Mass [m/z]','End Mass [m/z]','CE [eV]'
]
layout = ui.windowmaker_interface(wmstyle,offered_equations)

@callback(
    Output('windowmaker-ch-plot-area', 'children'),
    Input('windowmaker-filtered-data-store','data'),
    prevent_initial_call = True
)
def generate_chplot(data):
    if data is None:
        return no_update
    elif data['mgf']=='Error in MGF handling':
        return no_update
    histograms = False
    if histograms:
        figwidth = 800
        figheight = 200

        mgfdata = pd.read_json(data['mgf'], orient='split')
        fig = px.histogram(mgfdata, x='Mz', color='Charge', nbins=50, opacity=0.75, width=figwidth, height=figheight)
        fig.update_layout(title='Mz values per charge',
                        xaxis_title='mz',
                        yaxis_title='Frequency')
        fig2 = px.histogram(mgfdata, x='Mobility', color='Charge', nbins=50, opacity=0.75, width=figwidth, height=figheight)
        fig2.update_layout(title='Mobility values per charge',
                        xaxis_title='mobility',
                        yaxis_title='Frequency')
        retkids = [
            dcc.Graph(id='windowmaker-ch-mz-plot',figure=fig),
            dcc.Graph(id='windowmaker-ch-mob-plot',figure=fig2)
        ]
    else:

        charge_graphs = []
        sorted_charges = []

        ch_cols = len(data['charge_states'].keys())
        if ch_cols > 1:
            figwidth = 800
            figheight = 400
            ch_width = figwidth/2
            ch_height = figheight/2
            sorted_charges = sorted(list(data['charge_states'].keys()))
            for ch in sorted_charges:
                ch_dfs = data['charge_states'][ch]
                pdf = pd.read_json(ch_dfs[1],orient='split')
                ch_fig = px.imshow(pdf,origin='lower',aspect='equal',width=ch_width, height=ch_height)
                #ch_fig.update_traces(name='mgfplot')
                ch_fig.update_layout(coloraxis_showscale=False, margin={'t':0,'l':0,'b':0,'r':0})
                #ch_fig.update_layout(autosize=True)
                ch_graph = dcc.Graph(id=f'windowmaker-pre-plot-{ch}',figure=ch_fig, style={'padding': '0px 0px 0px 0px','float': 'left','display': 'flex'})
                charge_graphs.append(dbc.Col([html.H4(f'Charge {ch}'), ch_graph], width=6,style={'float': 'left','display': 'block'}))
        if len(charge_graphs) % 2 != 0:
            charge_graphs.append('')
        retkids = [
            dbc.Row([
                charge_graphs[i], charge_graphs[i+1]
                ],style={'float': 'left','display': 'block'}
            ) for i in range(0,len(charge_graphs),2)]
    return retkids

@callback(
    Output('windowmaker-pre-plot-area','children'),
    Input('windowmaker-filtered-data-store','data'),
    prevent_initial_call = True
)
def generate_initial_plot(data):

    figwidth = 800
    figheight = 400

    if data is None:
        return no_update
    elif data['mgf']=='Error in MGF handling':
        return no_update
    pdf = pd.read_json(data['plot'],orient='split')
    # TODO: figure size from parameters
    fig = px.imshow(pdf,origin='lower',aspect='equal', width=figwidth, height=figheight)
    fig.update_traces(name='mgfplot')
    fig.update_layout(coloraxis_showscale=False, margin={'t':0,'l':0,'b':0,'r':0})
    fig.update_yaxes(automargin=True)#['left+top+right+bottom'] 
    fig.update_xaxes(automargin=True)
    #fig.update_layout(autosize=True)
    graph = dcc.Graph(id='windowmaker-pre-plot',figure=fig)
    return graph

@callback(
    Output('windowmaker-pre-plot','figure'),
    Input('windowmaker-equations-list-group','children'),
    State('windowmaker-pre-plot','figure')
)
def insert_eqs_to_figure(eq_list, current_figure):
    nfig = {}
    keep = {'mgfplot'}
    for hdiv in eq_list:
        equation = hdiv['props']['children'][0]['props']['children']
        keep.add(equation)
    for k, v in current_figure.items():
        if k == 'data':
            ndata = []
            for figdic in v:
                if figdic['name'] in keep:
                    ndata.append(figdic)
            nfig['data'] = ndata
        else:
            nfig[k] = v
    current_figure = nfig
    fig = go.Figure(current_figure)
    for hdiv in eq_list:
        equation = hdiv['props']['children'][0]['props']['children']
        x_axis_vals = current_figure['data'][0]['x']
        y_axis_vals = current_figure['data'][0]['y']
        line_points = [[],[]]
        for i, y in enumerate(wu.get_line_y(equation, range(0, len(x_axis_vals)), (0, len(y_axis_vals)))):
            if y is None:
                continue
            ystr = y_axis_vals[y]
            xstr = x_axis_vals[i]
            line_points[0].append(xstr)
            line_points[1].append(ystr)
        if len(line_points[0])>0:
            fig.add_scatter(x=line_points[0], y=line_points[1], mode='lines', line=dict(color='black', width=2),
                showlegend = False, name=equation)
    return fig

def get_eq_group_div(equation, eqname):
    abvals: list = ['Above','Below']
    ab_guess: str = abvals[0]
    if 'x-' in equation.replace(' ',''):
        ab_guess = abvals[1]
    return html.Div(
        [
            dbc.Button(equation, id={'type': 'EQBUTTON', 'name':eqname}, style={'padding': '5px 5px 5px 5px'}),
            html.P('Exclude: ', style={'padding': '5px 5px 5px 5px','align-items': 'center'}),
            dcc.RadioItems(abvals, ab_guess, id={'type': 'eq-radioitems', 'name': eqname}, style={'padding': '5px 5px 5px 5px'})
        ], style = {'width': '100%', 'display': 'inline-flex','padding': '5px 5px 5px 5px'}, id = eqname
    )

@callback(
    Output('windowmaker-equations-list-group', 'children', allow_duplicate=True),
    Input('windowmaker-add-line-button', 'n_clicks'),
    Input({'type': 'PREDEFEQBUTTON', 'name':ALL}, 'n_clicks'),
    State('windowmaker-line-equation-input', 'value'),
    State('windowmaker-equations-list-group', 'children'),
    prevent_initial_call=True
)
def modify_eq_group(_, eq_clicks, equation, current_contents, remove='no'):
    if isinstance(ctx.triggered_id, dict):
        for (button_equation, eqname) in offered_equations:
            if eqname == ctx.triggered_id['name']:
                equation = button_equation
                break
    if equation is None:
        return no_update
    if equation == '':
        return no_update
    eq_area = []
    if remove == 'no':
        equation = equation.lower().split('=')[-1].strip()
        eqname = wu.get_eq_id_str(equation)
        eq_area.append(get_eq_group_div(equation, eqname))
    if current_contents:
        if len(current_contents) > 0:
            for div_dict in current_contents:
                equation = div_dict['props']['children'][0]['props']['children']
                eqname = div_dict['props']['children'][0]['props']['id']['name']
                if remove == eqname:
                    continue
                else:
                    eq_area.append(get_eq_group_div(equation, eqname))
    return eq_area

@callback(
    Output('windowmaker-equations-list-group','children', allow_duplicate=True),
    Output('windowmaker-prev-clicks-data-store', 'data'),
    Input({'type': 'EQBUTTON', 'name': ALL}, 'n_clicks'),
    State('windowmaker-prev-clicks-data-store', 'data'),
    State('windowmaker-equations-list-group','children'),
    prevent_initial_call=True
)
def delete_equation(clicks, clickdata, current_contents):
    noclicks = True
    for c in clicks:
        if c:
            if c > 0:
                noclicks = False
    if not noclicks:
        if ctx.triggered_id['name'] in clickdata:
            if clickdata[ctx.triggered_id['name']] == clicks:
                noclicks = True
    clickdata[ctx.triggered_id['name']] = clicks
    if noclicks:
        return (no_update, clickdata)
    else:
        return (
            modify_eq_group(
                clicks,
                None,
                'dummyeq',
                current_contents,
                remove=ctx.triggered_id['name']
            ),
            clickdata
        )

@callback(
    Output({'type': 'windowmaker-filter-div','name': ALL}, 'hidden'),
    Input('windowmaker-filter-col-dropdown','value'),
    State('windowmaker-filter-col-dropdown','options'),
    #State({'type': 'windowmaker-filter-div','name': ALL},'id')
)
def set_visible_filter_col(leave_visible, options):
    retlist = []
    for k_id in options:
        retlist.append(not (k_id == leave_visible))
    return retlist

@callback(
    Output('enabled-filters-list','children'),
    Input({'type': 'windowmaker-filter-checklist', 'name':ALL}, 'value'),
    State('windowmaker-filter-columns','data')
)
def add_enabled_filters_to_list(all_filter_values, filcol_names):
    to_add = []
    for i, fv in enumerate(all_filter_values):
        if len(fv) > 0:
            to_add.append(filcol_names[i])
    return [html.Li(filcol) for filcol in to_add]

# NExt put all the filter names into the filter list, and move equations to the same level. 

@callback(
    Output('windowmaker-filtered-data-store','data'),
    State('windowmaker-filter-columns','data'),
    Input('windowmaker-full-data-store','data'),
    Input({'type': 'windowmaker-filter-checklist', 'name':ALL}, 'value'),
    prevent_initial_call=True
)
def filter_data(column_names_for_filters, data, chosen_filters):
    mgf_df = pd.read_json(data['mgf'], orient='split')
    for i, colname in enumerate(column_names_for_filters):
        if len(chosen_filters[i]) > 0:
            wu.filter_col(mgf_df, colname, chosen_filters[i], inplace=True)
    return {'mgf': mgf_df.to_json(orient='split'), 'plot': wu.make_pdata(mgf_df).to_json(orient='split'), 'charge_states': wu.do_charges(mgf_df)}
    
@callback(
    Output('windowmaker-filter-col','children'),
    Output('windowmaker-filter-columns','data'),
    Input('windowmaker-full-data-store','data'),
    prevent_initial_call = True
)
def generate_filter_groups(full_data):
    mgfdata = pd.read_json(full_data['mgf'], orient='split')
    dropdown, checklists, checklist_target_cols = ui.generate_filter_group(mgfdata, wu.get_potential_filcols(mgfdata))

    fcol = [dropdown] + checklists
    return fcol, checklist_target_cols

@callback(
    Output('windowmaker-post-plot-area','children'),
    Input('windowmaker-calculate-windows-button','n_clicks'),
    State('windowmaker-filtered-data-store','data'),
    State({'type': 'EQBUTTON', 'name': ALL}, 'children'),
    State({'type': 'eq-radioitems', 'name': ALL}, 'value'),
    State('windowmaker-mz-input-min', 'value'),
    State('windowmaker-mz-input-max', 'value'),
    State('windowmaker-mob-input-min', 'value'),
    State('windowmaker-mob-input-max', 'value'),
    State('windowmaker-play-notification-sound-when-done','value'),
    prevent_initial_call=True
)
def test(n_clicks, full_data, eqs, eq_criteria, mzmin, mzmax, mobmin, mobmax, notify):
    if n_clicks is None: 
        return no_update
    if n_clicks == 0:
        return no_update
    mgfdata = pd.read_json(full_data['mgf'], orient='split')
    mgfdata = wu.filter_mob_and_mz(mgfdata, [mzmin, mzmax], [mobmin, mobmax])
    pdata = wu.make_pdata(mgfdata)
    
    if len(eqs) > 0:
        equation_to_criteria = {eq: eq_criteria[i] for i, eq in enumerate(eqs)}
        wu.remove_from_matrix(pdata, equation_to_criteria)
        wu.remove_from_long(mgfdata, equation_to_criteria, pdata.columns, pdata.index)

    # TODO: num_windows from parameters
    num_windows = 12
    mgfdata = mgfdata.sort_values(by='Peptide mass')
    target_per_window = mgfdata.shape[0]/(num_windows*2)
    windows = [
        ([],[])
    ]
    for _,row in mgfdata.iterrows():
        if len(windows[-1][0]) >= target_per_window:
            windows.append(([],[]))
        windows[-1][0].append(round(row['Peptide mass'], 2))
        windows[-1][1].append(round(row['Mobility'],2))
    windows = wu.count_windows(windows)
    windows = wu.pair_windows(windows)
    window_mob_ranges = []
    window_mz_ranges = []
    for i, w in enumerate(windows):
        window_mz_ranges.append(w[0][:3])
        if i == 0:
            window_mob_ranges.append([w[0][3][0], max(0.01, w[0][3][-1])])
        else:
            w1 = max(w[0][3][0], window_mob_ranges[i-1][1]+0.01)
            w2 = max(w[0][3][-1], w1+0.01)
            window_mob_ranges.append([w1,w2])
    best_tree = None
    best_path = None
    prev_adj_perc = -1
    all_adjmobs = []

    rnd = True
    start = datetime.now()
    for i in range(0, 100):
        if best_tree is None:
            if rnd:
                adj_mob_perc = 0.5
                adj_mob = round((window_mob_ranges[0][1]-window_mob_ranges[0][0]) * adj_mob_perc,2)
                rnd_range = list(np.arange(window_mob_ranges[0][0], window_mob_ranges[0][1]+0.01, 0.01))
                todo = random.sample(rnd_range,min(10, len(rnd_range)))
            else:
                adj_mob_perc = 1
                adj_mob = max(0.01, round((window_mob_ranges[0][1]-window_mob_ranges[0][0]) * adj_mob_perc,2))
                todo = np.arange(window_mob_ranges[0][0], window_mob_ranges[0][1]+adj_mob, adj_mob)
        else:
            adj_mob_perc = round(adj_mob_perc/2,2)
            adj_mob = max(0.01, round((window_mob_ranges[0][1]-window_mob_ranges[0][0]) * adj_mob_perc,2))
            if adj_mob == prev_adj_mob:
                break
            if adj_mob < 0.01:
                break
            todo = np.arange(best_path[0].mobility_value - adj_mob, best_path[0].mobility_value + adj_mob, adj_mob)
        if adj_mob_perc == prev_adj_perc:
            break
        all_adjmobs.append(adj_mob)
        todo = sorted(list(set([round(t, 2) for t in todo])))
        prev_adj_mob = adj_mob
        prev_adj_perc = adj_mob_perc
        num_processes = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_processes)
        for start_mob in todo:
            result = wu.process_iteration(start_mob, windows, adj_mob_perc, window_mob_ranges, best_tree, windows)
            if result[0]: 
                best_tree, best_path = result
        pool.close()
        pool.join()
        print('iteration', (datetime.now()-start))
    print('optimizing', (datetime.now()-start))
    wu.optimize_window_position(best_tree, windows)
    print('done', (datetime.now()-start))
    finished_windows = []
    prev_mzhigh = None
    for window_index, (window1, window2) in enumerate(windows):
        mz1_low, mz1_high = window1[:2]
        mz2_low, mz2_high = window2[:2]
        if prev_mzhigh:
            mz1_low, mz2_low = prev_mzhigh
        prev_mzhigh = (mz1_high, mz2_high)
        mob_leaf = best_path[window_index].to_dict()
        coordinates1=tuple(round(x, 2) for x in [mobmin, mob_leaf['mobility_value'], mz1_low, mz1_high])
        coordinates2=tuple(round(x, 2) for x in [mob_leaf['mobility_value'], mobmax, mz2_low, mz2_high])
        finished_windows.append({
            'coord1': coordinates1,
            'coord2': coordinates2,
            'coverage1': wu.get_covered_ions(coordinates1[:2], window1), 
            'coverage2': wu.get_covered_ions(coordinates2[:2], window2)
        })
    # TODO: figure size from parameters
    fig = px.imshow(pdata,width=750, height=370,origin='lower',aspect='equal')
    #fig.update_layout(yaxis=dict(autorange=True))
    fig2 = px.imshow(pdata,width=750, height=370,origin='lower',aspect='equal')
    #fig2.update_layout(yaxis=dict(autorange=True))
    window_rects = []
    use_cols = pasef_method_table_columns + ['Num covered']
    table_data = [{c: ['MS1', 0, '-', '-', '-', '-', '-','-'][i] for i, c in enumerate(use_cols)}]
    cycle_id = 1
    for fw in finished_windows:
        window_rects.append(wu.coord_to_plotly_rect(*fw['coord1'], pdata))#, defaults = defvals))
        window_rects.append(wu.coord_to_plotly_rect(*fw['coord2'], pdata))#, defaults = defvals))
        rowdata = [
            ['PASEF', cycle_id] + list(fw['coord2']) + ['-',fw['coverage2']],
            ['PASEF', cycle_id] + list(fw['coord1']) + ['-',fw['coverage1']]
        ]
        table_data.append(
            {c: rowdata[0][i] for i, c in enumerate(use_cols)}
        )
        table_data.append(
            {c: rowdata[1][i] for i, c in enumerate(use_cols)}
        )
        cycle_id += 1
    fig.update_layout(shapes=window_rects)
    #fig.show()
    notify = len(notify)>0
    return [
        dbc.Row([ 
            html.Div([
                html.Audio(id='windowmaker-music', controls=True,src=notification_music,children='',autoPlay=notify),
                html.Div(dbc.Button(children = 'Download DIAPASEF method', id='windowmaker-download-diapasef'), style={'padding': '5px 20px 5px 5px'})
            ], style={'display': 'inline-flex'})
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='windowmaker-filtered-plot',figure=fig2),width=6),
            dbc.Col(dcc.Graph(id='windowmaker-windowed-plot',figure=fig),width=6),
            dash_table.DataTable(
                id='windowmaker-window-output-table',
                columns=(
                    [{'id': cname, 'name': cname} for cname in use_cols]
                ),
                data=table_data,
                editable=True
            ),
        ]),
    ]
    #fig.write_html(f'Final windows plo {ln}.html')

@callback(
    Output('windowmaker-download-method', 'data'),
    Input('windowmaker-download-diapasef','n_clicks'),
    State('windowmaker-window-output-table', 'data'),
    State('windowmaker-mgf-file-upload', 'filename'),
    prevent_initial_call = True
)
def download_pasef_method(clicks, table_data, input_filename):
    if clicks is None or clicks < 1:
        return no_update
    output = [
        '#----------------------------------------------------------------------------------------------------',
        f'# {" | ".join(pasef_method_table_columns)} ',
        '#----------------------------------------------------------------------------------------------------'
    ]
    output_column_widths = [10, 10, 17, 15, 18, 16, 8]
    for td in table_data:
        output.append(
            ','.join([
                str(td[col]).rjust(output_column_widths[i],' ') for i, col in enumerate(
                    pasef_method_table_columns
                )
            ])
        )
    timestamp: str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    return dict(content='\n'.join(output), filename=f'{timestamp}-{input_filename}_diaPASEFmethod.txt')

@callback(
    Output('windowmaker-music', 'src'),
    Output('windowmaker-music','children'),
    Input('windowmaker-change-music-button', 'n_clicks'),
    State('windowmaker-music','children')
)
def change_music(click, current) -> html.Div:
    if current == '':
        newnum = 0
    else:
        newnum = current + 1
    if newnum >= len(music_choices):
        newnum = 0
    return (music_choices[newnum], newnum)

@callback(
    Output({'type': 'PREDEFEQBUTTON', 'name':ALL}, 'disabled'),
    Output('windowmaker-add-line-button', 'disabled'),
    Output('windowmaker-calculate-windows-button','disabled'),
    Input('windowmaker-mgf-file-upload-success','style'),
    State({'type': 'PREDEFEQBUTTON', 'name':ALL}, 'disabled')
)
def enabled_buttons_if_successful_upload(upload_style, eq_buttons):
    if upload_style['background-color'] in ['green','yellow','blue']:
        return ([False for button in eq_buttons],False,False)
    else:
        return ([True for button in eq_buttons],True,True)


@callback(
    Output('windowmaker-full-data-store','data'),
    Output('windowmaker-mgf-file-upload-success', 'style'),
    Output('windowmaker-input-file-info-text','children'),
    Input('windowmaker-mgf-file-upload', 'filename'),
    State('windowmaker-mgf-file-upload', 'contents'),
    State('windowmaker-mgf-file-upload-success', 'style'),
    prevent_initial_call=True
)
def handle_mgf(mgf_file_name, mgf_file_contents, current_style):

    mgf_json = 'Error in MGF handling'

    #if ctx.triggered_id == 'windowmaker-run-number-inputted-button':
        # TODO: implement fetching of ready data
        # TODO: pre-check after button press whether the run number is in the database. If not, open a modal and present closest five in both directions to choose from. Save value into windowmaker-run-id, and use that to trigger.
     #   return {'mgf': mgf_json, 'plot': pdf_json},current_style
    if mgf_file_name is None:
        return no_update, no_update
    mgf_df, osize = wu.handle_file(mgf_file_name, mgf_file_contents)
    if slim_testing_factor:
        mgf_df = mgf_df.loc[random.sample(list(mgf_df.index.values), max(100, int(mgf_df.shape[0]/slim_testing_factor)))]
    current_style['background-color'] = 'red'
    file_info: str = f'{mgf_file_name}: {osize} rows of ions slimmed down to {mgf_df.shape[0]} unique rows'
    if mgf_df.shape[0] > 0:
        mgf_json = mgf_df.to_json(orient='split')
        current_style['background-color'] = 'green'
    return {'mgf': mgf_json}, current_style, file_info
        #pdf_json = wu.make_pdata(mgf_df).to_json(orient='split')
        # Only done once for charge plots:
        # Hacky shit to produce a prettier plot
        #ch_dic = wu.do_charges(mgf_df)

    #return {'mgf': mgf_json, 'plot': pdf_json, 'charge_states': ch_dic}, current_style