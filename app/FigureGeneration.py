import base64
import os
import io
import numpy as np
import pandas as pd
from typing import Any
from plotly import express as px
from plotly import io as pio
import plotly.graph_objects as go
from dash import html, dcc
import dash_bio
from statsmodels.stats import multitest
from scipy.stats import ttest_ind
from sklearn import manifold, decomposition
import matplotlib as mpl
from matplotlib import pyplot as plt
from supervenn import supervenn as svenn

class FigureGeneration:

    @property
    def defaults(self) -> dict:
        return self._defaults

    @defaults.setter
    def defaults(self, default_dict: dict) -> None:
        self._defaults: dict = default_dict
    
    @property
    def generated_figures(self) -> list:
        return self._generated_figures

    def add_figure(self, figure, title, legend) -> None:
        self._generated_figures[title] = [legend,figure]

    @property
    def save_dir(self) -> str: 
        return self._defaults['save dir']
    @save_dir.setter
    def save_dir(self, save_dir:str) -> None: 
        self._defaults['save dir'] = save_dir

    def change_imagebutton_format(self,format:str) -> None:
        format: str = format.lower()
        assert format in 'svg png jpegt webp'.split()
        self.defaults['config']['toImageButton']['format'] = format

    def __init__(self, config_keys: dict = None) -> None:
        # Move these defaults to the parameters.json
        self.defaults = {
            'config': {
                'toImageButtonOptions': {
                    'format': 'svg', # one of png, svg, jpeg, webp
                },
            },
            'width': 1200,
            'height': 700
        }
        if config_keys is not None:
            for key, value in config_keys.items():
                self._defaults[key] = value
        self._defaults: dict
        self._save_dir: str
        self._generated_figures: dict = {}

    def save_figures(self, figure_dir: str, save_data: list) -> None:
        for figure_name, (figure_legend, figure) in self.generated_figures.items():
            if isinstance(figure, str):
                os.rename(figure, os.path.join(figure_dir,figure_name+'.pdf'))
            else:
                figure: go.Figure = go.Figure(figure)
                if figure_legend is None:
                    figure_legend = ''
                elif isinstance(figure_legend,list):
                    figure_legend = '\n'.join(figure_legend)
                with open(os.path.join(figure_dir, figure_name + '.txt'),'w',encoding='utf-8') as fil:
                    fil.write(figure_legend)
                figure.write_html(os.path.join(figure_dir,figure_name+'.html'),config=self.defaults['config'])
                figure.write_image(os.path.join(figure_dir,figure_name+'.pdf'))

        if save_data is not None:
            for i, figure in enumerate(save_data[0]):
                figure_name:str = save_data[1][i][0]
                figure_legend:str = save_data[1][i][1]
                if isinstance(figure, str):
                    os.rename(figure, os.path.join(figure_dir,figure_name+'testing2.pdf'))
                else:
                    figure: go.Figure = go.Figure(figure)
                    with open(os.path.join(figure_dir, figure_name + 'testing2.txt'),'w',encoding='utf-8') as fil:
                        fil.write(figure_legend)
                        figure.write_html(os.path.join(figure_dir,figure_name+'testing2.html'),config=self.defaults['config'])

    def get_cut_colors(self, colormapname: str = 'gist_ncar', number_of_colors: int = 15,
                    cut: float = 0.4) -> list:
        """Returns cut colors from the given colormapname

        Parameters:
        - colormap: which matplotlib colormap to use
        - number_of_colors: how many colors to return. Colors will be equally spaced in the map
        - cut: how much to cut the colors.
        """
        number_of_colors += 1
        colors = (1. - cut) * (plt.get_cmap(colormapname)(np.linspace(0., 1., number_of_colors))) + \
            cut * np.ones((number_of_colors, 4))
        colors = colors[:-1]
        return colors

    def add_replicate_colors(self, data_df: pd.DataFrame, group_dict: dict) -> None:
        """Adds "Color" column to long-format data_df based on group_dict dictionary.
        Each group will get its own color, and data_df.index should correspond to sample_names, which are found in group_dict as keys.
        
        Parameters:
        data_df: long data frame with groups in index
        group_dict: dictionary of {sample_name: group_name}
        """
        need_cols: int = list(
            {
                group_dict[sname] for sname in
                data_df.index.unique()
                if sname in group_dict
            }
        )
        colors: list = self.get_cut_colors(number_of_colors=len(need_cols))
        colors = {sname: colors[i] for i, sname in enumerate(need_cols)}
        color_column: list = []
        for sample_name in data_df.index.values:
            color_column.append(colors[group_dict[sample_name]])
            
        colors: list = []
        for c1,c2,c3,c4 in color_column:
            colors.append(f'rgba({c1},{c2},{c3},{c4})')
        data_df.loc[:, 'Color'] = colors


    def comparative_violin_plot(self, sets: list, names: list = None, id_name: str = None, title: str = None, legend: str = None, colors: list = None, showbox:bool = False) -> dcc.Graph:
        if id_name is None:
            id_name: str = 'comparative-violin-plot'
        if isinstance(colors, list):
            assert ((len(sets) == len(colors)) | (colors is None)
                    ), 'Length of "sets" should be the same as length of "colors"'
        if isinstance(names, list):
            assert ((len(sets) == len(names))
                    ), 'Length of "sets" should be the same as length of "names"'
        else:
            names: list = []
            for i in range(0, len(sets)):
                names.append(f'Set {i+1}')
        plot_data: list = np.array([])
        plot_legend: list = [[], []]
        for i, data_frame in enumerate(sets):
            for col in data_frame.columns:
                plot_data = np.append(plot_data, data_frame[col].values)
                plot_legend[0].extend([names[i]]*data_frame.shape[0])
                plot_legend[1].extend([f'{col} {names[i]}']*data_frame.shape[0])
        plot_df: pd.DataFrame = pd.DataFrame(
            {
                'Values': plot_data,
                'Column': plot_legend[1],
                'Name': plot_legend[0]
            }
        )

        if title is None:
            title: str = ''
        figure: go.Figure = px.violin(
                plot_df,
                y='Values',
                x='Column',
                color='Name',
                box=showbox,
                title=title,
                color_discrete_sequence=colors,
                height=self.defaults['height'],
                width=self.defaults['width'],
                )
        figure.update_xaxes(title=None)
        self.add_figure(figure, title, legend)
        return (
            figure,
            dcc.Graph(
                config=self.defaults['config'],
                id=id_name,
                figure=figure
            )
        )


    def distribution_figure(self, data_table, color_dict, sample_groups, title: str = 'Value distribution', legend:str = None, ) -> dcc.Graph:
        rev_sample_groups: dict = {}
        for k, v in sample_groups.items():
            if v not in rev_sample_groups:
                rev_sample_groups[v] = []
            rev_sample_groups[v].append(k)
        names: list = sorted(list(rev_sample_groups.keys()))
        colors: list = [color_dict[rev_sample_groups[k][0]] for k in names]
        data_frames: list = []
        for sample_group in names:
            data_frames.append(data_table[rev_sample_groups[sample_group]])
        id_str: str = 'value-distribution-figure'
        return self.comparative_violin_plot(data_frames, names=names, title=title, id_name=id_str, colors=colors, legend=legend)


    def before_after_plot(self, before: pd.Series, after: pd.Series, title: str = None, name_legend:list = None, legend:str = None, ) -> dcc.Graph:
        data: list = [['Before or after', 'Count', 'Sample']]
        for i in before.index:
            if i in after.index:
                data.extend([
                    ['before', before[i], i],
                    ['after', after[i], i]
                ])
        if title is None:
            title: str = ''
        dataframe: pd.DataFrame = pd.DataFrame(data=data[1:], columns=data[0])
        figure: go.Figure = px.bar(
                dataframe,
                x='Sample',
                y='Count',
                color='Before or after',
                barmode='group',
                title=title,
                height=self.defaults['height'],
                width=self.defaults['width']
            )
        return (
            figure,
            dcc.Graph(config=self.defaults['config'], 
            id='before-and-after-na-filter-figure',
            figure=figure
            )
        )


    def histogram(self, data_table: pd.DataFrame, x_column: str, title: str, legend:str = None, **kwargs) -> dcc.Graph:
        if 'height' not in kwargs:
            kwargs: dict = dict(kwargs,height=self.defaults['height'])
        if 'width' not in kwargs:
            kwargs = dict(kwargs,width=self.defaults['width'])
        figure: go.Figure = px.histogram(
            data_table,
            x=x_column,
            title=title,
            **kwargs
        )
        self.add_figure(figure, title, legend)
        return (
            figure,
            dcc.Graph(config=self.defaults['config'], 
                id=title.lower().replace('.', '-').replace('_', '-').replace(' ', '-'),
                figure=figure
            )
        )


    def imputation_histogram(self, non_imputed, imputed, title: str = None, legend:str = None, **kwargs) -> dcc.Graph:
        #x,y = sp.coo_matrix(non_imputed.isnull()).nonzero()
        non_imputed: pd.DataFrame = non_imputed.melt(ignore_index=False).rename(
            columns={'variable': 'Sample'})
        imputed: pd.DataFrame = imputed.melt(ignore_index=False).rename(
            columns={'variable': 'Sample'}).rename(columns={'value': 'log2 value'})
        imputed['Imputed'] = non_imputed['value'].isna()
        if title is None:
            title: str = 'Imputation'
        if 'height' not in kwargs:
            kwargs: dict = dict(kwargs,height=self.defaults['height'])
        if 'width' not in kwargs:
            kwargs = dict(kwargs,width=self.defaults['width'])
            
        figure: go.Figure = px.histogram(
            imputed,
            x='log2 value',
            marginal='violin',
            color='Imputed',
            title=title,
            **kwargs
        )
        self.add_figure(figure, title, legend)
        return (
            figure,
            dcc.Graph(config=self.defaults['config'], id='missing-value-histogram', figure=figure)
        )
    def bar_graph(self, graph_id:str,*args,**kwargs) -> None:
        return dcc.Graph(
            id=graph_id,
            config=self.defaults['config'],
            figure=self.bar_plot(
                *args,
                **kwargs
            )
        )


    def bar_plot(self, value_df: pd.DataFrame, 
                 title: str, 
                 legend:str = None, 
                 x_name: str = None, 
                 x_label:str = None,
                 y_name: str = None,
                 y_label:str = None, 
                 y_idx: int = 0, 
                 barmode:str = 'relative', 
                 color: bool = True, 
                 color_col: str = None, 
                 hide_legend=False, 
                 color_discrete_map=False, 
                 color_discrete_map_dict:dict=None,
                 width: int = None,
                 height: int = None) -> px.bar:
        """Draws a bar plot from the given input.
        
        Parameters:
        value_df: dataframe containing the plot data
        title: title for the figure
        x_name: name of the column to use for x-axis values
        y_name: name of the column to use for y-axis values
        y_idx: index of the column to use for y-axis values
        barmode: see https://plotly.com/python-api-reference/generated/plotly.express.bar
        color: True(default) if a column called "Color" contains color values for the plot
        color_col: name of color information containing column, see px.bar reference
        hide_legend: True, if legend should be hidden
        color_discrete_map: if True, color_discrete_map='identity' will be used with the plotly function.
        """
        colorval: str
        if color_col is not None:
            colorval = color_col
        elif color:
            colorval = 'Color'
        else:
            colorval = None

        cdm_val: dict = None
        if color_discrete_map_dict is not None:
            cdm_val = color_discrete_map_dict
        else:
            if color_discrete_map:
                cdm_val = 'identity'
        if y_name is None:
            y_name: str = value_df.columns[y_idx]
        if x_name is None:
            x_name: str = value_df.index
        if height is None:
            height: int = self.defaults['height']
        if width is None:
            width: int = self.defaults['width']
        figure: px.bar = px.bar(
            value_df,
            x=x_name,  # 'Sample name',
            y=y_name,
            title=title,
            color=colorval,
            barmode=barmode,
            color_discrete_map=cdm_val,
            height=height,
            width=width
        )
        if x_label is not None:
            figure.update_layout(
                xaxis_title=x_label
            )
        if y_label is not None:
            figure.update_layout(
                yaxis_title=y_label
            )
        figure.update_xaxes(type='category')
        if hide_legend:
            figure.update_layout(showlegend=False)
        
        self.add_figure(figure, title, legend)
        return figure


    def contaminant_figure(self, data_table: pd.DataFrame, contaminant_list: list, title:str = None, legend:str = None, ) -> dcc.Graph:

        if title is None:
            title = 'Signal sum from contaminants per sample'
        contaminant_list: list = list(set(contaminant_list) & set(data_table.index.values))
        plot_data: list = []
        plot_index: list = []
        for column in data_table.columns:
            plot_index.append(column)
            plot_index.append(column)
        #plot_index:pd.Index = data_table.columns
        contaminants: pd.DataFrame = data_table.loc[contaminant_list]
        non_contaminants: pd.DataFrame = data_table.loc[~data_table.index.isin(contaminant_list)]

        for i in range(0, len(plot_index), 2):
            sample_name: str = plot_index[i]
            plot_data.append([contaminants[sample_name].sum(), 'Contaminants'])
            plot_data.append([non_contaminants[sample_name].sum(), 'Non-contaminants'])
        figure: go.Figure = self.bar_plot(
                pd.DataFrame(
                    data=plot_data,
                    index=plot_index,
                    columns=['Sum value','Contaminant'],
                ),
                x_label='Sample',
                title=title,
                color_col='Contaminant',
                color_discrete_map_dict = {'Contaminants': 'Red','Non-contaminants': 'Blue'}
            )

        self.add_figure(figure, title, legend)
        return (
            figure,
            dcc.Graph(config=self.defaults['config'], 
                id='contaminant-bar-plot',
                figure=figure
            )
        )

    # TODO: Merge these six functions into one. Or rather, delete them and just call bar_plot from data analysis.py
    def sum_value_figure(self, sum_data,valname:str='Value', legend:str = None, ) -> dcc.Graph:
        figure: go.Figure = self.bar_plot(
            sum_data, title=f'{valname} sum per sample', color_discrete_map=True)
        return (
            figure,
            dcc.Graph(config=self.defaults['config'], id='value-sum-figure', figure=figure)
        )

    def avg_value_figure(self, avg_data,valname:str='Value', legend:str = None, ) -> dcc.Graph:
        figure: go.Figure = self.bar_plot(
            avg_data, title=f'{valname} mean per sample', color_discrete_map=True)
        return (
            figure,
            dcc.Graph(config=self.defaults['config'], id='value-sum-figure', figure=figure)
        )

    def missing_figure(self, na_data, legend:str = None, ) -> dcc.Graph:
        figure: px.bar = self.bar_plot(
            na_data, title='Missing values per sample', color_discrete_map=True)
        return (
            figure,
            dcc.Graph(config=self.defaults['config'], id='missing-count-figure', figure=figure)
        )

    def protein_count_figure(self, count_data, legend:str = None, ) -> dcc.Graph:
        """Generates a bar plot of given data"""
        figure: px.bar = self.bar_plot(
            count_data, title='Proteins per sample', color_discrete_map=True)
        return (
            figure,
            dcc.Graph(config=self.defaults['config'], id='protein-count-figure', figure=figure)
        )

    def protein_coverage(self, data_table, title:str = None, legend:str = None, ) -> dcc.Graph:
        if title is None:
            title = 'Protein coverage'
        if legend is None:
            legend = [
                'Protein coverage in the analyzed samples.',
                'The plot shows how many proteins were shared by how many samples.'
            ]
        figure: go.Figure = self.bar_plot(
                pd.DataFrame(
                    data_table.notna()
                    .astype(int)
                    .sum(axis=1)
                    .value_counts(), columns=['Identified in # samples']),
                title,
                y_label = 'Protein count',
                color=False
                )
        self.add_figure(figure, title, legend)
        return (
            figure,
            dcc.Graph(config=self.defaults['config'], 
                id=f'protein-coverage-{title}',
                figure=figure
            )
        )


    def volcano_plots(self, data_table, sample_groups, control_group, replacement_index:str = None, legend_for_all:str = None, data_is_log2_transformed:bool = True) -> list:
        """Generates volcano plots of all sample groups vs given control group in data_table.

        :param data_table: table of samples (columns) and measurements(rows)
        :param sample_groups: dictionary of {sample_group_name: [sample_columns]}
        :param control_group: name of the control group
        :param replacement_index: replace index with these values. Must be in same order.
        
        :returns: a list of [dcc.Graph] volcano plots.
        """
        if legend_for_all is None:
            legend_for_all = 'Changes are considered significant, if the associated FDR-corrected p-value is under 0.05, and the associated log2 fold change either over 1 or under -1.'
        if replacement_index is not None:
            data_table: pd.DataFrame = data_table.copy()
            data_table.index = replacement_index
        control_cols: list = sample_groups[control_group]
        volcanoes: list = []
        figures: list = []
        significants: list = []
        for group_name, group_cols in sample_groups.items():
            if group_name == control_group:
                continue
            sig_df: pd.DataFrame
            figure: go.Figure
            sig_df, figure = self.volcano_plot(
                        data_table,
                        group_name,
                        control_group,
                        group_cols,
                        control_cols,
                        data_is_log2_transformed = data_is_log2_transformed
                    )
            significants.append(sig_df)
            figures.append(figure)
            title: str = f'Volcano plot {group_name} vs {control_group}'
            legend: list = [
                    f'Volcano plot of {group_name} vs {control_group}.',
                    legend_for_all
                ]
            self.add_figure(figure, title, legend)
            volcanoes.append(
                dcc.Graph(config=self.defaults['config'],
                    id=f'volcano-{group_name}-vs-{control_group}',
                    figure=figure
                )
            )
        return (pd.concat(significants), figures, volcanoes)

    def tic_summary_graphs(self, info_df, tics_found) -> html.Div:
        #tics_found[run_id] = ticdf
        return [html.Div('figure Not implemented'), html.Div('graph Not implemented')]

    def volcano_plot(
            self, data_table, sample_name, control_name, sample_columns, control_columns,
            data_is_log2_transformed:bool = True, adj_p_threshold: float = 0.05,
            fc_threshold: float = 1, fc_axis_min_max: float = 2, highlight_only: list = None
        ) -> tuple:
        """Draws a Volcano plot of the given data_table

        :param data_table: table of samples (columns) and measurements(rows)
        :param sample_name: Name for the test sample
        :param control_name: Name for control
        :param sample_columns: columns corresponding to test sample data
        :param control_columns: columns corresponding to control data
        :param adj_p_threshold: threshold of significance for the calculated adjusted p value (Default 0.05)
        :param fc_threshold: threshold of significance for the log2 fold change. Proteins with fold change of <-fc_threshold or >fc_threshold are considered significant (Default 1)
        :param fc_axis_min_max: minimum for the maximum value of fold change axis. Default of 2 is used to keep the plot from becoming ridiculously narrow
        :param highlight_only: only highlight significant ones that are also in this list
        
        :returns: (result: pd.DataFrame, volcano_plot: go.Figure)
        """

        # Calculate log2 fold change for each protein between the two sample groups
        if data_is_log2_transformed:
            log2_fold_change: pd.Series = data_table[sample_columns].mean(
                axis=1) - data_table[control_columns].mean(axis=1)
        else:
            log2_fold_change: pd.Series = np.log2(data_table[sample_columns].mean(
                axis=1)) - np.log2(data_table[control_columns].mean(axis=1))

        # Calculate the p-value for each protein using a two-sample t-test
        p_value: float = data_table.apply(lambda x: ttest_ind(
            x[sample_columns], x[control_columns])[1], axis=1)
        # if all values between mutant and control are exactly the same, p value will be undefined. 
        # This can happen with some normalizations, e.g. baitnorm
        # However, if it does happen, we want to make a note of it for debugging, hence we will first save all info:
        if p_value.isna().sum()>0:
            if not os.path.isdir('DEBUG'):
                os.makedirs('DEBUG')
            i = 1
            debugfile: str = os.path.join('DEBUG',f'p_value_nan_{i}')
            while os.path.isfile(debugfile):
                i+=1
                debugfile = os.path.join('DEBUG',f'p_value_nan_{i}')
            with open(debugfile) as fil:
                fil.write(f'nan encountered in p values\nSample({sample_name}) cols:\n')
                fil.write('\t'.join(sample_columns) + '\n')
                fil.write(f'Control({control_name}) cols:\n')
                fil.write('\t'.join(control_columns) + '\n')
            data_table.to_csv(debugfile + '.tsv',sep='\t')
            p_value = p_value.fillna(1) 

        # Adjust the p-values for multiple testing using the Benjamini-Hochberg correction method
        _: Any
        p_value_adj: np.ndarray
        _, p_value_adj, _, _ = multitest.multipletests(p_value, method='fdr_bh')

        # Create a new dataframe containing the fold change and adjusted p-value for each protein
        result: pd.DataFrame = pd.DataFrame(
            {'fold_change': log2_fold_change, 'p_value_adj': p_value_adj, 'p_value': p_value})
        result['Name'] = data_table.index.values
        comparison_name: str = f'{sample_name} vs {control_name}'
        result['Comparison'] = comparison_name
        result['significant'] = ((result['p_value_adj'] < adj_p_threshold) & (
            abs(result['fold_change']) >= 1))
        result['p_value_adj_neg_log10'] = -np.log10(result['p_value_adj'])
        
        if not highlight_only:
            highlight_only = set(result['Name'].values)
        result['Highlight'] = [row['Name'] if ((row['significant']) & (row['Name'] in highlight_only))
                            else '' for _, row in result.iterrows()]

        col_order: list = ['Comparison','Name','significant']
        col_order.extend([c for c in result.columns if c not in col_order])
        result = result[col_order]
        result.sort_values(by='significant',ascending=True,inplace=True)
        # Draw the volcano plot using plotly express
        fig: go.Figure = px.scatter(
            result,
            x='fold_change',
            y='p_value_adj_neg_log10',
            title=comparison_name,
            color='significant',
            text='Highlight',
            height=self.defaults['height'],
            width=self.defaults['width']
        )

        # Set yaxis properties
        p_thresh_val: float = -np.log10(adj_p_threshold)
        pmax: float = max(result['p_value_adj_neg_log10'].max(), p_thresh_val)+0.5
        fig.update_yaxes(title_text='-log10 (q-value)', range=[0, pmax])
        # Set the x-axis properties
        fcrange: float = max(abs(result['fold_change']).max(), fc_threshold)
        if fcrange < fc_axis_min_max:
            fcrange = fc_axis_min_max
        fcrange += 0.25
        fig.update_xaxes(title_text='Fold change', range=[-fcrange, fcrange])

        # Add vertical lines indicating the significance thresholds
        fig.add_shape(type='line', x0=-fc_threshold, y0=0, x1=-
                    fc_threshold, y1=pmax, line=dict(width=2, dash='dot'))
        fig.add_shape(type='line', x0=fc_threshold, y0=0,
                    x1=fc_threshold, y1=pmax, line=dict(width=2, dash='dot'))
        # And horizontal line:
        fig.add_shape(type='line', x0=-fcrange, y0=p_thresh_val,
                    x1=fcrange, y1=p_thresh_val, line=dict(width=2, dash='dot'))

        # Return the plot
        return (result, fig)

    def improve_text_position(self, data_frame: pd.DataFrame) -> list:
        """Returns a list of text positions in alternating pattern."""
        
        positions: list = ['top left','top right','top center','middle left','middle right','middle center','bottom left','bottom right','bottom center']
        return [positions[i % len(positions)] for i in range(data_frame.shape[0])]

    def pca_plot(self, data_table: pd.DataFrame, rev_sample_groups: dict, n_components: int = 2, plot_name: str = None, legend:str = None, plot_title: str = None) -> dcc.Graph:
        """Draws a PCA plot of the given data_table

        Parameters:
        data_table: table of samples (columns) and measurements(rows)
        rev_sample_groups: dictionary of {sample_column_name: sample_group_name} containing all sample columns.
        n_components: how many components the PCA should have. Only first two will be used for the plot.
        plot_name: name for the plot. The id of the returned dcc.Graph will be f"pca-plot-{plot_name}"
        
        Returns: 
        dcc.Graph containing a go.Figure of the PCA plot.
        """
        # Compute PCA of the data
        data_df: pd.DataFrame = data_table.T
        pca: decomposition.PCA = decomposition.PCA(n_components=n_components)
        pca_result: np.ndarray = pca.fit_transform(data_df)
        
        pc1: float
        pc2: float
        pc1,pc2 = pca.explained_variance_ratio_
        pc1 = int(pc1*100)
        pc2 = int(pc2*100)
        pc1 = f'PC1 ({pc1}%)'
        pc2 = f'PC2 ({pc2}%)'

        data_df[pc1] = pca_result[:, 0]
        data_df[pc2] = pca_result[:, 1]
        data_df['Sample group'] = [rev_sample_groups[i] for i in data_df.index]
        data_df['Sample name'] = data_df.index
        if plot_name is None:
            plot_name: str = ''
        else:
            plot_name = '-' + plot_name
        if plot_title is None:
            plot_title:str = f'PCA {plot_name.strip("-")}'
        data_df.sort_values(by=pc1,ascending=True,inplace=True)
        figure: go.Figure = px.scatter(
            data_df,
            x=pc1,
            y=pc2,
            title=plot_title,
            color=data_df['Sample group'],
            height=self.defaults['height'],
            width=self.defaults['width']
        )
        figure.update_traces(
            marker_size=15,
            textposition=self.improve_text_position(data_df)
        )
        return (
            figure,
            dcc.Graph(config=self.defaults['config'], figure=figure, id=f'pca-plot{plot_name}')
        )


    def df_coefficient_of_variation(self, data_table: pd.DataFrame) -> pd.DataFrame:
        """Calculates coefficient of variation for a given dataframe with samples in columns and measurements in rows.

        Parameters:
        data_table: table of samples (columns) and measurements(rows)

        Returns:
        a pandas dataframe with three columns:
            "mean intensity": mean of each row in the data_table
            "CV": CV of each row in the data_table
            "%CV": CV*100
        """
        new_df: pd.DataFrame = pd.DataFrame(index=data_table.index)
        new_df['mean intensity'] = data_table.mean(axis=1)
        new_df['CV'] = data_table.std(axis=1)/new_df['mean intensity']
        new_df['%CV'] = new_df['CV']*100
        return new_df


    def coefficient_of_variation_plot(self, data_table: pd.DataFrame, plot_name: str = None, title: str = None,legend:str = None, draw_trendline:bool=True,trendline:str=None,trendline_options:dict=None) -> dcc.Graph:
        """Draws a CV plot of the given data_table with a trendline

        Parameters:
        data_table: table of samples (columns) and measurements(rows)
        plot_name: name for the plot. will be used for the id of the returned dcc.Graph object. default is "coefficient-of-variation"
        title: title for the figure
        draw_trendline: True(default) if trendline should be drawn
        trendline: Which type of trendline to draw (default lowess), see https://plotly.com/python-api-reference/generated/plotly.express.trendline_functions.html for options.
        trendline_options: options dict for the trendline

        Returns: 
        dcc.Graph containing a go.Figure of the CV plot.
        """

        if plot_name is None:
            plot_name:str = 'coefficient-of-variation'
        data_table: pd.DataFrame = self.df_coefficient_of_variation(data_table)
        int_col: str = 'mean intensity'
        cv_col: str = '%CV'
        data_table = data_table.sort_values(
            by=int_col, ascending=True, inplace=False)
        if draw_trendline:
            if trendline is None:
                trendline: str = 'lowess'
        else:
            trendline = None
        figure = px.scatter(
            data_table,
            x=cv_col,
            y=int_col,
            title=title,
            trendline=trendline,
            trendline_options = trendline_options,
            #trendline_options={'alpha': 0.1},
            trendline_color_override='black',
            height=self.defaults['height'],
            width=self.defaults['width'],
            #    color='hsl(210, 100%, 50%)'
        )
        if title is None:
            title: str = ''
        cv_str: str = f'Median CV: {data_table["CV"].median()}, '
        figure.update_layout(
            title=f'Coefficients of variation. {cv_str}',
            xaxis_title='Log2 normalized imputed mean intensity',
            yaxis_title='% CV'
        )
        return (
            figure,
            dcc.Graph(config=self.defaults['config'], figure=figure, id=f'scatter-{plot_name}')
        )

    def heatmap(self,matrix_df: pd.DataFrame, plot_name:str, value_name:str,legend:str = None) -> dcc.Graph:
        figure: go.Figure = px.imshow(matrix_df,
                                      aspect='auto',
                                      labels=dict(
                                        x=matrix_df.columns.name,
                                        y=matrix_df.index.name,
                                        color=value_name))
        return dcc.Graph(config=self.defaults['config'], figure=figure, id=f'heatmap-{plot_name}')

    def clustergram(self, plot_data: pd.DataFrame, color_map: list = None, **kwargs) -> dash_bio.Clustergram:
        """Draws a clustergram figure from the given plot_data data table.

        Parameters:
        plot_data: Clustergram data
        color-map: list of values and corresponding colors for the color map. default:  [[0.0, "#FFFFFF"], [1.0, "#EF553B"]]
        **kwargs: keyword arguments to pass on to dash_bio.Clustergram
        
        Returns: 
        dash_bio.Clustergram drawn with the input data.
        """
        if color_map is None:
            color_map: list = [
                [0.0, '#FFFFFF'],
                [1.0, '#EF553B']
            ]
        return dash_bio.Clustergram(
            data=plot_data,
            column_labels=list(plot_data.columns.values),
            row_labels=list(plot_data.index),
            color_map=color_map,
            link_method='average',
            height=self.defaults['height'],
            width=self.defaults['width'],
            **kwargs
        )


    def correlation_clustermap(self, data_table: pd.DataFrame, plot_name: str = None, legend:str = None) -> dcc.Graph:
        """Draws a correltion clustergram figure from the given data_table.

        Parameters:
        data_table: table of samples (columns) and measurements(rows)
        plot_name: name for the plot. will be used for the id of the returned dcc.Graph object.

        Returns: 
        dcc.Graph containing a dash_bio.Clustergram describing correlation between samples.
        """
        if plot_name is None:
            plot_name: str = 'correlation-clustermap'
        corr_df: pd.DataFrame = data_table.corr()
        figure: dash_bio.Clustergram = self.clustergram(
            corr_df,
        )
        # fig.update_zaxes(range=[0,1])
        # kokeile vielä layoutin kautta zmin ja zmax parametrejä
        # tai fig.update_layout(coloraxis=dict(cmax=6, cmin=3))
        return (
            figure,
            dcc.Graph(config=self.defaults['config'], figure=figure, id=plot_name)
        )


    def full_clustermap(self, data_table: pd.DataFrame, plot_name: str = None,hiderows:bool = True, hidecols:bool=False, legend:str = None, ) -> dcc.Graph:
        """Draws a clustermap figure from the given data_table.

        Parameters:
        data_table: table of samples (columns) and measurements(rows)
        plot_name: name for the plot. will be used for the id of the returned dcc.Graph object.

        Returns:
        dcc.Graph containing a dash_bio.Clustergram.
        """ 
        hidden_labels: list = []
        if hiderows:
            hidden_labels.append('row')
        if hidecols:
            hidden_labels.append('col')

        if plot_name is None:
            plot_name: str = 'full-clustermap'
        figure: dash_bio.Clustergram = self.clustergram(
            data_table,
            hidden_labels=hidden_labels,
            # The cluster parameter should be "column", but that does not work. According to source code, "col" is the correct usage, but it might get changed to reflect documentation later.
            cluster='col',
            center_values=False
        )
        return (
            figure,
            dcc.Graph(config=self.defaults['config'], figure=figure, id=plot_name)
        )
    
    def violin_reproducibility(self, data_table: pd.DataFrame, sample_groups: dict, title: str) -> go.Figure:
        distances_to_average_point_per_sample_group: pd.DataFrame = pd.DataFrame(index=data_table.index)
        drop_columns: list = []
        for sample_group, sample_columns in sample_groups.items():
            sample_group_mean: pd.Series = data_table[sample_columns].mean(axis=1)
            for column in sample_columns:
                column_distances:pd.Series = abs(data_table[column] - sample_group_mean)
                distances_to_average_point_per_sample_group[column] = column_distances
            distances_to_average_point_per_sample_group[sample_group] = distances_to_average_point_per_sample_group[sample_columns].mean(axis=1)
            drop_columns.extend(sample_columns)
        distances_to_average_point_per_sample_group.drop(columns=drop_columns,inplace=True)
        figure: go.Figure = px.violin(
            distances_to_average_point_per_sample_group,
            box=True,
            height=self.defaults['height'],
            width=self.defaults['width']
        )

        figure.update_layout(
            title=title,
            xaxis_title='Sample group',
            yaxis_title='Mean distance from average per protein'
        )

    def scatter_reproducibility(self, data_table: pd.DataFrame, sample_groups: dict, title) -> go.Figure:
        figure_datapoints:list = []
        for sample_group, sample_columns in sample_groups.items():
            for i, column in enumerate(sample_columns):
                if i == (len(sample_columns)-1):
                    break
                for column2 in sample_columns[i+1:]:
                    figure_datapoints.extend([[r[column], r[column2], sample_group] for _,r in data_table.iterrows()])
        plot_dataframe: pd.DataFrame = pd.DataFrame(data=figure_datapoints, columns = ['Sample A','Sample B', 'Sample group'])
        plot_dataframe = plot_dataframe.dropna() # No use plotting data points with missing values.
        figure: go.Figure = px.scatter(
            plot_dataframe, 
            title=title,
            x=plot_dataframe['Sample A'],
            y=plot_dataframe['Sample B'],
            color = 'Sample group',
            height=self.defaults['height'],
            width=self.defaults['width'],
            opacity=0.5)
        
        return figure

    def heatmap_reproducibility(self, data_table: pd.DataFrame, sample_groups: dict, title) -> go.Figure:
        figure_datapoints:list = []
        for sample_group, sample_columns in sample_groups.items():
            for i, column in enumerate(sample_columns):
                if i == (len(sample_columns)-1):
                    break
                for column2 in sample_columns[i+1:]:
                    figure_datapoints.extend([[r[column], r[column2], sample_group] for _,r in data_table.iterrows()])
        plot_dataframe: pd.DataFrame = pd.DataFrame(data=figure_datapoints, columns = ['Sample A','Sample B', 'Sample group'])
        plot_dataframe = plot_dataframe.dropna() # No use plotting data points with missing values.
        figure: go.Figure = px.density_heatmap(
            plot_dataframe,
            title=title,
            x='Sample A',
            y='Sample B',
            height=self.defaults['height']*(len(plot_dataframe['Sample group'].unique())/2),
            width=self.defaults['width'],
            color_continuous_scale='blues',
            #marginal_x = 'histogram',
            #marginal_y = 'histogram',
            facet_col= 'Sample group',
            facet_col_wrap=2,
            nbinsx=50,
            nbinsy=50,
        )
        return figure

    def reproducibility_figure(self, data_table: pd.DataFrame, sample_groups: dict, title='Sample reproducibility (missing values ignored)', style='heatmap', legend:str = None) -> dcc.Graph:
        """Produces a graph describing reproducibility within sammple groups

        Parameters:
        data_table: table of samples (columns) and measurements(rows)
        sample_groups: dictionary of {sample_group_name: [sample_columns]}
        title: title for the figure
        style: style of plot. Default is heatmap. Other valid entries are violin and scatter(computationally intensive).

        Returns:
        dcc.Graph containing a go.Figure describing the reproducibility of the samples within sample groups.
        """
        if style == 'violin':
            figure: go.Figure = self.violin_reproducibility(data_table, sample_groups, title)
        elif style == 'scatter':
            figure: go.Figure = self.scatter_reproducibility(data_table, sample_groups, title)
        else:
            figure: go.Figure = self.heatmap_reproducibility(data_table, sample_groups, title)
        return (
            figure,
            dcc.Graph(config=self.defaults['config'], figure=figure, id=f'{title.lower().replace(" ","-")}')
        )

    def reproducibility_figure_old_method(self, data_table: pd.DataFrame, sample_groups: dict, title='Reproducibility plot') -> dcc.Graph:
        results: list = []
        output_columns: list = None
        min_num_of_samples: int = None
        for _, sgcols in sample_groups.items():
            if len(sgcols) < 2:
                continue
            if min_num_of_samples is None:
                min_num_of_samples = len(sgcols)
            else:
                min_num_of_samples = min(min_num_of_samples, len(sgcols))
        ranked_proteins: list = list(data_table.sum(
            axis=1).sort_values(ascending=False).index)
        for groupname, groupcols in sample_groups.items():
            if len(groupcols) < 2:
                continue
            data_points: dict = {protein: {} for protein in data_table.index}
            for col in groupcols:
                index: int = 1
                for protein in ranked_proteins:
                    intensity: float = data_table.loc[protein, col]
                    data_points[protein][col] = [index, intensity]
                    index += 1
            if min_num_of_samples == 2:
                for protein, pdic in data_points.items():
                    results.append([
                        intensity for column, (index, intensity) in pdic.items()
                    ][:2])
                    results[-1].append(groupname)
            elif min_num_of_samples > 2:
                for protein, protein_dict in data_points.items():
                    ranks: list = []
                    intensities: list = []
                    for _, values in protein_dict.items():
                        ranks.append(values[0])
                        intensities.append(values[1])
                    int_series: pd.Series = pd.Series(intensities)
                    if int_series.notna().sum() == 0:
                        continue
                    avg_intensity: float = int_series.median()  # sum(intensities)/len(intensities)
                # avg_rank: float = sum(ranks)/len(ranks)

                    summed_distance: float = 0.0
                    for protein, (intensity, _) in protein_dict.items():
                        summed_distance += abs(intensity-avg_intensity)
                    results.append(
                        [
                            avg_intensity, summed_distance, groupname
                        ]
                    )
        figure: go.Figure = go.Figure(
            height=self.defaults['height'],
            width=self.defaults['width']
        )
        if min_num_of_samples == 2:
            output_columns = ['Intensity1', 'Intensity2', 'Group name']
            plot_dataframe: pd.DataFrame = pd.DataFrame(
                data=results, columns=output_columns)
            groups: list = list(plot_dataframe['Group name'].unique())
            for group in groups:
                df_group: pd.DataFrame = plot_dataframe[plot_dataframe['Group name'] == group]
                trace: go.Figure = go.Scatter(
                    x=df_group[output_columns[0]],
                    y=df_group[output_columns[1]],
                    xaxis='x',
                    yaxis='y',
                    mode='markers',
                    name=group,
                    marker=dict(
                        size=3,
                        opacity=1
                    )
                )
                figure.add_trace(trace)
            figure.update_layout(
                title=title,
                xaxis_title=output_columns[0],
                yaxis_title=output_columns[1],
                showlegend=True
            )
        else:
            output_columns = ['Average intensity',
                            'Summed distance to avg', 'Group name']
            plot_dataframe: pd.DataFrame = pd.DataFrame(
                data=results, columns=output_columns)
            groups: list = list(plot_dataframe['Group name'].unique())

            x: pd.Series = plot_dataframe[output_columns[0]]
            y: pd.Series = plot_dataframe[output_columns[1]]

            figure.add_trace(go.Histogram2dContour(
                x=x,
                y=y,
                colorscale='Blues',
                xaxis='x',
                yaxis='y'
            ))
            for group in groups:
                df_group: pd.DataFrame = plot_dataframe[plot_dataframe['Group name'] == group]
                trace: go.Figure = go.Scatter(
                    x=df_group[output_columns[0]],
                    y=df_group[output_columns[1]],
                    xaxis='x',
                    yaxis='y',
                    mode='markers',
                    name=group,
                    marker=dict(
                        size=3,
                        opacity=1
                    )
                )
                figure.add_trace(trace)

            figure.add_trace(go.Histogram(
                y=y,
                xaxis='x2',
                marker=dict(
                    color='rgba(0,0,0,1)'
                )
            ))
            figure.add_trace(go.Histogram(
                x=x,
                yaxis='y2',
                marker=dict(
                    color='rgba(0,0,0,1)'
                )
            ))

            figure.update_layout(
                autosize=False,
                xaxis=dict(
                    zeroline=False,
                    domain=[0, 0.85],
                    showgrid=False
                ),
                yaxis=dict(
                    zeroline=False,
                    domain=[0, 0.85],
                    showgrid=False
                ),
                xaxis2=dict(
                    zeroline=False,
                    domain=[0.85, 1],
                    showgrid=False
                ),
                yaxis2=dict(
                    zeroline=False,
                    domain=[0.85, 1],
                    showgrid=False
                ),
                height=600,
                width=600,
                bargap=0,
                hovermode='closest',
                showlegend=False
            )

            figure.update_layout(
                title=title,
                xaxis_title=output_columns[0],
                yaxis_title=output_columns[1]
            )
        return (
            figure,
            dcc.Graph(config=self.defaults['config'], figure=figure, id=f'{title.lower().replace(" ","-")}')
        )
    
    def sample_commonality_plot(self, data_table, rev_sample_groups,save_figure, save_format, legend:str = None) -> go.Figure:
        
        group_sets: dict = {}
        for column in data_table.columns:
            col_proteins: set = set(data_table[[column]].dropna().index.values)
            group_name: str = rev_sample_groups[column]
            if group_name not in group_sets:
                group_sets[group_name] = set()
            group_sets[group_name] |= col_proteins

        if len(group_sets.keys()) <= 6:
            return self.supervenn(group_sets, save_figure, save_format)
        else:
            return self.common_heatmap(group_sets)

    def common_heatmap(self,group_sets: dict) -> dcc.Graph:
        hmdata: list = []
        index: list = list(group_sets.keys())
        done = set()
        for gname in index:
            hmdata.append([])
            for gname2 in index:
                val: float
                if gname == gname2:
                    val = np.nan
                nstr: str = ''.join(sorted([gname,gname2]))
                if nstr in done:
                    val = np.nan
                else:
                    val = len(group_sets[gname] & group_sets[gname2])
                hmdata[-1].append(val)
        figure = px.imshow(pd.DataFrame(data=hmdata,index=index,columns=index))
        return [
            figure, 
            dcc.Graph(
            figure = figure
            )]



    def supervenn(self, group_sets: dict, save_figure:str=None, save_format:str='svg') -> html.Img:
        """Draws a super venn plot for the input data table.

        See https://github.com/gecko984/supervenn for details of the plot.
        Parameters:
        data_table: table of samples (columns) and measurements(rows)
        rev_sample_groups: dictionary of {sample_column_name: sample_group_name} containing all sample columns.
        figure_name: name for the figure title, as well as saved file
        save_figure: Path to save the generated figure. if None (default), figure will not be saved.
        save_format: format for the saved figure. default is svg.

        Returns:
        returns html.Img object containing the figure data in png form.
        """

        # Buffer for use
        buffer: io.BytesIO = io.BytesIO()
        fig: mpl.figure
        axes: mpl.Axes
        fig, axes = plt.subplots()
        fig.set_figheight(8)
        fig.set_figwidth(8)

        plot_sets: list = []
        plot_setnames: list = []
        for set_name, set_proteins in group_sets.items():
            plot_sets.append(set(set_proteins))
            plot_setnames.append(set_name)
        svenn(
            plot_sets,
            plot_setnames,
            ax=axes,
            rotate_col_annotations=True,
            col_annotations_area_height=1.2,
            widths_minmax_ratio=0.1
        )
        plt.xlabel('Shared proteins')
        plt.ylabel('Sample group')
        plt.savefig(buffer, format="png")
        ret_figure: str = None
        if save_figure:
            plt.savefig(f'{save_figure}.{save_format}', format=save_format)
            with open(os.path.join(save_figure + '.txt'),'w',encoding='utf-8') as fil:
                fil.write(
                    'Supervenn describes how many proteins each sample group has in common, and how many proteins are in each of the groups (e.g. how many proteins samples 1 and 2 have in common etc.)'
                )
            ret_figure = f'{save_figure}.{save_format}'
        plt.close()
        data: str = base64.b64encode(buffer.getbuffer()).decode(
            "utf8")  # encode to html elements
        buffer.close()
        return (
            ret_figure,
            html.Img(id='supervennfigure', src=f'data:image/png;base64,{data}')
        )

    def missing_clustermap(self, data_table: pd.DataFrame, plot_name: str = None, title: str = None, legend:str = None) -> dcc.Graph:
        """Generates a clustergram of missing values in a data_table.
        
        Parameters:
        data_table: table of samples (columns) and measurements(rows)
        plot_name: name for the output plot. will be used as the id value for the returned dcc.Graph object. If none, id will be "missing-value-clustergram"
        title: Plot title. If none, title will be "Missing values"

        Returns: 
        dcc.Graph containing a dash_bio.Clustergram of missing values.
        """
        if plot_name is None:
            plot_name: str = 'missing-value-clustermap'
        figure: dash_bio.Clustergram = self.clustergram(
            data_table.notna().astype(int),
            cluster='col',
            hidden_labels=['row'],
            color_map=[
                [0.0, '#000000'],
                [1.0, '#FFFFFF']
            ],
            center_values=False
        )
        if title is None:
            title: str = 'Missing values'
        figure.update_layout(title=title)
        return (
            figure,
            dcc.Graph(config=self.defaults['config'], 
                figure=figure,
                id=plot_name
            )
        )


    # TODO: NEeds testing, and should figure out if these are even necessary.
    def get_volcano_df(self, data_table: pd.DataFrame, sample_columns, control_columns, data_is_log2_transformed:bool = True) -> pd.DataFrame:
        # Calculate log2 fold change for each protein between the two sample groups
        if data_is_log2_transformed:
            log2_fold_change: pd.Series = data_table[sample_columns].mean(
                axis=1) / data_table[control_columns].mean(axis=1)
        else:
            log2_fold_change: pd.Series = np.log2(data_table[sample_columns].mean(
                axis=1)) / np.log2(data_table[control_columns].mean(axis=1))

        # Calculate the p-value for each protein using a two-sample t-test
        p_value: float = data_table.apply(lambda x: ttest_ind(
            x[sample_columns], x[control_columns])[1], axis=1)

        # Adjust the p-values for multiple testing using the Benjamini-Hochberg correction method
        _: Any
        p_value_adj: np.ndarray
        _, p_value_adj, _, _ = multitest.multipletests(p_value, method='fdr_bh')

        # Create a new dataframe containing the fold change and adjusted p-value for each protein
        result: pd.DataFrame = pd.DataFrame(
            {'fold_change': log2_fold_change, 'p_value_adj': p_value_adj, 'p_value': p_value})
        result['Name'] = data_table.index.values
        #result['p_value_adj_neg_log10'] = -np.log10(result['p_value_adj'])
        return result


    def volcano_plot2(self, data_table) -> html.Div:
        html.Div([
            'log2FC:',
            dcc.RangeSlider(
                id='range-slider',
                min=-3,
                max=3,
                step=0.05,
                marks={i: {'label': str(i)} for i in range(-3, 3)},
                value=[-0.5, 1]
            ),
            html.Br(),
            html.Div(
                dcc.Graph(config=self.defaults['config'], 
                    id='graph',
                    figure=dash_bio.VolcanoPlot(
                        dataframe=data_table,
                        height=self.defaults['height'],
                        width=self.defaults['width']
                    )
                )
            )
        ])


    def volcano_plots2(self, data_table, sample_groups, control_group) -> list:
        control_cols: list = sample_groups[control_group]
        volcanoes: list = []
        for group_name, group_cols in sample_groups.items():
            if group_name == control_group:
                continue
            volcanoes.append(
                dcc.Graph(config=self.defaults['config'], 
                    id=f'volcano-{group_name}-vs-{control_group}',
                    figure=self.volcano_plot(
                        data_table,
                        group_name,
                        control_group,
                        group_cols,
                        control_cols
                    )
                )
            )
        return volcanoes


    def t_sne_plot(self, data_table: pd.DataFrame, rev_sample_groups: dict, perplexity: int = 15, n_components: int = 2, iterations: int = 5000, figname: str = None) -> dcc.Graph:
        """Untested code"""
        data_df: pd.DataFrame = data_table.T
        perplexity: int = min(data_df.shape[0]-1, perplexity)
        tsne: manifold.TSNE = manifold.TSNE(n_components=n_components, verbose=0,
                                            perplexity=perplexity, n_iter=iterations)
        tsne_results: np.ndarray = tsne.fit_transform(data_df)
        data_df['t-SNE one'] = tsne_results[:, 0]
        data_df['t-SNE two'] = tsne_results[:, 1]
        data_df['Sample group'] = [rev_sample_groups[i] for i in data_df.index]
        data_df['Sample name'] = data_df.index
        if figname is None:
            figname: str = ''
        else:
            figname = '-' + figname
        fig: go.Figure = px.scatter(
            data_df, x='t-SNE one', y='t-SNE two', title=f't-SNE{figname}', color=data_df['Sample group'],
            text='Sample name')
        fig.update_traces(marker_size=15)
        return dcc.Graph(config=self.defaults['config'], figure=fig, id=f'tsne-plot{figname}')