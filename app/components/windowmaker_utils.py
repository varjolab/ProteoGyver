from datetime import datetime
from uuid import uuid4
import ahocorasick
import pandas as pd
import io
import base64
import numpy as np
from random import sample
from components.text_handling import replace_special_characters
from components.mathparser import MathParser

class TreeNode:
    def __init__(self, idn, mobility_value, window_index,  prev_num_covered_ions, num_ions, windows, mob_increment_perc, mob_ranges, parent=None):
        self.mobility_value = round(mobility_value,2)
        self.idn = idn
        self.num_covered_ions = prev_num_covered_ions+num_ions
        self.num_ions = num_ions
        self.window_index = window_index
        self.parent = parent
        self.parent_idn = None
        self.best_child_idn = None
        if self.parent is not None:
            self.parent_idn = parent.idn
        self.best_child = self.generate_best_child(windows, mob_ranges, mob_increment_perc)
        if self.best_child is not None:
            self.best_child_idn = self.best_child.idn        
        
    def generate_best_child(self, windows, mob_ranges, mob_increment_perc):
        kid_wind_ind = self.window_index + 1
        best_kid = None
        if kid_wind_ind == len(windows):
            return best_kid
         
        mob_rang = (
            max(mob_ranges[kid_wind_ind][0], self.mobility_value + 0.01),
            max(mob_ranges[kid_wind_ind][1], self.mobility_value + 0.02)
        )
        if mob_rang[0]>mob_rang[1]:
            print('range error')
            return None
        
        mob_increment = max(round((mob_rang[1]-mob_rang[0]) * mob_increment_perc,2), 0.01)
        for mob in np.arange(mob_rang[0], mob_rang[1]+mob_increment, mob_increment):
            if mob < self.mobility_value: 
                #print(kid_wind_ind, mob, self.mobility_value)
                continue
            kid = TreeNode(
                uuid4(),
                mob,
                kid_wind_ind,
                self.num_covered_ions,
                count_window_by_mob(windows[kid_wind_ind], mob),
                windows,
                mob_increment_perc,
                mob_ranges,
                parent = self                                                            
            )
            if (best_kid is None) or (kid.num_covered_ions > best_kid.num_covered_ions):
                best_kid = kid
            else:
                del kid
        return best_kid

    def get_tree_sum(self):
        rs = self.num_ions
        if self.best_child:
            rs += self.best_child.get_tree_sum()
        return rs        
        
    def update_prev_count(self):
        count = self.num_ions
        if self.parent:
            count += self.parent.update_prev_count()
        self.num_covered_ions = count
        return count
    def get_full_tree(self, start=False):
        ret = [self]
        if self.best_child is not None:
            ret.extend(self.best_child.get_full_tree())
        return ret        
    def set_parent(self, parent):
        self.parent = parent
    def set_child(self, child):
        self.best_child = child
    def to_dict(self):
        return {
            'id': str(self.idn),
            'mobility_value': self.mobility_value,
            'num_covered_ions': self.num_covered_ions,
            'num_ions': self.num_ions,
            'window_index': self.window_index,
            'parent': str(self.parent_idn),
            'best_child': str(self.best_child_idn)
        }
    def from_dict(self, dic):
        self.idn = dic['id']
        self.mobility_value = dic['mobility_value']
        self.num_covered_ions = dic['num_covered_ions']
        self.num_ions = dic['num_ions']
        self.window_index = dic['window_index']
        self.parent_idn = dic['parent']
        self.best_child_idn = dic['best_child']

    def optimize_position(self, mob_min, mob_max, window):
        start_count = self.num_ions
        best = [self.mobility_value, start_count]
        for mob in np.arange(mob_min, mob_max, 0.01):
            mob = round(mob,2)
            new_count = count_window_by_mob(window, mob)
            if new_count > best[1]:
                best = [mob, new_count]
        self.mobility_value = round(best[0],2)
        self.num_ions = best[1]


def handle_mgf(decoded_content):
    content_list: list = decoded_content.decode(encoding='utf-8').split('\n')
    mgf_df = pd.DataFrame()
    try:
        mgflibname = f'mgf_file_name'
        lib_data = []
        mgfheaders = ['ms','msms','scans','Mobility','Mz','Peptide mass','Charge','RT','fs','charge2']
        vals = ['','','','','','','','','','']
        for i, line in enumerate(content_list):
            line = line.strip('\r')
            if line.startswith('###FS:'):
                sc,mz,charge = line.replace('###FS:','').split('#')
                charge = charge.strip().split()[-1]
                if charge == 'undefined': charge = np.nan
                else: charge = str(int(charge[-1] + charge[:-1]))
                mz = float(mz.split()[-1])
                ms = content_list[i+1].split()[-1]
                msms = content_list[i+2].split()[-1]
                i+=2
                vals[8] = sc
                vals[0] = ms
                vals[1] = msms
                vals[4] = mz
                vals[9] = charge
            elif line.startswith('RTINSECONDS'):
                rt = float(line.split('=')[-1])
                vals[7] = rt
            elif line.startswith('RAWSCANS'):
                scans = line.split('=')[-1].strip()
                vals[2] = scans
            elif line.startswith('PEPMASS'):
                pm = float(line.split('=')[-1].split()[0])
                vals[5] = pm
            elif line.startswith('CHARGE'):
                ch = line.split('=')[-1].strip()
                ch = int(ch[-1]+ch[:-1])
                vals[6] = ch
            elif line.startswith('ION_MOBILITY'):
                im = float(line.split('=')[-1].split()[-1])
                if vals[6]=='':
                    vals[6] = np.nan
                vals[3] = im
                if len([c for c in vals if c == ''])>0:
                    break
                lib_data.append(vals)
                vals = ['','','','','','','','','','']
        mgf_df = pd.DataFrame(data=lib_data, columns=mgfheaders).replace('undefined', np.nan).drop(columns = ['charge2'])
        del lib_data
    except IndexError:
        print(f'IndexError: {mgflibname}')
    return (mgf_df, ['Mobility','Mz','Charge', 'Peptide mass','RT'])

def handle_spreadsheet(decoded_content, f_end):
    if f_end == 'csv':
        data: pd.DataFrame = pd.read_csv(io.StringIO(
            decoded_content.decode('utf-8')), index_col=False)
    elif f_end in ['tsv', 'tab', 'txt']:
        data: pd.DataFrame = pd.read_csv(io.StringIO(
            decoded_content.decode('utf-8')), sep='\t', index_col=False)
    elif f_end == 'xlsx':
        data: pd.DataFrame = pd.read_excel(
            io.BytesIO(decoded_content), engine='openpyxl')
    elif f_end == 'xls':
        data: pd.DataFrame = pd.read_excel(
            io.BytesIO(decoded_content), engine='xlrd')
    mobcol = ['']
    mzcol = ['']
    rtcol = ['']
    chargecol = ['']
    masscol = ['']
    modcol  = None
    for c in data.columns:
        if 'mobility' in c.lower():
            mobcol.append(c)
        elif 'mz' in c.lower():
            mzcol.append(c)
        elif 'retention' in c.lower():
            if 'time' in c.lower():
                rtcol.append(c)
        elif 'rt' in c.lower():
            rtcol.append(c)
        elif 'charge' in c.lower():
            chargecol.append(c)
        elif 'mass' in c.lower():
            if c != 'ExcludeFromAssay':
                masscol.append(c)
        if 'modified' in c.lower(): 
            if 'peptide' in c.lower():
                if not 'int' in c.lower():
                    modcol = c
    if modcol is not None:
        modvals = []
        for _,row in data.iterrows():
            mstr = []
            sc = None
            if '[' in row[modcol]:
                sc = '[]'
            elif '(' in row[modcol]:
                sc = '()'
            if sc is not None:
                for chunk in row[modcol].split(sc[0])[1:]:
                    mstr.append(chunk.split(sc[1])[0])
            if len(mstr)>0:
                modvals.append(';'.join(mstr))
            else:
                modvals.append('No modifications')
        data['Modifications'] = modvals
    if len(mobcol) > 1:
        mobcol = check_for(mobcol, ['precursor','peptide'])
    else:
        mobcol = mobcol[0]
    if len(mzcol) > 1:
        mzcol = check_for(mzcol, ['precursor','peptide'])
    else:
        mzcol = mzcol[0]
    if len(chargecol) > 1:
        chargecol = check_for(chargecol, ['precursor','peptide'])
    else:
        chargecol = chargecol[0]
    if len(rtcol) > 1:
        rtcol = check_for(rtcol, ['precursor','peptide'])
    else:
        rtcol = rtcol[0]
    if len(masscol) > 1:
        masscol = check_for(masscol, ['precursor','peptide'])
    else:
        masscol = masscol[0]
    return (data, [mobcol, mzcol, chargecol, masscol, rtcol])

def check_for(vals, tocheck):
    retval = []
    for v in vals:
        if tocheck[0] in v:
            retval.append(v)
    if len(retval) > 1:
        return check_for(retval, tocheck[1:])
    elif len(retval) == 0:
        return [v for v in vals if v!=''][0]
    return retval[0]

def do_charges(mgf_df):
    ch_dic = {}
    ion_threshold = 10
    for ch in mgf_df['Charge'].unique():
        if pd.isna(ch):
            ch_mgf_df = mgf_df[mgf_df['Charge'].isna()]
            ch = 'undefined'
        else:
            ch_mgf_df = mgf_df[mgf_df['Charge']==ch]
            ch = int(ch)
        if ch_mgf_df.shape[0] < ion_threshold:
            continue
        ch_pdf = make_pdata(ch_mgf_df)
        ch_dic[ch] = (ch_mgf_df.to_json(orient='split'), ch_pdf.to_json(orient='split'))
    return ch_dic
            

def handle_file(filename, file_contents):
    ext = filename.split('.')[-1].lower()
    _, content_string = file_contents.split(',')
    decoded_content: bytes = base64.b64decode(content_string)
    colnames = ['Mobility','Mz','Charge', 'Peptide mass','RT']
    if ext == 'mgf':
        retdf, retcols = handle_mgf(decoded_content)
    elif ext in ('xlsx','xls','txt','tsv','csv'):
        retdf, retcols = handle_spreadsheet(decoded_content,ext)
    else:
        retdf = pd.DataFrame(columns=colnames)
        retcols = colnames
    ogsize = retdf.shape[0]
    retdf.rename(columns={c: colnames[i] for i, c in enumerate(retcols)},inplace=True)
    cols_to_drop = ['RT']
    banned_words = ['intensity','fragment','product','annotation','qvalue']
    for c in retdf.columns:
        drop = False
        for b in banned_words:
            if b in c.lower():
                drop=True
                break
        if drop:
            cols_to_drop.append(c)
    retdf = retdf.drop(columns=cols_to_drop).drop_duplicates()
    return retdf, ogsize

def get_potential_filcols(data: pd.DataFrame):
    retlist = []
    if 'Modifications' in data.columns:
        retlist.append('Modifications')
    for c in data.columns:
        uvals = len([v for v in data[c].unique() if (pd.notna(v) & (str(v) != ''))])
        if uvals > 1:
            if uvals < 15:
                retlist.append(c)
    return sorted(retlist)

def make_pdata(mgf_df):
    # Hacky shit to produce a prettier plot
    num_samples = 10
    mzbinsize = 10
    mob_binsize = 0.01
    mobscale = mob_binsize/2
    mzscale = mzbinsize/2

    # TODO: Make better min and max values for both of these
    mzbins = np.arange(400,1800+mzbinsize, mzbinsize)
    mobbins = np.arange(0.8,1.4+mob_binsize, mob_binsize)
    matrix_df = mgf_df[['Mz','Mobility']]
    max_num = 10000
    if False:#matrix_df.shape[0] > max_num:
        matrix_df = matrix_df.loc[sample(list(matrix_df.index), max_num)]
    samples = np.empty((len(matrix_df), num_samples, 2))
    for i, (x, y) in enumerate(zip(matrix_df['Mz'], matrix_df['Mobility'])):
        samples[i] = np.random.normal(loc=[x, y], scale=[mzscale,mobscale], size=(num_samples, 2))
    reshaped_samples = samples.reshape((-1, 2))
    df = pd.DataFrame(data=reshaped_samples, columns=['X', 'Y'])

    hist, _, _ = np.histogram2d(df['X'], df['Y'], bins=[mzbins, mobbins])
    hist = hist.T
    mob_ind = []
    for i, m in enumerate(mobbins[1:]):
        mob_ind.append(f'{round(mobbins[i],3)}-{round(m,3)}')
    mz_ind = []
    for i, m in enumerate(mzbins[1:]):
        mz_ind.append(f'{mzbins[i]}-{m}')
    pdf = pd.DataFrame(hist, index=mob_ind, columns=mz_ind).replace(0,np.nan)
    return pdf

    
def count_windows(windows):
    ret = []
    for w, m in windows:
        ret.append([
            min(w),
            max(w),
            len(w),
            sorted(m)
        ])
    return ret

def get_covered_ions(mob_minmax: tuple, window: list):
    mob_min, mob_max = mob_minmax
    return len(
        [x for x in window[3] if ((x >= mob_min) & (x<= mob_max))]
    )

def pair_windows(windows):
    ret = []
    halfway = int(len(windows)/2)
    for i in range(0, halfway):
        ret.append([windows[i],windows[i+halfway]])
    return ret
    
def count_window_by_mob(window, mob_cutoff): 
    count = 0
    for w in window[0][3]:
        if w > mob_cutoff:
            break
        count +=1
    for w in window[1][3][::-1]:
        if w < mob_cutoff:
            break
        count += 1
    return count

def remove_from_matrix(df, equation_to_criteria):
    x_axis_vals = list(df.columns)
    y_axis_vals = list(df.index)
    for equation, discard_above_or_below in equation_to_criteria.items():
        new_data = []
        for i, y in enumerate(get_line_y(equation, range(0, len(x_axis_vals)), (0, len(y_axis_vals)))):
            if y is None:
                continue
            xstr = x_axis_vals[i]

            if discard_above_or_below == 'Above':
                df.loc[y_axis_vals[y+1:], xstr] = np.nan
            else:
                df.loc[y_axis_vals[:y], xstr] = np.nan

def remove_from_long(df, equation_to_criteria, mzbins, mobbins):
    keep_rows = set()
    df[['Mz','Mobility']]
    mzbins = [[float(x) for x in mb.split('-')] for mb in mzbins]
    mobbins = [[float(x) for x in mb.split('-')] for mb in mobbins]
    ## TODO: This is horrifyingly inefficient. Refactor!
    for row_index,row in df.iterrows():
        row_mzbin = [i for i, mb in enumerate(mzbins) if ((row['Mz']>= mb[0]) and (row['Mz']<mb[1]))][0]
        row_mobbin = [i for i, mb in enumerate(mobbins) if ((row['Mobility']>= mb[0]) and (row['Mobility']<mb[1]))][0]
        keep_row = True
        for equation, discard_above_or_below in equation_to_criteria.items():
            y = eval_eq(equation, row_mzbin)
            if discard_above_or_below == 'Above':
                if row_mobbin > y:
                    keep_row = False
                    break
            else:
                if row_mobbin < y:
                    keep_row = False
                    break
        if keep_row:
            keep_rows.add(row_index)
    df.drop(index=list(set(df.index.values)-keep_rows), inplace=True)

def process_iteration(start_mob, use_windows, adj_mob_perc, window_mob_ranges, best_tree, windows):
    best = TreeNode(
        uuid4(),
        start_mob,
        0,
        0,
        count_window_by_mob(use_windows[0], start_mob),
        use_windows, 
        adj_mob_perc,
        window_mob_ranges
    )
    bp = best.get_full_tree()
    ret = (None, None)
    if len(bp) < len(windows):
        ret = (None,'None1')
    elif (best_tree is None):
        ret = (best, bp)
    elif (best.num_covered_ions > best_tree.num_covered_ions):
        ret = (best, bp)
    else:
        ret = (None, 'None2')
    return ret

def tree_as_list(root):
    ret = []
    root_dict = root.to_dict()
    ret.append([
        str(root_dict['id']),
        root_dict['mobility_value'],
        root_dict['parent'],
        root_dict['prec'],
        root_dict['depth'],
        ';'.join([str(x) for x in root_dict['children']])
    ])
    for c in root.children:
        ret.extend(tree_as_list(c))
    return ret

def get_wind(window_ind, mobval, windows):
    count = len([x for x in windows[window_ind][0][3] if x <= mobval])
    count += len([x for x in windows[window_ind][1][3] if x >= mobval])
    return count
    
def retrieve_tree(node_dict, df,depth=0):
    retlist = [node_dict]
    if pd.notna(node_dict['parent']):
        retlist.extend(retrieve_tree(df.loc[str(node_dict['parent'])].to_dict(), df, depth+1))
    return retlist

def eval_eq(equation, x):
    return int(MathParser({'x': x}).parse(equation))


def get_eq_id_str(equation):
    rep_dict = {
        '**':'EXP',
        '^': 'DIV',
        '+': 'PLUS',
        '-': 'MINUS',
        '/': 'DIV',
        '*': 'MUL',
        }
    return replace_special_characters(equation, replacewith='_',dict_and_re=True,replacement_dict=rep_dict)


def get_line_y(equation, xvals, yrange):
    points = []
    try:
        for x in xvals:
            y = eval_eq(equation, x)
            if (y >= yrange[0]) and (y < yrange[1]):
                points.append(y)
            else:
                points.append(None)
    except SyntaxError:
        print('invalid syntax')
    except AttributeError:
        print('Wrong attribute')
    return points

def find_val_from(val, fromiter, rev=False):
    ret = -1
    try:
        if val < int(fromiter[0].split('-')[0]):
            ret = 0
        elif val >= int(fromiter[-1].split('-')[1]):
            ret = len(fromiter)
    except ValueError:
        if val < float(fromiter[0].split('-')[0]):
            ret = 0
        elif val >= float(fromiter[-1].split('-')[1]):
            ret = len(fromiter)
    if ret == -1:
        if not rev:
            for i, c in enumerate(fromiter):
                try:
                    low,high = [int(j) for j in c.split('-')]
                except ValueError:
                    low,high = [float(j) for j in c.split('-')]
                if val >= low:
                    if val < high:
                        ret = i
                        break
        else:
            for i, c in enumerate(fromiter[::-1]):
                try:
                    low,high = [int(j) for j in c.split('-')]
                except ValueError:
                    low,high = [float(j) for j in c.split('-')]
                if val >= low:
                    if val < high:
                        ret = len(fromiter)-i-1
                        break
    return ret



def find_defaults(mzmin, mzmax, mobmin, mobmax, df):
    return (
        find_val_from(mzmin, df.columns,rev=False),
        find_val_from(mzmax, df.columns,rev=True),
        find_val_from(mobmin, df.index,rev=False),
        find_val_from(mobmax, df.index,rev=True)   
    )


def coord_to_plotly_rect(y1,y2,x1,x2, df):
    return {
        'type': 'rect',
        'x0': find_val_from(x1, df.columns,rev=False),
        'y0': find_val_from(y1, df.index,rev=False),
        'x1': find_val_from(x2, df.columns,rev=True),
        'y1': find_val_from(y2, df.index,rev=True),
        'line': {'color': 'yellow', 'width': 1},
        'fillcolor': 'rgba(255,255,0,0.5)'
    }

def optimize_window_position(root_node, windows, iterations=5):
    orig_root_node = root_node
    same_count = 0
    for j in range(iterations):
        root_node = orig_root_node
        next_node = root_node.best_child
        parent_node = None
        cumsum_start = root_node.get_tree_sum()
        i = 0
        start = datetime.now()
        while True:
            if i >= len(windows):
                break
            min_mob = 0
            if parent_node:
                min_mob = round(parent_node.mobility_value + 0.01, 2)
            max_mob = windows[i][1][3][-1]
            if next_node:
                max_mob = round(next_node.mobility_value-0.01, 2)
            original = (root_node.mobility_value, root_node.num_ions)
            root_node.optimize_position(min_mob, max_mob, windows[i])
            done = (root_node.mobility_value, root_node.num_ions)
            i += 1
            parent_node = root_node
            root_node = root_node.best_child
            if root_node:
                next_node = root_node.best_child
            else:
                break
        cumsum_end = orig_root_node.get_tree_sum()
        if cumsum_start == cumsum_end:
            same_count +=1
        else:
            same_count = 0
        if same_count >= 2:
            break

def filter_mob_and_mz_matrix(data, mz_bounds, mob_bounds):
    del_rows = []
    del_cols = []
    for c in data.columns:
        if int(c.split('-')[1]) < mz_bounds[0]:
            del_cols.append(c)
        elif int(c.split('-')[0]) > mz_bounds[1]:
            del_cols.append(c)
    for r in data.index:
        if float(r.split('-')[1]) < mob_bounds[0]:
            del_rows.append(r)
        elif float(r.split('-')[0]) > mob_bounds[1]:
            del_rows.append(r)
    for c in del_cols:
        data[c] = np.nan
    for r in del_rows:
        data.loc[r] = np.nan
    data = data.dropna(how='all',axis=1).dropna(how='all',axis=0)
    return data

def filter_mob_and_mz(data, mz_bounds, mob_bounds):
    return data[
        (data['Mz'].between(*mz_bounds, inclusive='both')) & 
        (data['Mobility'].between(*mob_bounds, inclusive = 'both'))
    ]

def ahocorasick_mask(column: pd.Series, exclude: list):
    A = ahocorasick.Automaton(ahocorasick.STORE_INTS)
    for word in exclude:
        A.add_word(word.lower())
    A.make_automaton() 
    column = column.str.lower()
    mask = column.apply(lambda x: bool(list(A.iter(x))))
    return mask

def filter_col(df, col, remove_these, inplace=True):
    rem_vals = set()
    for r in remove_these:
        try:
            rem_vals.add(df[col].dtype.type(r))
        except ValueError:
            continue
    if col == 'Modifications':
        mask = ahocorasick_mask(df[col], remove_these)
    else:
        mask = df[col].isin(rem_vals)
    if inplace:
        df.drop(df.loc[mask].index, inplace=True)
    else:
        return df.drop(df.loc[mask].index)
    