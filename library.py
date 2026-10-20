import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer
import torch.nn.functional as F
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def create_embeddings(ntl_data,  device, d_window = None, cases = None , standardize=True, model = 'Mantis8M', SEQ_LEN=512):
    """
    Calcuates embeddings for each case in the ntl_data using the specified model.
    
    Parameters:
    ntl_data (pd.DataFrame): DataFrame containing the NTL data with columns ['case_id', 'time', 'ntl_filled','timestep', 'ntl_std'].
    device (torch.device): The device  
    
    d_window (int, optional): The window size in days around the event date to consider for embedding calculation. If None, the entire signal is used. Defaults to None.
    cases (pd.DataFrame, optional): DataFrame containing the case information with columns ['case_id', 'event_date', 'n_events', 'event_dates']. Required if d_window is specified.
    standardize (bool, optional): Whether to standardize the signal before embedding calculation. Defaults to True. 
    model (str, optional): The model to use for embedding calculation. Currently only supports 'Mantis8M'. Defaults to 'Mantis8M'.
    SEQ_LEN (int, optional): The length to which the signal should be resampled before embedding calculation. Defaults to 512.

    """ 
    
    
    if model == 'Mantis8M':
        network = Mantis8M(device=device)
        network = network.from_pretrained("paris-noah/Mantis-8M")
        model = MantisTrainer(device=device, network=network)
        
    #case_ids, signals = [], []
    signals = []
    case_ids = dict()
    
    if d_window == None:
        new_ntl = None
        cases_window = None
        #if standardize:
            # Standardize signal per case
         #   ntl_data['ntl_std'] = (ntl_data.groupby('case_id')['ntl_filled']
          #                      .transform(lambda s: StandardScaler().fit_transform(s.to_frame()).ravel()))
            
        for case_id, df in ntl_data.groupby('case_id'):

            signal = df['ntl_std'].values if standardize else df['ntl_filled'].values
            
            # resample to 512 -> (1, 1, 512)
            sig_t = torch.tensor(signal)[None, None, :]
            sig_512 = F.interpolate(sig_t, size=SEQ_LEN, mode="linear", align_corners=False)
            
            case_ids[case_id] = case_id
            signals.append(sig_512[0].numpy())       # (1, 512)
            
    else: 
        
        new_ntl = pd.DataFrame(columns=ntl_data.columns)
        cases_window = cases.copy()
        for case_id, df in ntl_data.groupby('case_id'):
            
            if cases.loc[case_id, 'n_events'] == 1:
                event_date = cases.loc[case_id, 'event_date']
                
                start_date = event_date - pd.Timedelta(days=d_window)
                end_date = event_date + pd.Timedelta(days=d_window)
                
                cases_window.loc[case_id, 'window_start'] = start_date
                cases_window.loc[case_id, 'window_end'] = end_date
                
                df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
                
                #recalculate timestep
                df.loc[:, 'time_step'] = df.groupby('case_id').cumcount()
                df.loc[:, 'ntl_std'] = StandardScaler().fit_transform(df['ntl_filled'].values.reshape(-1, 1)).ravel()

                signal = df['ntl_std'].values if standardize else df['ntl_filled'].values
                
                # resample to 512 -> (1, 1, 512)
                sig_t = torch.tensor(signal)[None, None, :]
                sig_512 = F.interpolate(sig_t, size=SEQ_LEN, mode="linear", align_corners=False)
                
                case_ids[case_id] = case_id
                signals.append(sig_512[0].numpy())       # (1, 512)
                
                new_ntl = pd.concat([new_ntl, df], ignore_index=True)
                
                
            else:
                event_dates = pd.to_datetime(list(set(cases.loc[case_id, 'event_dates'].split(';'))))
                
                cases_window.drop(index=case_id, inplace=True)
                
                for i, event_date in enumerate(event_dates):
                    
                    new_case_id = f"{case_id}_event_{i+1}"
                    start_date = event_date - pd.Timedelta(days=d_window)
                    end_date = event_date + pd.Timedelta(days=d_window)
                    
                    cases_window.loc[new_case_id] = cases.loc[case_id]
                    cases_window.loc[new_case_id, 'event_date'] = event_date
                    cases_window.loc[new_case_id, 'window_start'] = start_date
                    cases_window.loc[new_case_id, 'window_end'] = end_date
                    

                    df_event = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
                    #recalculate timestep
                    df_event.loc[:, 'time_step'] = df_event.groupby('case_id').cumcount()
                    
                    # recalculate the standardized signal for the event window
                    df_event.loc[:, 'ntl_std'] = StandardScaler().fit_transform(df_event['ntl_filled'].values.reshape(-1, 1)).ravel()
                    
                    signal = df_event['ntl_std'].values if standardize else df_event['ntl_filled'].values

                    # resample to 512 -> (1, 1, 512)
                    sig_t = torch.tensor(signal)[None, None, :]
                    sig_512 = F.interpolate(sig_t, size=SEQ_LEN, mode="linear", align_corners=False)

                    case_ids[new_case_id] = case_id
                    signals.append(sig_512[0].numpy())
                    
                    df_event.loc[:, 'case_id'] = new_case_id  # Update the case_id for the event window
                    new_ntl = pd.concat([new_ntl, df_event], ignore_index=True)
                    
        

    X = np.stack(signals)                        # (n_cases, 1, 512)

    # one call for all cases
    Z = model.transform(X)                       # (n_cases, embedding_dim)
    # row i of Z belongs to case_ids[i]
    embeddings = dict(zip(case_ids, Z))
    emb_df = pd.DataFrame(Z, index=pd.Index(case_ids, name='case_id')).add_prefix('ntl_emb_').reset_index()
    
    return case_ids, emb_df, new_ntl, cases_window


def plot_cluster_mean_series(ntl_df, COLORS_DICT,standardize=True, title='Mean NTL signal per cluster (KMeans)' , output_dir="plots/kmeans"):
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in sorted(ntl_df['cluster'].unique()):
        mean_series = (ntl_df[ntl_df['cluster'] == c]
                    .groupby('time_step')['ntl_std'].mean()) if standardize else (ntl_df[ntl_df['cluster'] == c]
                    .groupby('time_step')['ntl_filled'].mean())
        ax.plot(mean_series.index, mean_series.values,color = COLORS_DICT[c], label=f'cluster {c}')
        
        plt.title(title)
        plt.xlabel('Time step')
        plt.ylabel('Mean standardized NTL' if standardize else 'Mean NTL')
        ax.legend()
        #plt.show()
        fig.savefig(f"{output_dir}/mean_ntl_signals.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        
def plot_nbr_cases_per_cluster_per_event(cases_df, EVENT_COLORS_DICT, title='Number of cases per cluster per event type', output_dir="plots/kmeans"):
    plt.figure(figsize=(10, 4))
    sns.countplot(cases_df, x='cluster', hue= 'event_type', stat="percent", palette=EVENT_COLORS_DICT)
    plt.title(title)
    plt.xlabel('Cluster')
    plt.ylabel('Percentage of cases')
    plt.legend(title='Event Type')
    #plt.show()
    plt.savefig(f"{output_dir}/cases_per_cluster.png", dpi=200, bbox_inches="tight")
    
def plot_nbr_cases_per_cluster_per_continent(cases_df, continent_colors, title='Number of cases per cluster per continent', output_dir="plots/kmeans"):
    plt.figure(figsize=(10, 4))
    sns.countplot(cases_df, x='cluster', hue= 'Continent', stat="percent", palette=continent_colors)
    plt.title(title)
    plt.xlabel('Cluster')
    plt.ylabel('Percentage of cases')
    plt.legend(title='Continent')
    #plt.show()
    plt.savefig(f"{output_dir}/cases_per_cluster_per_continent.png", dpi=200, bbox_inches="tight")
    
def plot_country_event_heatmap(cases_df, title='Share of each country\'s cases per cluster \nEvent type: ', output_dir="plots/kmeans"):
    df = cases_df.sort_values(['city', 'country', 'event_type', 'event_date'])
    
    for event_type in df['event_type'].unique():
        sub_df = df[df['event_type'] == event_type]
        ct_pct = pd.crosstab(sub_df['country'], sub_df['cluster'], normalize='index').round(2)
        plt.figure(figsize=(5, max(4, 0.3 * len(ct_pct))))
        sns.heatmap(ct_pct, annot=True, cmap='Blues', vmin=0, vmax=1)
        plt.title(title + event_type)
        plt.show()
        plt.savefig(f"{output_dir}/country_{event_type}_clusters.png", dpi=200, bbox_inches="tight")
        
def plot_time_series_example(ntl_df, cases_df, colors_dict, full_time_series = True, title_detail = '',  output_dir="plots/kmeans"):
    for event_type in cases_df['event_type'].unique():
        for c in sorted(cases_df['cluster'].unique()):
            subset = cases_df[(cases_df['event_type'] == event_type) & (cases_df['cluster'] == c)]
            if len(subset) == 0:
                continue
            case_ids = subset.sample(n=min(4, len(subset)),random_state=0).index
            nrows = min(2, max(round(len(case_ids)/2), 1))
            ncols = min(2, len(case_ids))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows), squeeze=False)
            for i, case_id in enumerate(case_ids):
                ax = axes[i // ncols, i % ncols]
                series = ntl_df[ntl_df['case_id'] == case_id]
                
                ax.plot(series['time'], series['ntl_filled'], color=colors_dict[c],)

                if cases_df.loc[case_id]['n_events'] == 1 or not full_time_series:
                    ax.axvline(cases_df.loc[case_id]['event_date'], color='black', linestyle='--', label='Event date')
                
                else:
                    event_dates = pd.to_datetime(cases_df.loc[case_id]['event_dates'].split(';'))
                    for ed in event_dates:
                        ax.axvline(ed, color='black', linestyle='--', label='Event date')
                ax.set_title(
                    f'{cases_df.loc[case_id]["city"]}, '
                    f'{cases_df.loc[case_id]["country"]}',
                    pad=8
                )

                # Format x-axis dates
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

                # Rotate dates
                ax.tick_params(axis='x', labelrotation=45,labelsize=8)
                
                # Legend
                if i == 0:
                    ax.legend(loc='upper left', fontsize=8)


            fig.suptitle(f'Example NTL signals for cluster {c} and event type: {event_type} \n{title_detail}',y=1.02)

            fig.subplots_adjust( hspace=0.55, wspace=0.20, top=0.88, left=0.08,right=0.98, bottom=0.18)    
            fig.savefig(f"{output_dir}/example_signals_cluster_{c}_{event_type}.png", dpi=200, bbox_inches="tight")

def plot_mean_signal_continent(ntl_df, cases_df, color_dict, continent_colors,standardize,  title='Mean NTL signal per cluster per continent (KMeans)' , output_dir="plots/kmeans"):
    nrows = 3
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows), squeeze=False)
    ntl_df = ntl_df.merge(
                cases_df[['Continent']],
                left_on='case_id',
                right_index=True,
                how='left'
            )
    for i, continent in enumerate(continent_colors.keys()):
        ax = axes[i // ncols, i % ncols]
        ntl_continent = ntl_df[ntl_df['Continent'] == continent]
        if len(ntl_continent) > 0:
            for c in sorted(ntl_continent['cluster'].unique()):
                mean_series = (ntl_continent[ntl_continent['cluster'] == c]
                                    .groupby('time_step')['ntl_std'].mean()) if standardize else (ntl_continent[ntl_continent['cluster'] == c]
                                    .groupby('time_step')['ntl_filled'].mean())
                ax.plot(mean_series.index, mean_series.values,color = color_dict[c], label=f'cluster {c}')
                ax.set_title(continent)
                if i % ncols == 0:
                    ax.set_ylabel('Mean standardized NTL' if standardize else 'Mean NTL')
                if i // ncols == 2:
                    ax.set_xlabel('Time step')
                
        if i == 0:
            ax.legend(loc='upper left', fontsize=8)
                        
    fig.suptitle(title)
    fig.savefig(f'{output_dir}/mean_ntl_signals_continent.png', dpi=200, bbox_inches="tight")
    #fig.xlabel('Time step')
    
            
            #plt.show()
            
            
def run_k_means_and_plot_mean_signal(ntl_data, device, d_window, cases, standardize, k, colors_dict):
    case_ids, emb_df, ntl_window, cases_window = create_embeddings(ntl_data, device=device, d_window=d_window, cases=cases, standardize=standardize, model='Mantis8M', SEQ_LEN=512)

        
    emb_cols = [c for c in emb_df.columns if c.startswith('ntl_emb_')]
    Z = emb_df[emb_cols].to_numpy()
    Z_scaled = StandardScaler().fit_transform(Z)
    pca = PCA(n_components=0.95, random_state=0)
    Z_pca = pca.fit_transform(Z_scaled)  # keep 95% variance
            
            
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    emb_df['cluster'] = km.fit_predict(Z_pca)
                    
    ntl_window = ntl_window.merge(emb_df[['case_id', 'cluster']], on='case_id')
    cases_window = cases_window.merge(emb_df[['case_id', 'cluster']], on='case_id').set_index('case_id')
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in sorted(ntl_window['cluster'].unique()):
        mean_series = (ntl_window[ntl_window['cluster'] == c]
                    .groupby('time_step')['ntl_std'].mean())
        ax.plot(mean_series.index, mean_series.values,color = colors_dict[c], label=f'cluster {c}')
        plt.title(f'Mean NTL signal per cluster (K={k}, W={d_window}, standardize={standardize})')
    plt.xlabel('Time step')
    plt.ylabel('Mean standardized NTL')
    ax.legend()
    plt.show()
    
    
import os
import random
import colorsys
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D


def plot_umap_windows(Z_2d, case_ids, ts_df, point_values, color_map,
                      legend_title, title, w_frac=0.04, h_frac=0.03,
                      n_grid=30, figsize=(14, 12), shared_ylim=True):
    """
    Z_2d         : (n, 2) UMAP coordinates
    case_ids     : (n,) case_id of each row of Z_2d
    ts_df        : DataFrame, index=case_id, columns=time_step (ntl_std)
    point_values : (n,) category of each case (Continent, event_type, country)
    color_map    : dict {category: color}
    """
    x, y = Z_2d[:, 0], Z_2d[:, 1]
    px, py = 0.05 * np.ptp(x), 0.05 * np.ptp(y)
    xmin, xmax = x.min() - px, x.max() + px
    ymin, ymax = y.min() - py, y.max() + py

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    w, h = w_frac * (xmax - xmin), h_frac * (ymax - ymin)
    y_lo, y_hi = np.nanmin(ts_df.values), np.nanmax(ts_df.values)

    # at most one series per grid cell to limit overlap
    gx = np.digitize(x, np.linspace(xmin, xmax, n_grid))
    gy = np.digitize(y, np.linspace(ymin, ymax, n_grid))
    seen, keep = set(), []
    for i, cell in enumerate(zip(gx, gy)):
        if cell not in seen and case_ids[i] in ts_df.index:
            seen.add(cell)
            keep.append(i)

    for i in keep:
        color = color_map.get(point_values[i], 'lightgray')
        ins = ax.inset_axes([x[i] - w / 2, y[i] - h / 2, w, h],
                            transform=ax.transData)
        ins.plot(ts_df.loc[case_ids[i]].values, color=color, lw=1)
        if shared_ylim:
            ins.set_ylim(y_lo, y_hi)
        ins.set_xticks([]); ins.set_yticks([])
        for s in ins.spines.values():
            s.set_edgecolor(color); s.set_linewidth(0.8)
        ins.patch.set_alpha(0.8)

    # legend only for categories actually present
    present = set(point_values)
    handles = [Line2D([0], [0], color=c, lw=2, label=k)
               for k, c in color_map.items() if k in present]
    ax.legend(handles=handles, title=legend_title,
              bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    ax.set_title(f'{title} ({len(keep)}/{len(x)} series shown)')
    return fig, ax


def _draw_windows(ax, x, y, case_ids, ts_df, colors, lims, y_lim,
                  w_frac, h_frac, n_grid, shared_ylim):
    xmin, xmax, ymin, ymax = lims
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    w, h = w_frac * (xmax - xmin), h_frac * (ymax - ymin)

    # at most one series per grid cell to limit overlap
    gx = np.digitize(x, np.linspace(xmin, xmax, n_grid))
    gy = np.digitize(y, np.linspace(ymin, ymax, n_grid))
    seen, keep = set(), []
    for i, cell in enumerate(zip(gx, gy)):
        if cell not in seen and case_ids[i] in ts_df.index:
            seen.add(cell)
            keep.append(i)

    for i in keep:
        color = colors[i]
        ins = ax.inset_axes([x[i] - w / 2, y[i] - h / 2, w, h],
                            transform=ax.transData)
        ins.plot(ts_df.loc[case_ids[i]].values, color=color, lw=1)
        if shared_ylim:
            ins.set_ylim(*y_lim)
        ins.set_xticks([]); ins.set_yticks([])
        for s in ins.spines.values():
            s.set_edgecolor(color); s.set_linewidth(0.8)
        ins.patch.set_alpha(0.8)
    return len(keep)


def plot_umap_windows_by_event(Z_2d, case_ids, ts_df, event_values,
                               continent_values, continent_colors,
                               event_order, title, ncols=2,
                               w_frac=0.06, h_frac=0.05, n_grid=25,
                               subplot_size=(8, 7), shared_ylim=True):
    """
    One subplot per event type; mini time series colored by continent.
    Z_2d, case_ids, event_values, continent_values : arrays of length n
    """
    case_ids = np.asarray(case_ids)
    event_values = np.asarray(event_values)
    continent_values = np.asarray(continent_values)
    x, y = Z_2d[:, 0], Z_2d[:, 1]

    # common limits so every subplot has the same frame
    px, py = 0.05 * np.ptp(x), 0.05 * np.ptp(y)
    lims = (x.min() - px, x.max() + px, y.min() - py, y.max() + py)
    y_lim = (np.nanmin(ts_df.values), np.nanmax(ts_df.values))

    point_colors = np.array([continent_colors.get(c, 'lightgray')
                             for c in continent_values], dtype=object)

    nrows = int(np.ceil(len(event_order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True,
                             figsize=(subplot_size[0] * ncols,
                                      subplot_size[1] * nrows),
                             squeeze=False)
    axes = axes.ravel()

    for ax, ev in zip(axes, event_order):
        m = event_values == ev
        n_shown = _draw_windows(
            ax, x[m], y[m], case_ids[m], ts_df, point_colors[m],
            lims, y_lim, w_frac, h_frac, n_grid, shared_ylim)
        ax.set_title(f'{ev} ({n_shown}/{m.sum()} series shown)')
        ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    for ax in axes[len(event_order):]:
        ax.axis('off')

    present = set(continent_values)
    handles = [Line2D([0], [0], color=c, lw=2, label=k)
               for k, c in continent_colors.items() if k in present]
    fig.legend(handles=handles, title='Continent',
               loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig, axes

def calculate_window(ntl_data, cases, d_window):
    case_ids = dict()
    frames = []                       # collect pieces, concat once at the end
    cases_window = cases.copy()

    # plain Python float -> no NumPy "generic unit" timedelta warning
    window = pd.Timedelta(days=float(d_window))

    def _rescale(frame):
        frame = frame.copy()          # avoids SettingWithCopyWarning
        frame['time_step'] = frame.groupby('case_id').cumcount()
        frame['ntl_std'] = StandardScaler().fit_transform(
            frame['ntl_filled'].values.reshape(-1, 1)
        ).ravel()
        return frame

    for case_id, df in ntl_data.groupby('case_id'):
        if cases.loc[case_id, 'n_events'] == 1:
            event_date = cases.loc[case_id, 'event_date']
            start_date = event_date - window
            end_date = event_date + window

            cases_window.loc[case_id, 'window_start'] = start_date
            cases_window.loc[case_id, 'window_end'] = end_date

            df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
            case_ids[case_id] = case_id

            if not df.empty and not df.isna().all().all():
                frames.append(_rescale(df))

        else:
            event_dates = pd.to_datetime(
                list(set(cases.loc[case_id, 'event_dates'].split(';')))
            )
            cases_window.drop(index=case_id, inplace=True)

            for i, event_date in enumerate(event_dates):
                new_case_id = f"{case_id}_event_{i+1}"
                start_date = event_date - window
                end_date = event_date + window

                cases_window.loc[new_case_id] = cases.loc[case_id]
                cases_window.loc[new_case_id, 'event_date'] = event_date
                cases_window.loc[new_case_id, 'window_start'] = start_date
                cases_window.loc[new_case_id, 'window_end'] = end_date

                case_ids[new_case_id] = case_id

                df_event = df[(df['time'] >= start_date) & (df['time'] <= end_date)]

                if not df_event.empty and not df_event.isna().all().all():
                    df_event = df_event.copy()
                    df_event['case_id'] = new_case_id   # set before cumcount
                    frames.append(_rescale(df_event))

    new_ntl = (pd.concat(frames, ignore_index=True)
               if frames else pd.DataFrame(columns=ntl_data.columns))
    
    cases_window['original_id'] = cases_window.index.map(case_ids)

    return case_ids, new_ntl, cases_window