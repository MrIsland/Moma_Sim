# Function to plot joint position / velocity trends and return summary statistics.
# This will produce one plot per requested signal type (e.g. "position" and/or "velocity").
# The function accepts:
#  - data: pandas.DataFrame OR dict OR numpy array
#      * If DataFrame: columns are joint names, index is timestamps (optional).
#      * If dict: keys are joint names, values are 1D arrays (all same length). OR keys 'position'/'velocity' mapping to DataFrames/dicts/arrays.
#      * If numpy array: shape (N, J) and you must pass `joints` list.
#  - timestamps: optional 1D array-like to use as x-axis if DataFrame/index not present.
#  - joints: optional list of joint names (required for numpy array input).
#  - kinds: 'position', 'velocity', or ['position','velocity'] if you pass a dict with those keys.
#  - save_path: optional path to save each figure (PNG).
#  - show: whether to display the plot (True).
#
# The function returns a dictionary of pandas.DataFrames with summary statistics for each plotted kind.
#
# NOTE: matplotlib is used (no seaborn); each kind gets its own figure (no subplots).
from typing import Union, List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _to_dataframe(obj, joints=None, timestamps=None):
    """Internal helper: convert input into pandas.DataFrame where columns are joint names.
    Accepts DataFrame, dict of arrays/series, or numpy array with joints provided."""
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, dict):
        # If dict values are arrays/series, convert to DataFrame directly
        return pd.DataFrame({k: np.asarray(v).ravel() for k, v in obj.items()})
    arr = np.asarray(obj)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if joints is None:
        # create default joint names
        joints = [f"joint_{i}" for i in range(arr.shape[1])]
    if len(joints) != arr.shape[1]:
        raise ValueError(f"Number of joint names ({len(joints)}) does not match array width ({arr.shape[1]}).")
    df = pd.DataFrame(arr, columns=joints)
    if timestamps is not None:
        df.index = pd.Index(timestamps, name='time')
    return df


def plot_joint_trends(
        data: Union[pd.DataFrame, dict, np.ndarray],
        timestamps: Union[None, List[float], np.ndarray] = None,
        joints: Union[None, List[str]] = None,
        kinds: Union[str, List[str]] = "position",
        save_path: Union[None, str] = None,
        show: bool = True,
        figsize=(10, 4)
) -> Dict[str, pd.DataFrame]:
    """
    绘制关节 position/velocity 的折线图并返回统计信息。

    参数说明（中文）:
    - data: 支持三种形式：
        1) pandas.DataFrame：每列为一个关节，索引为时间（可选）
        2) dict: {joint_name: array_like, ...} 或 {'position': <df/dict/array>, 'velocity': <...>}
        3) numpy.ndarray: shape (N, J) —— 需要提供 joints 列表
    - timestamps: x 轴时间戳（可选）。如果 data 是 DataFrame 且有 index，会优先使用 index。
    - joints: 仅在传入 ndarray 时需要；也可用来重命名/选择列。
    - kinds: 'position' / 'velocity' / ['position','velocity'] 。如果 data 是 dict 且包含这些键，会分别绘制。
    - save_path: 如果指定，会把每个图保存为 <save_path>_position.png 或 _velocity.png
    - show: 是否显示图像
    - figsize: 图像大小

    返回:
    - 一个 dict，键为每个绘制的 kind，值为对应的统计信息 DataFrame（每行是关节，列为 count, mean, std, min, 25%, 50%, 75%, max）。
    """
    # Normalize kinds to list
    if isinstance(kinds, str):
        kinds = [kinds]
    results = {}

    # If data is a dict that contains kinds as keys (position/velocity), handle separately
    # if isinstance(data, dict) and any(k in data for k in ['position', 'velocity']):
    if isinstance(data, dict):
        # For each requested kind, extract that sub-data
        for k in kinds:
            if k not in data:
                print(f"Warning: requested kind '{k}' not found in data dict; skipping.")
                continue
            df = _to_dataframe(data[k], joints=joints, timestamps=timestamps)
            # If timestamps provided, set index
            if timestamps is not None and df.shape[0] == len(timestamps):
                df.index = pd.Index(timestamps, name='time')
            stats = df.describe().T  # statistics per column
            results[k] = stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
            # Plot
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_title(f"Joint {k} trends")
            ax.set_xlabel("time" if df.index.name is not None else "sample")
            ax.set_ylabel(k)
            # plot each joint on same axes (no color specification as requested)
            for col in df.columns:
                ax.plot(df.index.values if df.index is not None else np.arange(df.shape[0]), df[col].values, label=col)
            ax.legend(loc='best', fontsize='small')
            ax.grid(True, linestyle=':', linewidth=0.5)
            if save_path:
                fname = f"{save_path}_{k}.png"
                fig.savefig(fname, bbox_inches='tight')
                print(f"Saved figure: {fname}")
            if show:
                plt.show()
            plt.close(fig)
    else:
        # Treat data as single table (DataFrame/array/dict of joint->series)
        df = _to_dataframe(data, joints=joints, timestamps=timestamps)
        if timestamps is not None and df.shape[0] == len(timestamps):
            df.index = pd.Index(timestamps, name='time')
        stats = df.describe().T
        # Use provided kinds list but we only have one set of signals - use the first kind name for labeling
        kind_label = kinds[0] if kinds else "signal"
        results[kind_label] = stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"{kind_label} trends")
        ax.set_xlabel("time" if df.index.name is not None else "sample")
        ax.set_ylabel(kind_label)
        for col in df.columns:
            ax.plot(df.index.values if df.index is not None else np.arange(df.shape[0]), df[col].values, label=col)
        ax.legend(loc='best', fontsize='small')
        ax.grid(True, linestyle=':', linewidth=0.5)
        if save_path:
            fname = f"{save_path}_{kind_label}.png"
            fig.savefig(fname, bbox_inches='tight')
            print(f"Saved figure: {fname}")
        if show:
            plt.show()
        plt.close(fig)

    # Display summary tables to user using dataframe viewer if available
    try:
        from caas_jupyter_tools import display_dataframe_to_user
        for k, df_stats in results.items():
            display_dataframe_to_user(f"summary_{k}", df_stats.reset_index().rename(columns={'index': 'joint'}))
    except Exception:
        # If display tool not available, just print small summary
        for k, df_stats in results.items():
            print(f"\nSummary statistics for '{k}':")
            print(df_stats.head(20).to_string())

    return results


# ------------------------
# Example usage (demo data). Run to show sample plots.
if __name__ == "__main__":
    # demo: 4 joints, 200 samples, sinusoidal positions + derivative as velocity
    t = np.linspace(0, 10, 200)
    joints = ['hip', 'knee', 'ankle', 'toe']
    pos = np.vstack([np.sin(t * freq) for freq in [1.0, 1.5, 0.7, 0.4]]).T  # (200, 4)
    vel = np.gradient(pos, t, axis=0)
    data = {'position': pd.DataFrame(pos, columns=joints, index=t),
            'velocity': pd.DataFrame(vel, columns=joints, index=t)}
    summaries = plot_joint_trends(data, kinds=['position', 'velocity'], save_path=None, show=True)
    print("Returned summaries keys:", summaries.keys())

