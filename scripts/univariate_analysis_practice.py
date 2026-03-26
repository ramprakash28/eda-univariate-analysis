import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

OUTPUT_DIR = "."
import os


def univariate_numerical(df, column, show_plot=True, save_prefix=None):
    col = df[column].dropna()

    print(f"\n{'='*65}")
    print(f"  NUMERICAL UNIVARIATE ANALYSIS: {column}")
    print(f"{'='*65}")

    total = len(df[column])
    missing = df[column].isna().sum()
    print(f"\n DATA QUALITY")
    print(f"   Total records:    {total:,}")
    print(f"   Missing values:   {missing:,} ({missing/total*100:.1f}%)")
    print(f"   Valid records:    {len(col):,}")
    print(f"   Unique values:    {col.nunique():,}")

    mean_val = col.mean()
    median_val = col.median()
    mode_val = col.mode().values

    print(f"\n CENTRAL TENDENCY")
    print(f"   Mean:    {mean_val:,.2f}")
    print(f"   Median:  {median_val:,.2f}")
    print(f"   Mode:    {mode_val}")

    if abs(mean_val - median_val) / median_val > 0.1:
        direction = "right" if mean_val > median_val else "left"
        print(f"   Mean and Median differ by >10% - likely {direction}-skewed")
    else:
        print(f"   Mean ~ Median - approximately symmetric")

    std_val = col.std()
    var_val = col.var()
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    cv = (std_val / mean_val) * 100 if mean_val != 0 else float('inf')

    print(f"\n SPREAD (DISPERSION)")
    print(f"   Std Deviation:  {std_val:,.2f}")
    print(f"   Variance:       {var_val:,.2f}")
    print(f"   Range:          {col.min():,.2f}  to  {col.max():,.2f}")
    print(f"   IQR:            {IQR:,.2f}  (Q1={Q1:,.2f}, Q3={Q3:,.2f})")
    print(f"   CV:             {cv:.1f}%")

    print(f"\n PERCENTILES")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"   {p:>3}th:  {col.quantile(p/100):,.2f}")

    skew = col.skew()
    kurt = col.kurtosis()

    print(f"\n SHAPE")
    print(f"   Skewness:  {skew:.3f}", end="")
    if abs(skew) < 0.5:
        print("  -> Approximately symmetric")
    elif skew > 0:
        print(f"  -> Right-skewed ({'moderately' if skew < 1 else 'highly'})")
    else:
        print(f"  -> Left-skewed ({'moderately' if abs(skew) < 1 else 'highly'})")

    print(f"   Kurtosis:  {kurt:.3f}", end="")
    if abs(kurt) < 0.5:
        print("  -> Mesokurtic (similar to normal)")
    elif kurt > 0:
        print("  -> Leptokurtic (heavy tails, more outliers)")
    else:
        print("  -> Platykurtic (light tails, fewer outliers)")

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_iqr = col[(col < lower_bound) | (col > upper_bound)]

    z_scores = (col - mean_val) / std_val
    outliers_z = col[z_scores.abs() > 3]

    print(f"\n OUTLIER DETECTION")
    print(f"   IQR Method (1.5xIQR):")
    print(f"     Bounds:  [{lower_bound:,.2f}, {upper_bound:,.2f}]")
    print(f"     Outliers: {len(outliers_iqr):,} ({len(outliers_iqr)/len(col)*100:.1f}%)")
    print(f"   Z-Score Method (|z| > 3):")
    print(f"     Outliers: {len(outliers_z):,} ({len(outliers_z)/len(col)*100:.1f}%)")

    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Univariate Analysis: {column}', fontsize=16, fontweight='bold', y=1.02)

        ax = axes[0, 0]
        sns.histplot(col, bins=min(50, max(20, len(col)//100)), kde=True, ax=ax, color='steelblue')
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:,.1f}')
        ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:,.1f}')
        ax.legend(fontsize=9)
        ax.set_title('Histogram + KDE', fontweight='bold')
        ax.set_xlabel(column)

        ax = axes[0, 1]
        sns.boxplot(x=col, ax=ax, color='lightcoral')
        ax.set_title('Box Plot (Outlier Detection)', fontweight='bold')
        ax.set_xlabel(column)

        ax = axes[1, 0]
        sns.violinplot(x=col, ax=ax, color='mediumpurple', inner='quartile')
        ax.set_title('Violin Plot (Distribution Shape)', fontweight='bold')
        ax.set_xlabel(column)

        ax = axes[1, 1]
        sns.ecdfplot(col, ax=ax, color='darkorange')
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(median_val, color='green', linestyle='--', alpha=0.5, label=f'Median: {median_val:,.1f}')
        ax.legend(fontsize=9)
        ax.set_title('ECDF (Cumulative Distribution)', fontweight='bold')
        ax.set_xlabel(column)

        plt.tight_layout()
        if save_prefix:
            filepath = os.path.join(OUTPUT_DIR, f'{save_prefix}_{column}_numerical.png')
            plt.savefig(filepath, dpi=150)
            print(f"\n   Plot saved: {filepath}")
        plt.close()

    return {
        'mean': mean_val, 'median': median_val, 'std': std_val,
        'skew': skew, 'kurtosis': kurt, 'iqr_outliers': len(outliers_iqr)
    }


def univariate_categorical(df, column, top_n=10, show_plot=True, save_prefix=None):
    col = df[column].dropna()

    print(f"\n{'='*65}")
    print(f"  CATEGORICAL UNIVARIATE ANALYSIS: {column}")
    print(f"{'='*65}")

    total = len(df[column])
    missing = df[column].isna().sum()
    n_unique = col.nunique()

    print(f"\n DATA QUALITY")
    print(f"   Total records:    {total:,}")
    print(f"   Missing values:   {missing:,} ({missing/total*100:.1f}%)")
    print(f"   Unique categories: {n_unique}")

    freq = col.value_counts()
    freq_pct = col.value_counts(normalize=True) * 100

    print(f"\n FREQUENCY TABLE (Top {min(top_n, n_unique)})")
    print(f"   {'Category':<30} {'Count':>8} {'Percent':>8} {'Cumul%':>8}")
    print(f"   {'---'*18}")

    cumulative = 0
    for i, (cat, count) in enumerate(freq.head(top_n).items()):
        pct = freq_pct[cat]
        cumulative += pct
        print(f"   {str(cat):<30} {count:>8,} {pct:>7.1f}% {cumulative:>7.1f}%")

    mode_val = col.mode().values[0]
    mode_freq = freq.iloc[0]
    mode_pct = freq_pct.iloc[0]

    print(f"\n KEY METRICS")
    print(f"   Mode: '{mode_val}' ({mode_freq:,} occurrences, {mode_pct:.1f}%)")

    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Univariate Analysis: {column}', fontsize=16, fontweight='bold')

        ax = axes[0]
        top_data = freq.head(top_n)
        colors = sns.color_palette("viridis", len(top_data))
        ax.barh(range(len(top_data)), top_data.values, color=colors)
        ax.set_yticks(range(len(top_data)))
        ax.set_yticklabels(top_data.index)
        ax.invert_yaxis()
        ax.set_xlabel('Count')
        ax.set_title(f'Top {min(top_n, n_unique)} Categories (Bar Chart)', fontweight='bold')

        ax = axes[1]
        top5 = freq.head(5)
        if n_unique > 5:
            other = pd.Series({'Other': freq.iloc[5:].sum()})
            top5 = pd.concat([top5, other])

        wedges, texts, autotexts = ax.pie(
            top5.values,
            labels=top5.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("Set2", len(top5)),
            pctdistance=0.75
        )
        centre_circle = plt.Circle((0, 0), 0.50, fc='white')
        ax.add_artist(centre_circle)
        ax.set_title('Distribution (Donut Chart)', fontweight='bold')

        plt.tight_layout()
        if save_prefix:
            filepath = os.path.join(OUTPUT_DIR, f'{save_prefix}_{column}_categorical.png')
            plt.savefig(filepath, dpi=150)
            print(f"\n   Plot saved: {filepath}")
        plt.close()

    return {'mode': mode_val, 'n_unique': n_unique, 'mode_pct': mode_pct}
