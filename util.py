import numpy as np
import polars as pl
import tabulate

tabulate.PRESERVE_WHITESPACE = True

def line_hist(data, bins=16): 
    bars = " ▁▂▃▄▅▆▇█"
    n, _ = np.histogram(data, bins=bins)
    n2 = n * (len(bars) - 1) // (max(n))
    res = "".join(bars[i] for i in n2)
    return res


def summarize(df: pl.DataFrame):
    metadata = [
        ("column", df.columns),
        ("dtype", [pl.datatypes.dtype_to_ffiname(dt) for dt in df.dtypes]),
    ]
    
    pl_summaries = [
        ("mean", df.mean()),
        ("std", df.std()),
        ("5.5%", df.quantile(0.055, "linear")),
        ("94.5%", df.quantile(0.945, "linear")),
        
    ]
    if sum(df.null_count().row(0)) > 0:
        pl_summaries.append(("% null", df.null_count() * 100.0 / df.shape[0]))
        
    summaries = metadata + [(n, vs.row(0)) for n, vs in pl_summaries]
    summaries.append(("histogram", [line_hist(df[c].drop_nulls()) for c in df.columns]))
    summary = tabulate.tabulate(dict(summaries), headers='keys', tablefmt="fancy_grid")
    return f"pl.DataFrame of shape {df.shape}\n\n" + summary

def covariance(df: pl.DataFrame):
    cov = np.cov(np.stack([df[c] for c in df.columns], axis=0)) 
    return pl.DataFrame({'name': df.columns, **{c: row for c, row in zip(df.columns, cov)}} )