import numpy as np
import polars as pl

def line_hist(data, bins=16):
    bars = " ▁▂▃▄▅▆▇█"
    n, _ = np.histogram(data, bins=bins)
    n2 = n * (len(bars) - 1) // (max(n))
    res = "".join(bars[i] for i in n2)
    return res


def summarize(df: pl.DataFrame):
    metadata = {
        "column": df.columns,
        "dtype": [pl.datatypes.dtype_to_ffiname(dt) for dt in df.dtypes],
    }
    summaries = [
        ("median", df.median()),
        ("mean", df.mean()),
        ("std", df.std()),
        ("5.5%", df.quantile(0.055, "linear")),
        ("94.5%", df.quantile(0.945, "linear")),
        ("histogram", df.transpose().apply(line_hist).transpose()),
    ]
    summaries = [(name, vs.transpose()) for (name, vs) in summaries]
    summaries = [
        vs.with_column_renamed(vs.columns[0], name) for (name, vs) in summaries
    ]
    summaries = [pl.DataFrame(metadata)] + summaries
    summary = str(pl.concat(summaries, how="horizontal"))
    summary = summary[summary.index("\n") + 1 :]  # Skip initial "shape: (...)" header.
    return f"[pl.DataFrame of shape {df.shape}:\n{summary}"


def covariance(df: pl.DataFrame):
    cov = np.cov(np.stack([df[c] for c in df.columns], axis=0)) 
    return pl.DataFrame({'name': df.columns, **{c: row for c, row in zip(df.columns, cov)}} )