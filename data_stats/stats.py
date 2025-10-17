import pandas as pd

df = pd.read_csv("data_stats.csv")

# get the df where recon == "ref"
df = df[df["recon"] == "ref"]

# tracer mapping dictionary
tracer_mapping = {
    "FDG": "FDG",
    "Fluorodeoxyglucose": "FDG",
    "DOTA": "non-FDG",
    "GA68": "non-FDG",
    "Solution": "FDG",
}

# Apply the mapping to create a new column 'tacer_type'
df["tracer_type"] = df["tracer"].map(tracer_mapping)

screening_df = pd.read_csv("../screening/mips_screening_cc.csv")

# merge the cats
df = df.merge(screening_df[["acq", "cat"]], on="acq", how="left")


################################################################################
uexp_fdg_df = df[
    df["cat"].notna() & (df["tracer_type"] == "FDG") & (df["scanner"] == "uEXPLORER")
]

# keep only those row where cat > 0, or the 83 first rows where cat is 0
uexp_fdg_df_cat_gt_0 = uexp_fdg_df[uexp_fdg_df["cat"] > 0]
uexp_fdg_df_cat_eq_0 = uexp_fdg_df[uexp_fdg_df["cat"] == 0].head(83)
uexp_fdg_df_filtered = pd.concat(
    [uexp_fdg_df_cat_gt_0, uexp_fdg_df_cat_eq_0], ignore_index=True
)

# 90% per cat into train
uexp_fdg_df_train = uexp_fdg_df_filtered.groupby(
    "cat", group_keys=False, observed=True
).apply(lambda g: g.sample(frac=0.9, random_state=42))

# remaining 10% per cat into val
uexp_fdg_df_val = uexp_fdg_df_filtered.drop(uexp_fdg_df_train.index).copy()

print(
    uexp_fdg_df_train.groupby(["tracer_type", "scanner", "cat"])
    .size()
    .reset_index(name="count")
)

print(
    uexp_fdg_df_val.groupby(["tracer_type", "scanner", "cat"])
    .size()
    .reset_index(name="count")
)
################################################################################
# do the same for comibnation tracer_type == "FDG" and scanner == "Biograph128_Vision Quadra Edge"
biograph_fdg_df = df[
    df["cat"].notna()
    & (df["tracer_type"] == "FDG")
    & (df["scanner"] == "Biograph128_Vision Quadra Edge")
]
# keep only those row where cat > 0, or the 83 first rows where cat is 0
biograph_fdg_df_cat_gt_0 = biograph_fdg_df[biograph_fdg_df["cat"] > 0]
biograph_fdg_df_cat_eq_0 = biograph_fdg_df[biograph_fdg_df["cat"] == 0].head(50)
biograph_fdg_df_filtered = pd.concat(
    [biograph_fdg_df_cat_gt_0, biograph_fdg_df_cat_eq_0], ignore_index=True
)


# 90% per cat into train
biograph_fdg_df_train = biograph_fdg_df_filtered.groupby(
    "cat", group_keys=False, observed=True
).apply(lambda g: g.sample(frac=0.9, random_state=42))

# remaining 10% per cat into val
biograph_fdg_df_val = biograph_fdg_df_filtered.drop(biograph_fdg_df_train.index).copy()

print(
    biograph_fdg_df_train.groupby(["tracer_type", "scanner", "cat"])
    .size()
    .reset_index(name="count")
)

print(
    biograph_fdg_df_val.groupby(["tracer_type", "scanner", "cat"])
    .size()
    .reset_index(name="count")
)
################################################################################
# get all non-FDG tracers
uexp_nonfdg_df = df[df["tracer_type"] == "non-FDG"]

# take first 90% into train
uexp_nonfdg_df_train = uexp_nonfdg_df.sample(frac=0.8, random_state=42)
# remaining 10% into val
uexp_nonfdg_df_val = uexp_nonfdg_df.drop(uexp_nonfdg_df_train.index).copy()

print(
    uexp_nonfdg_df_train.groupby(["tracer_type", "scanner"])
    .size()
    .reset_index(name="count")
)

print(
    uexp_nonfdg_df_val.groupby(["tracer_type", "scanner"])
    .size()
    .reset_index(name="count")
)
