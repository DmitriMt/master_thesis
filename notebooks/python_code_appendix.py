# Library imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS

# Data import
panel = pd.read_parquet("../data/working_data.parquet")

# Plot average wages year comparison
df_wages = (
    panel[["year", "has_disablity", "wages"]]
    .groupby(["year", "has_disablity"])
    .mean()
    .reset_index()
)
df_wages["year"] = pd.to_datetime(df_wages["year"], format="%Y")
df_wages["has_disablity"] = df_wages["has_disablity"].map({0: "No", 1: "Yes"})
df_wages = df_wages.pivot(index="year", columns="has_disablity", values="wages")

colors = {"Yes": "steelblue", "No": "lightblue"}
fig, ax = plt.subplots(figsize=(12, 5))

for column in df_wages.columns:
    ax.plot(
        df_wages.index,
        df_wages[column],
        marker="o",
        linewidth=2.5,
        markersize=8,
        color=colors.get(column, "#3498db"),
        label=f"Has Disability: {column}",
        markeredgecolor="white",
        markeredgewidth=1.5,
        alpha=0.9,
    )

ax.set_title(
    f"average Wages by year comparison",
    fontsize=16,
    fontweight="bold",
    pad=20,
    # fontfamily='serif'
)
ax.axvline(
    x=pd.Timestamp("2019-01-01"),
    color="gray",
    linestyle="--",
    linewidth=1.5,
    label="period of interest",
    alpha=0.7,
)

ax.axvline(
    x=pd.Timestamp("2020-01-01"), color="gray", linestyle="--", linewidth=1.5, alpha=0.7
)
ax.set_xlabel("year", fontsize=12, fontweight="semibold", labelpad=10)
ax.set_ylabel(f"average Wages", fontsize=12, fontweight="semibold", labelpad=10)

legend = ax.legend(
    title="Has Disability",
    title_fontsize=12,
    fontsize=11,
    loc="best",
    frameon=True,
    fancybox=True,
    shadow=True,
    edgecolor="#cccccc",
    facecolor="white",
)
legend.get_title().set_fontweight("bold")

# Data manipulation
panel_2019_2020 = panel.copy()
panel_2019_2020["year"] = pd.to_datetime(panel_2019_2020["year"], format="%Y")
panel_2019_2020_ids = panel_2019_2020["idind"]
panel_2019_2020 = panel_2019_2020.set_index(["idind", "year"])

# Serial correlation test computation
fixed_effects_ind_model = PanelOLS.from_formula(
    "wages ~ period + EntityEffects", data=panel_2019_2020
).fit()
print(fixed_effects_ind_model.summary)

fe_resid = fe_model.resids

resid_df = pd.DataFrame(
    {
        "idind": panel_2019_2020.reset_index()["idind"].values,
        "year": panel_2019_2020.reset_index()["year"].values,
        "resid": fe_resid.values,
    }
).sort_values(["idind", "year"])


resid_df["resid_diff"] = resid_df.groupby("idind")["resid"].diff()
resid_df["resid_diff_lag"] = resid_df.groupby("idind")["resid_diff"].shift(1)
resid_df = resid_df.dropna()

model_ols = smf.ols("resid_diff ~ resid_diff_lag", resid_df).fit(cov_type="HC1")
print(model_ols.summary2())


# Difference in differences TWFE model computation
model_twfe = PanelOLS.from_formula(
    "wages ~ I(has_disability * period) + EntityEffects + TimeEffects",
    data=panel_2019_2020,
).fit(cov_type="clustered", cluster_entity=True, group_debias=False)
print(model_twfe.summary)

# Dynamic difference in differences TWFE model computation
panel = pl.read_parquet("../data/working_data_multiple_periods.parquet")
panel_2014_2024 = (
    panel.filter(
        (pl.col("period_relevance") > -5) & (pl.col("period_relevance") < 5)
    ).with_columns(pl.col("year").cast(pl.Int32).cast(pl.String).str.to_date("%Y"))
).to_pandas()
panel_2014_2024_dta = panel_2014_2024.set_index(["idind", "year"])

event_study_terms = []
for p in sorted(panel_2014_2024_dta["period_relevance"].unique()):
    if p != 0:
        col_name = f"rel_period_{p}_treated"
        panel_2014_2024_dta[col_name] = (
            (panel_2014_2024_dta["period_relevance"].values == p)
            & (panel_2014_2024_dta["has_disability_period"].values == 1)
        ).astype(int)
        event_study_terms.append(col_name)

X_event = panel_2014_2024_dta[event_study_terms]
y_event = panel_2014_2024_dta["wages"]

event_study_model = PanelOLS(
    y_event, X_event, entity_effects=True, check_rank=True
).fit(cov_type="clustered", cluster_entity=True)
print(event_study_model.summary)


# Dynamic difference in differences TWFE model plot
def plot_event_margins(model, measure_unit="RUB"):
    coef_names = [
        col for col in model.params.index if col.startswith("period_relevance")
    ]
    coefs = model.params[coef_names]
    std_errors = model.std_errors[coef_names]

    rel_periods = [int(re.search(r"-?\d+", name).group()) for name in coef_names]
    coef_df = pd.DataFrame(
        {
            "relative_period": rel_periods,
            "estimate": coefs.values,
            "std_error": std_errors.values,
        }
    )

    coef_df["conf_low"] = coef_df["estimate"] - 1.96 * coef_df["std_error"]
    coef_df["conf_high"] = coef_df["estimate"] + 1.96 * coef_df["std_error"]

    coef_df["type"] = np.where(
        coef_df["relative_period"] < 0, "Pre-treatment", "Post-treatment"
    )

    ref_row = pd.DataFrame(
        {
            "relative_period": [-1],
            "estimate": [0],
            "std_error": [0],
            "conf_low": [0],
            "conf_high": [0],
            "type": ["Pre-treatment"],
        }
    )
    coef_df = pd.concat([coef_df, ref_row], ignore_index=True)
    coef_df = coef_df.sort_values("relative_period").reset_index(drop=True)

    max_effect_row = coef_df.loc[coef_df["estimate"].idxmax()]
    max_conf_high = coef_df["conf_high"].max()
    min_conf_low = coef_df["conf_low"].min()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axvspan(-6.5, 0, alpha=0.15, color=book_colors["light_gray"], zorder=0)
    ax.axvspan(3, 5, alpha=0.15, color=book_colors["light_gray"], zorder=0)
    ax.axhline(
        y=0, linestyle="--", color=book_colors["muted"], linewidth=0.8, alpha=0.8
    )
    ax.axvline(
        x=0, linestyle="--", color=book_colors["muted"], linewidth=0.8, alpha=0.8
    )
    ax.axvline(
        x=3, linestyle="--", color=book_colors["muted"], linewidth=0.8, alpha=0.8
    )
    ax.errorbar(
        coef_df["relative_period"],
        coef_df["estimate"],
        yerr=[
            coef_df["estimate"] - coef_df["conf_low"],
            coef_df["conf_high"] - coef_df["estimate"],
        ],
        fmt="o",
        color=book_colors["primary"],
        markersize=6,
        capsize=4,
        capthick=1.5,
        linewidth=0.8,
        ecolor=book_colors["primary"],
        elinewidth=0.8,
        zorder=3,
    )

    annotation_y = max_conf_high * 0.7 if max_conf_high > 0 else max_conf_high * 0.3
    ax.annotate(
        "pre-pandemic",
        xy=(-1.5, annotation_y),
        fontsize=13,
        color=book_colors["muted"],
        ha="center",
        style="italic",
    )
    ax.annotate(
        "pandemic",
        xy=(1.5, annotation_y),
        fontsize=13,
        color=book_colors["primary"],
        ha="center",
        style="italic",
    )
    ax.annotate(
        "post-pandemic",
        xy=(4.5, annotation_y),
        fontsize=13,
        color=book_colors["muted"],
        ha="center",
        style="italic",
    )
    ax.set_xlabel(
        "period relative to the event", fontsize=14, color=book_colors["dark_gray"]
    )
    ax.set_ylabel(
        f"estimated effect ({measure_unit})",
        fontsize=14,
        color=book_colors["dark_gray"],
    )
    ax.set_title(
        "effect of COVID-19 on earnings over time",
        fontsize=16,
        fontweight="bold",
        color="#333333",
        pad=15,
    )
    ax.text(
        0.5,
        1.02,
        "period relevance k = -1 (2019)",
        transform=ax.transAxes,
        fontsize=11,
        color="grey",
        ha="center",
    )
    ax.set_xticks(range(-5, 5))
    ax.set_xlim(-6.5, 5)
    y_padding = (
        (max_conf_high - min_conf_low) * 0.2 if max_conf_high != min_conf_low else 1
    )
    ax.set_ylim(min_conf_low - y_padding, max_conf_high + y_padding)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(book_colors["muted"])
    ax.spines["bottom"].set_color(book_colors["muted"])
    ax.tick_params(colors=book_colors["dark_gray"], labelsize=11)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.show()

    return fig, ax, coef_df