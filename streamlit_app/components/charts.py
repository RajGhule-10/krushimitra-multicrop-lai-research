from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

GREEN = "#1F5A45"
SAGE = "#739A7A"
GOLD = "#A56F2A"
SLATE = "#24332D"
SECONDARY = "#58665F"
BORDER = "#DCE4DD"


def base_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=56, r=24, t=68, b=70),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color=SLATE, size=13),
        title=dict(font=dict(color=SLATE, size=17), x=0.02, xanchor="left"),
        legend_title_text="",
        legend=dict(font=dict(color=SLATE), bgcolor="rgba(255,255,255,0.85)", bordercolor=BORDER, borderwidth=1),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color=SLATE, bordercolor=BORDER, font_size=13),
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER, tickfont=dict(color=SECONDARY), title_font=dict(color=SLATE, size=13))
    fig.update_yaxes(gridcolor="#E8EEE9", linecolor=BORDER, tickfont=dict(color=SECONDARY), title_font=dict(color=SLATE, size=13), zerolinecolor="#C9D5CB")
    return fig


def lai_histogram(frame: pd.DataFrame, lai_col: str) -> go.Figure:
    fig = px.histogram(frame, x=lai_col, nbins=20, color_discrete_sequence=[GREEN], title="Measured LAI distribution")
    fig.update_xaxes(title="Measured LAI")
    fig.update_yaxes(title="Observation count")
    return base_layout(fig, 330)


def time_series(frame: pd.DataFrame, date_col: str, lai_col: str) -> go.Figure:
    fig = px.scatter(frame, x=date_col, y=lai_col, color="Season" if "Season" in frame else None,
                     color_discrete_sequence=[GREEN, GOLD], title="Measured LAI by Sentinel-2 acquisition date")
    fig.update_xaxes(title="Satellite acquisition date")
    fig.update_yaxes(title="Measured LAI")
    return base_layout(fig)


def feature_scatter(frame: pd.DataFrame, feature: str, lai_col: str, hover: list[str]) -> go.Figure:
    fig = px.scatter(frame, x=feature, y=lai_col, color="Season" if "Season" in frame else None,
                     hover_data=hover, color_discrete_sequence=[GREEN, GOLD], title=f"{feature} versus measured LAI")
    fig.update_yaxes(title="Measured LAI")
    return base_layout(fig)


def metric_bar(frame: pd.DataFrame, metric: str) -> go.Figure:
    plot = frame.copy()
    plot["Configuration"] = plot["Model label"] + " · " + plot["Experiment label"]
    plot = plot.sort_values(metric, ascending=True)
    fig = px.bar(
        plot,
        x=metric,
        y="Configuration",
        orientation="h",
        color="Model label",
        text_auto=".3f",
        color_discrete_sequence=[GREEN, GOLD],
        title=f"{metric} by evaluated configuration",
    )
    fig.update_xaxes(title=metric, automargin=True)
    fig.update_yaxes(title=None, automargin=True)
    styled = base_layout(fig, 390)
    styled.update_layout(margin=dict(l=205, r=34, t=68, b=56))
    return styled


def actual_predicted(frame: pd.DataFrame) -> go.Figure:
    fig = px.scatter(frame, x="Actual_LAI", y="Predicted_LAI", color="Season" if "Season" in frame else None,
                     hover_data=[c for c in ["ESU_ID", "Unit_ID", "S2_Date", "Residual"] if c in frame],
                     color_discrete_sequence=[GREEN, GOLD], title="Out-of-fold measured versus predicted LAI")
    values = pd.concat([frame["Actual_LAI"], frame["Predicted_LAI"]]).dropna()
    if not values.empty:
        lower, upper = values.min(), values.max()
        padding = max((upper - lower) * 0.05, 0.1)
        fig.add_shape(type="line", x0=lower-padding, y0=lower-padding, x1=upper+padding, y1=upper+padding,
                      line=dict(color="#737d77", dash="dash"))
    fig.update_xaxes(title="Measured LAI")
    fig.update_yaxes(title="Predicted LAI")
    return base_layout(fig, 440)


def residual_histogram(frame: pd.DataFrame, residual_col: str) -> go.Figure:
    fig = px.histogram(frame, x=residual_col, nbins=22, color_discrete_sequence=[SAGE], title="Residual distribution")
    fig.add_vline(x=0, line_dash="dash", line_color="#737d77")
    fig.update_xaxes(title="Residual (measured LAI − predicted LAI)")
    fig.update_yaxes(title="Prediction count")
    return base_layout(fig, 330)


def unit_error_chart(frame: pd.DataFrame) -> go.Figure:
    fig = px.bar(frame.sort_values("MAE"), x="Group", y="MAE", hover_data=[c for c in ["RMSE", "Observations"] if c in frame],
                 color_discrete_sequence=[GREEN], title="Mean absolute error by season-specific field unit")
    fig.update_xaxes(title="Season · field unit", tickangle=-25)
    fig.update_yaxes(title="MAE (LAI units)")
    return base_layout(fig, 420)


def generic_scatter(frame: pd.DataFrame, x: str, y: str, color: str | None = None, title: str | None = None) -> go.Figure:
    fig = px.scatter(frame, x=x, y=y, color=color if color in frame else None, color_discrete_sequence=[GREEN, GOLD, SAGE], title=title or f"{x} versus {y}")
    return base_layout(fig, 390)


def correlation_heatmap(frame: pd.DataFrame, columns: list[str]) -> go.Figure:
    correlation = frame[columns].corr(numeric_only=True)
    fig = px.imshow(correlation, color_continuous_scale=[[0, "#F4F7F3"], [0.5, "#739A7A"], [1, "#164634"]], zmin=-1, zmax=1, text_auto=".2f", title="Feature correlation matrix")
    return base_layout(fig, max(360, min(720, 80 + len(columns) * 28)))


def generic_metric_bar(frame: pd.DataFrame, metric: str, label: str = "Model") -> go.Figure:
    plot = frame.sort_values(metric).copy()
    fig = px.bar(plot, x=metric, y=label, orientation="h", color_discrete_sequence=[GREEN], text_auto=".3f", title=f"{metric} by model")
    return base_layout(fig, max(320, 100 + len(plot) * 48))


def feature_configuration_metric(frame: pd.DataFrame, metric: str) -> go.Figure:
    fig = px.bar(
        frame,
        x="Feature configuration",
        y=metric,
        color="Model display",
        barmode="group",
        text_auto=".3f",
        category_orders={"Feature configuration": ["NDVI Only", "Spectral Bands Only", "Spectral Bands + Vegetation Indices"]},
        color_discrete_sequence=[GREEN, GOLD],
        title=f"{metric} by wheat feature configuration and model",
    )
    fig.update_xaxes(title="Satellite feature configuration")
    fig.update_yaxes(title="R²" if metric == "R2" else f"{metric} (LAI units)")
    return base_layout(fig, 410)


def normalized_actual_predicted(frame: pd.DataFrame) -> go.Figure:
    fig = px.scatter(frame, x="actual_lai", y="predicted_lai", color="campaign_or_season" if "campaign_or_season" in frame else None,
                     hover_data=[column for column in ["group_id", "model", "residual", "absolute_error"] if column in frame],
                     color_discrete_sequence=[GREEN, GOLD, SAGE], title="Historical out-of-fold: measured versus predicted LAI")
    values = pd.concat([frame["actual_lai"], frame["predicted_lai"]]).dropna()
    if not values.empty:
        low, high = values.min(), values.max()
        pad = max((high - low) * .05, .1)
        fig.add_shape(type="line", x0=low-pad, y0=low-pad, x1=high+pad, y1=high+pad, line=dict(color="#737d77", dash="dash"))
    fig.update_xaxes(title="Measured LAI")
    fig.update_yaxes(title="Predicted LAI")
    return base_layout(fig, 440)


def error_by_group(frame: pd.DataFrame) -> go.Figure:
    grouped = frame.groupby("group_id", dropna=False).agg(Observations=("actual_lai", "size"), MAE=("absolute_error", "mean")).reset_index().sort_values("MAE")
    fig = px.bar(grouped, x="group_id", y="MAE", hover_data=["Observations"], color_discrete_sequence=[GREEN], title="Mean absolute error by group")
    fig.update_xaxes(title="Farm / plot / field unit", tickangle=-30)
    fig.update_yaxes(title="MAE (LAI units)")
    return base_layout(fig, 390)
