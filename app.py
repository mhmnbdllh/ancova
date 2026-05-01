"""
ANCOVA Analysis App - SPSS-Equivalent Output
=============================================
Comprehensive Analysis of Covariance following SPSS methodology.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
import statsmodels.stats.multicomp as mc
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from io import BytesIO
import warnings
import itertools
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle, Paragraph,
                                  Spacer, HRFlowable, Image, PageBreak)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
import io
import base64
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ANCOVA Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    border-left: 5px solid #00d4ff;
}

.main-header h1 {
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}

.main-header p {
    color: #a8d8ea;
    margin: 0;
    font-size: 1rem;
}

.spss-table {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    border-collapse: collapse;
    width: 100%;
    margin: 1rem 0;
}

.spss-table th {
    background-color: #1a3a5c;
    color: #ffffff;
    padding: 8px 12px;
    text-align: center;
    font-weight: 600;
    border: 1px solid #2d5a8a;
    font-size: 0.80rem;
}

.spss-table td {
    padding: 6px 12px;
    border: 1px solid #d0d7de;
    text-align: right;
    color: #1a1a2e;
}

.spss-table tr:nth-child(even) { background-color: #f0f4f8; }
.spss-table tr:nth-child(odd)  { background-color: #ffffff; }

.spss-table td:first-child {
    text-align: left;
    font-weight: 500;
    background-color: #f8fafc;
}

.section-title {
    background: #1a3a5c;
    color: white;
    padding: 10px 16px;
    border-radius: 6px 6px 0 0;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.5px;
    margin-top: 1.5rem;
    margin-bottom: 0;
    font-family: 'IBM Plex Mono', monospace;
}

.interpretation-box {
    background: linear-gradient(135deg, #e8f4f8 0%, #f0f8ff 100%);
    border-left: 4px solid #0077b6;
    padding: 1rem 1.2rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
    font-size: 0.88rem;
    line-height: 1.7;
    color: #1a1a2e;
}

.interpretation-box b { color: #0077b6; }

.sig-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
}

.sig-yes { background: #d4edda; color: #155724; }
.sig-no  { background: #f8d7da; color: #721c24; }

.metric-card {
    background: white;
    border: 1px solid #e1e8ed;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.metric-card .value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1a3a5c;
    font-family: 'IBM Plex Mono', monospace;
}

.metric-card .label {
    font-size: 0.78rem;
    color: #6c757d;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.warning-box {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 0.8rem 1rem;
    border-radius: 0 6px 6px 0;
    font-size: 0.85rem;
    color: #856404;
    margin: 0.5rem 0;
}

.assumption-pass { color: #28a745; font-weight: 700; }
.assumption-fail { color: #dc3545; font-weight: 700; }
.assumption-warn { color: #ffc107; font-weight: 700; }

stButton button {
    font-family: 'IBM Plex Sans', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def fmt(v, decimals=3):
    """Format a number to fixed decimals."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}"

def fmt_p(p):
    """Format p-value SPSS-style."""
    if p is None or np.isnan(p):
        return "—"
    if p < .001:
        return ".000"
    return f"{p:.3f}"

def stars(p):
    if p < .001: return "***"
    if p < .01:  return "**"
    if p < .05:  return "*"
    return ""

def eta_sq_label(eta2):
    if eta2 < .01: return "negligible"
    if eta2 < .06: return "small"
    if eta2 < .14: return "medium"
    return "large"

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    pooled = np.sqrt(((n1-1)*np.var(g1,ddof=1)+(n2-1)*np.var(g2,ddof=1))/(n1+n2-2))
    return (np.mean(g1)-np.mean(g2))/pooled if pooled else np.nan

def observed_power(f_val, df1, df2, alpha=.05):
    from scipy.stats import ncf, f as fdist
    if np.isnan(f_val) or f_val <= 0:
        return np.nan
    crit = fdist.ppf(1-alpha, df1, df2)
    ncp  = f_val * df1
    power = 1 - ncf.cdf(crit, df1, df2, ncp)
    return np.clip(power, 0, 1)


# ════════════════════════════════════════════════════════════════════════════════
# ANALYSIS ENGINE
# ════════════════════════════════════════════════════════════════════════════════

def run_ancova(df, dep_var, group_var, covariates, alpha=0.05):
    """Run full SPSS-equivalent ANCOVA and return results dict."""
    results = {}
    groups  = df[group_var].unique()
    k       = len(groups)
    n_total = len(df)

    # ── 1. Descriptive Statistics ────────────────────────────────────────────
    desc_rows = []
    for g in sorted(groups):
        sub = df[df[group_var]==g]
        for var in [dep_var]+covariates:
            vals = sub[var].dropna()
            desc_rows.append({
                "Group": g, "Variable": var,
                "N": len(vals),
                "Mean": vals.mean(),
                "Std. Deviation": vals.std(ddof=1),
                "Std. Error": vals.sem(),
                "Min": vals.min(), "Max": vals.max(),
                "95% CI Lower": vals.mean() - 1.96*vals.sem(),
                "95% CI Upper": vals.mean() + 1.96*vals.sem(),
                "Skewness": vals.skew(),
                "Kurtosis": vals.kurt(),
            })
    results["descriptives"] = pd.DataFrame(desc_rows)

    # ── 2. Build ANCOVA model (SPSS Type III SS) ─────────────────────────────
    cov_terms = "+".join([f"Q('{c}')" for c in covariates])
    formula   = f"Q('{dep_var}') ~ C(Q('{group_var}')) + {cov_terms}"
    try:
        model = smf.ols(formula, data=df).fit()
    except Exception as e:
        results["error"] = str(e)
        return results
    results["model"] = model

    # Type III SS via ANOVA table
    from statsmodels.stats.anova import anova_lm
    anova_t3 = anova_lm(model, typ=3)
    results["anova_t3"] = anova_t3

    # Build SPSS-style Between-Subjects Effects table
    ss_total   = np.sum((df[dep_var]-df[dep_var].mean())**2)
    ss_model   = model.ess
    ss_error   = model.ssr
    ss_corrected_total = ss_model + ss_error

    # Intercept
    intercept_row = anova_t3.loc["Intercept"] if "Intercept" in anova_t3.index else None
    group_key = [i for i in anova_t3.index if "group_var" in i.lower() or group_var in i][0] if any("group_var" in i.lower() or group_var in i for i in anova_t3.index) else None

    bse_rows = []
    # Corrected Model
    df_model  = model.df_model
    ms_model  = ss_model/df_model if df_model else np.nan
    f_model   = ms_model/(ss_error/model.df_resid) if model.df_resid else np.nan
    p_model   = 1-stats.f.cdf(f_model, df_model, model.df_resid) if not np.isnan(f_model) else np.nan
    bse_rows.append({"Source":"Corrected Model","SS":ss_model,"df":df_model,"MS":ms_model,"F":f_model,"Sig.":p_model,"Partial η²":ss_model/ss_corrected_total})

    # Intercept
    if intercept_row is not None:
        ss_i = intercept_row["sum_sq"]
        df_i = int(intercept_row["df"])
        ms_i = ss_i/df_i if df_i else np.nan
        f_i  = intercept_row["F"]
        p_i  = intercept_row["PR(>F)"]
        bse_rows.append({"Source":"Intercept","SS":ss_i,"df":df_i,"MS":ms_i,"F":f_i,"Sig.":p_i,"Partial η²":ss_i/(ss_i+ss_error)})

    # Covariates
    for cov in covariates:
        cov_key = None
        for idx in anova_t3.index:
            if cov.lower() in idx.lower() and "intercept" not in idx.lower():
                cov_key = idx; break
        if cov_key:
            row = anova_t3.loc[cov_key]
            ss_c = row["sum_sq"]; df_c = int(row["df"])
            ms_c = ss_c/df_c if df_c else np.nan
            f_c  = row["F"];  p_c  = row["PR(>F)"]
            bse_rows.append({"Source":cov,"SS":ss_c,"df":df_c,"MS":ms_c,"F":f_c,"Sig.":p_c,"Partial η²":ss_c/(ss_c+ss_error)})

    # Factor
    gk = None
    for idx in anova_t3.index:
        if group_var.lower() in idx.lower() and "intercept" not in idx.lower():
            gk = idx; break
    if gk:
        row = anova_t3.loc[gk]
        ss_g = row["sum_sq"]; df_g = int(row["df"])
        ms_g = ss_g/df_g if df_g else np.nan
        f_g  = row["F"];  p_g  = row["PR(>F)"]
        eta2_g = ss_g/(ss_g+ss_error)
        pwr    = observed_power(f_g, df_g, int(model.df_resid))
        bse_rows.append({"Source":group_var,"SS":ss_g,"df":df_g,"MS":ms_g,"F":f_g,"Sig.":p_g,"Partial η²":eta2_g,"Observed Power":pwr})
        results["factor_f"]   = f_g
        results["factor_p"]   = p_g
        results["factor_df1"] = df_g
        results["factor_df2"] = int(model.df_resid)
        results["factor_eta2"]= eta2_g
    else:
        results["factor_f"] = results["factor_p"] = np.nan

    # Error
    bse_rows.append({"Source":"Error","SS":ss_error,"df":int(model.df_resid),"MS":ss_error/model.df_resid,"F":np.nan,"Sig.":np.nan,"Partial η²":np.nan})
    # Total
    bse_rows.append({"Source":"Corrected Total","SS":ss_corrected_total,"df":n_total-1,"MS":np.nan,"F":np.nan,"Sig.":np.nan,"Partial η²":np.nan})

    results["between_subjects"] = pd.DataFrame(bse_rows)
    results["R_squared"] = model.rsquared
    results["adj_R_squared"] = model.rsquared_adj

    # ── 3. Estimated Marginal Means ──────────────────────────────────────────
    emm_rows = []
    cov_means = {c: df[c].mean() for c in covariates}
    ms_error  = ss_error/model.df_resid

    for g in sorted(groups):
        pred_data = {group_var: [g]}
        for c in covariates:
            pred_data[c] = [cov_means[c]]
        pred_df = pd.DataFrame(pred_data)
        try:
            pred_val = model.predict(pred_df).values[0]
        except:
            pred_val = np.nan
        n_g  = (df[group_var]==g).sum()
        se_g = np.sqrt(ms_error/n_g)
        ci_l = pred_val - stats.t.ppf(1-alpha/2, model.df_resid)*se_g
        ci_u = pred_val + stats.t.ppf(1-alpha/2, model.df_resid)*se_g
        emm_rows.append({"Group":g,"Mean":pred_val,"Std. Error":se_g,f"{int((1-alpha)*100)}% CI Lower":ci_l,f"{int((1-alpha)*100)}% CI Upper":ci_u})

    results["emm"] = pd.DataFrame(emm_rows)

    # ── 4. Pairwise Comparisons (Bonferroni) ─────────────────────────────────
    pairs = list(itertools.combinations(sorted(groups), 2))
    pair_rows = []
    ms_err = ss_error/model.df_resid
    for g1, g2 in pairs:
        emm1 = results["emm"].loc[results["emm"]["Group"]==g1,"Mean"].values[0]
        emm2 = results["emm"].loc[results["emm"]["Group"]==g2,"Mean"].values[0]
        diff = emm1 - emm2
        n1   = (df[group_var]==g1).sum()
        n2   = (df[group_var]==g2).sum()
        se_diff = np.sqrt(ms_err*(1/n1+1/n2))
        t_val = diff/se_diff if se_diff else np.nan
        p_raw = 2*(1-stats.t.cdf(abs(t_val), model.df_resid))
        p_bonf= min(p_raw*len(pairs), 1.0)
        ci_l = diff - stats.t.ppf(1-alpha/2, model.df_resid)*se_diff
        ci_u = diff + stats.t.ppf(1-alpha/2, model.df_resid)*se_diff
        pair_rows.append({"(I) Group":g1,"(J) Group":g2,"Mean Diff (I-J)":diff,"Std. Error":se_diff,
                          "Sig. (Bonferroni)":p_bonf,"95% CI Lower":ci_l,"95% CI Upper":ci_u})
    results["pairwise"] = pd.DataFrame(pair_rows)

    # ── 5. Assumption Tests ──────────────────────────────────────────────────
    resid = model.resid

    # Normality (Shapiro-Wilk on residuals)
    sw_stat, sw_p = stats.shapiro(resid)
    results["shapiro"] = {"stat":sw_stat,"p":sw_p,"pass":sw_p>alpha}

    # Levene's test (homogeneity of variance)
    group_vals = [df[df[group_var]==g][dep_var].values for g in sorted(groups)]
    lev_stat, lev_p = stats.levene(*group_vals)
    results["levene"] = {"stat":lev_stat,"p":lev_p,"pass":lev_p>alpha}

    # Homogeneity of regression slopes (interaction term)
    int_terms = "+".join([f"C(Q('{group_var}')):Q('{c}')" for c in covariates])
    formula_int = formula + f" + {int_terms}"
    try:
        model_int = smf.ols(formula_int, data=df).fit()
        at3_int   = anova_lm(model_int, typ=3)
        int_results = []
        for c in covariates:
            for idx in at3_int.index:
                if ":" in idx and c in idx and group_var in idx:
                    row = at3_int.loc[idx]
                    int_results.append({"term":idx,"F":row["F"],"df":int(row["df"]),"p":row["PR(>F)"],"pass":row["PR(>F)"]>alpha})
        results["homog_slopes"] = int_results
    except:
        results["homog_slopes"] = []

    # Multicollinearity (if >1 covariate)
    if len(covariates) > 1:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        X = df[covariates].dropna()
        X_with_const = sm.add_constant(X)
        vif_data = []
        for i, col in enumerate(X_with_const.columns[1:], 1):
            vif_data.append({"Variable":col,"VIF":variance_inflation_factor(X_with_const.values, i)})
        results["vif"] = pd.DataFrame(vif_data)

    # ── 6. Model Summary ─────────────────────────────────────────────────────
    results["model_summary"] = {
        "R": np.sqrt(model.rsquared),
        "R²": model.rsquared,
        "Adj R²": model.rsquared_adj,
        "Std Error Estimate": np.sqrt(ms_error),
        "F": model.fvalue,
        "df1": int(model.df_model),
        "df2": int(model.df_resid),
        "p": model.f_pvalue,
        "N": n_total
    }

    return results


# ════════════════════════════════════════════════════════════════════════════════
# INTERPRETATION ENGINE
# ════════════════════════════════════════════════════════════════════════════════

def interpret_ancova(results, dep_var, group_var, covariates, alpha):
    lines = []

    # Overall model
    ms = results["model_summary"]
    sig_model = ms["p"] < alpha
    lines.append(f"<b>Overall Model:</b> The ANCOVA model was {'statistically significant' if sig_model else 'not statistically significant'}, "
                 f"F({ms['df1']}, {ms['df2']}) = {ms['F']:.3f}, p {'< .001' if ms['p']<.001 else '= '+fmt_p(ms['p'])}. "
                 f"The model explained {ms['R²']*100:.1f}% of total variance in <i>{dep_var}</i> (R² = {ms['R²']:.3f}, "
                 f"Adjusted R² = {ms['Adj R²']:.3f}), indicating a {'strong' if ms['R²']>.5 else 'moderate' if ms['R²']>.3 else 'weak'} fit.")

    # Group effect
    f_g = results.get("factor_f", np.nan)
    p_g = results.get("factor_p", np.nan)
    eta2 = results.get("factor_eta2", np.nan)
    if not np.isnan(f_g):
        sig_g = p_g < alpha
        lines.append(f"<b>Group Effect ({group_var}):</b> After controlling for the covariate(s), there was "
                     f"{'a statistically significant' if sig_g else 'no statistically significant'} effect of {group_var} on {dep_var}, "
                     f"F({results['factor_df1']}, {results['factor_df2']}) = {f_g:.3f}, p {'< .001' if p_g<.001 else '= '+fmt_p(p_g)}, "
                     f"partial η² = {eta2:.3f} ({eta_sq_label(eta2)} effect size). "
                     f"{'The groups differ significantly on the dependent variable when covariate(s) are held constant.' if sig_g else 'The groups do not differ significantly.'}")

    # Covariates
    bse = results.get("between_subjects", pd.DataFrame())
    for cov in covariates:
        cov_row = bse[bse["Source"]==cov]
        if not cov_row.empty:
            f_c = cov_row["F"].values[0]; p_c = cov_row["Sig."].values[0]
            eta_c = cov_row["Partial η²"].values[0]
            sig_c = p_c < alpha if not np.isnan(p_c) else False
            lines.append(f"<b>Covariate ({cov}):</b> The covariate {cov} was "
                         f"{'a statistically significant' if sig_c else 'not a statistically significant'} predictor "
                         f"of {dep_var}, F = {f_c:.3f}, p {'< .001' if p_c<.001 else '= '+fmt_p(p_c)}, "
                         f"partial η² = {eta_c:.3f} ({eta_sq_label(eta_c)} effect). "
                         f"{'This confirms the covariate accounted for a meaningful portion of error variance, increasing statistical power.' if sig_c else 'The covariate did not significantly reduce error variance.'}")

    # Pairwise
    pw = results.get("pairwise", pd.DataFrame())
    if not pw.empty:
        sig_pairs = pw[pw["Sig. (Bonferroni)"] < alpha]
        if not sig_pairs.empty:
            pair_strs = []
            for _, r in sig_pairs.iterrows():
                pair_strs.append(f"{r['(I) Group']} vs. {r['(J) Group']} (Δ = {r['Mean Diff (I-J)']:.3f}, p = {fmt_p(r['Sig. (Bonferroni)'])})")
            lines.append(f"<b>Pairwise Comparisons (Bonferroni):</b> Bonferroni-corrected pairwise comparisons revealed significant differences between: {'; '.join(pair_strs)}.")
        else:
            lines.append("<b>Pairwise Comparisons (Bonferroni):</b> No significant pairwise differences were found after Bonferroni correction.")

    # Assumptions
    sw  = results.get("shapiro",{})
    lev = results.get("levene",{})
    norm_ok = sw.get("pass", True)
    hom_ok  = lev.get("pass", True)
    lines.append(f"<b>Assumption Checks:</b> "
                 f"Normality of residuals (Shapiro-Wilk: W = {sw.get('stat',np.nan):.3f}, p = {fmt_p(sw.get('p',np.nan))}) was "
                 f"{'satisfied' if norm_ok else 'violated (interpret results with caution)'}. "
                 f"Homogeneity of variance (Levene: F = {lev.get('stat',np.nan):.3f}, p = {fmt_p(lev.get('p',np.nan))}) was "
                 f"{'met' if hom_ok else 'violated'}. "
                 + ("Homogeneity of regression slopes should be verified; interaction terms were tested separately." if results.get("homog_slopes") else ""))

    return lines


# ════════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════════════════════════════════════════════

PALETTE = ["#1a3a5c","#0077b6","#00b4d8","#90e0ef","#caf0f8"]

def plot_emm(results, dep_var, group_var, alpha):
    emm = results["emm"]
    ci_lower = f"{int((1-alpha)*100)}% CI Lower"
    ci_upper = f"{int((1-alpha)*100)}% CI Upper"
    fig, ax = plt.subplots(figsize=(8,5), facecolor='#f8fafc')
    ax.set_facecolor('#f8fafc')
    groups  = emm["Group"].values
    means   = emm["Mean"].values
    lowers  = emm[ci_lower].values
    uppers  = emm[ci_upper].values
    yerr    = np.array([means-lowers, uppers-means])
    x       = np.arange(len(groups))
    bars = ax.bar(x, means, color=[PALETTE[i%len(PALETTE)] for i in range(len(groups))],
                  alpha=0.85, edgecolor='white', linewidth=1.5, zorder=3)
    ax.errorbar(x, means, yerr=yerr, fmt='none', color='#333', capsize=6, linewidth=1.5, zorder=4)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"{m:.2f}",
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1a1a2e')
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel(f"Estimated Marginal Mean of {dep_var}", fontsize=9)
    ax.set_title("Estimated Marginal Means", fontsize=12, fontweight='bold', color='#1a3a5c', pad=12)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_xlabel(group_var, fontsize=9)
    plt.tight_layout()
    return fig

def plot_residuals(results):
    model  = results["model"]
    resid  = model.resid
    fitted = model.fittedvalues
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor='#f8fafc')
    for ax in axes: ax.set_facecolor('#f8fafc')

    # Residuals vs Fitted
    axes[0].scatter(fitted, resid, color=PALETTE[1], alpha=0.7, s=40, edgecolors='white', linewidths=0.5)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1.2)
    axes[0].set_xlabel("Fitted Values", fontsize=8)
    axes[0].set_ylabel("Residuals", fontsize=8)
    axes[0].set_title("Residuals vs. Fitted", fontsize=10, fontweight='bold', color='#1a3a5c')
    axes[0].spines[['top','right']].set_visible(False)

    # Q-Q Plot
    (osm, osr), (slope, intercept, r) = stats.probplot(resid)
    axes[1].plot(osm, osr, 'o', color=PALETTE[1], alpha=0.7, markersize=5, markeredgecolor='white')
    axes[1].plot(osm, slope*np.array(osm)+intercept, 'r--', linewidth=1.5)
    axes[1].set_xlabel("Theoretical Quantiles", fontsize=8)
    axes[1].set_ylabel("Sample Quantiles", fontsize=8)
    axes[1].set_title("Normal Q-Q Plot", fontsize=10, fontweight='bold', color='#1a3a5c')
    axes[1].spines[['top','right']].set_visible(False)

    # Histogram of residuals
    axes[2].hist(resid, bins=12, color=PALETTE[1], alpha=0.8, edgecolor='white', linewidth=0.8)
    xmin, xmax = resid.min(), resid.max()
    x_line = np.linspace(xmin, xmax, 100)
    axes[2].plot(x_line, stats.norm.pdf(x_line, resid.mean(), resid.std())*len(resid)*(xmax-xmin)/12,
                 'r--', linewidth=1.5, label='Normal curve')
    axes[2].set_xlabel("Residuals", fontsize=8)
    axes[2].set_ylabel("Frequency", fontsize=8)
    axes[2].set_title("Residual Distribution", fontsize=10, fontweight='bold', color='#1a3a5c')
    axes[2].spines[['top','right']].set_visible(False)
    plt.tight_layout()
    return fig

def plot_scatter_cov(df, dep_var, group_var, covariate):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#f8fafc')
    ax.set_facecolor('#f8fafc')
    groups = sorted(df[group_var].unique())
    for i, g in enumerate(groups):
        sub = df[df[group_var]==g]
        ax.scatter(sub[covariate], sub[dep_var], color=PALETTE[i%len(PALETTE)],
                   alpha=0.75, s=50, label=g, edgecolors='white', linewidths=0.5)
        m, b = np.polyfit(sub[covariate], sub[dep_var], 1)
        xl = np.linspace(sub[covariate].min(), sub[covariate].max(), 100)
        ax.plot(xl, m*xl+b, color=PALETTE[i%len(PALETTE)], linewidth=1.8)
    ax.set_xlabel(covariate, fontsize=9)
    ax.set_ylabel(dep_var, fontsize=9)
    ax.set_title(f"{dep_var} vs. {covariate} by {group_var}", fontsize=10, fontweight='bold', color='#1a3a5c')
    ax.legend(fontsize=8, frameon=True, framealpha=0.8)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=0.2, linestyle='--')
    plt.tight_layout()
    return fig

def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf.read()


# ════════════════════════════════════════════════════════════════════════════════
# PDF REPORT GENERATOR
# ════════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(results, df, dep_var, group_var, covariates, alpha, interpretations):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=1.8*cm, leftMargin=1.8*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    # Header style
    title_style = ParagraphStyle('Title', fontSize=18, fontName='Helvetica-Bold',
                                  textColor=colors.HexColor('#1a3a5c'), spaceAfter=6, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle('Subtitle', fontSize=10, fontName='Helvetica',
                                     textColor=colors.HexColor('#6c757d'), spaceAfter=20, alignment=TA_CENTER)
    h1_style = ParagraphStyle('H1', fontSize=13, fontName='Helvetica-Bold',
                               textColor=colors.white, spaceAfter=6, spaceBefore=14,
                               backColor=colors.HexColor('#1a3a5c'), leftIndent=-5, rightIndent=-5,
                               borderPadding=(5,8,5,8))
    h2_style = ParagraphStyle('H2', fontSize=10, fontName='Helvetica-Bold',
                               textColor=colors.HexColor('#1a3a5c'), spaceAfter=4, spaceBefore=8)
    body_style = ParagraphStyle('Body', fontSize=8.5, fontName='Helvetica',
                                 textColor=colors.HexColor('#1a1a2e'), spaceAfter=4, leading=13)
    interp_style = ParagraphStyle('Interp', fontSize=8.5, fontName='Helvetica',
                                   textColor=colors.HexColor('#1a1a2e'), spaceAfter=4,
                                   leading=13, backColor=colors.HexColor('#e8f4f8'),
                                   borderPadding=(6,8,6,8), leftIndent=8)
    note_style = ParagraphStyle('Note', fontSize=7.5, fontName='Helvetica-Oblique',
                                 textColor=colors.HexColor('#6c757d'))

    def section(title):
        story.append(Spacer(1, 8))
        story.append(Paragraph(f"  {title}", h1_style))
        story.append(Spacer(1, 6))

    def df_to_pdf_table(data_df, col_widths=None):
        tbl_data = [list(data_df.columns)]
        for _, row in data_df.iterrows():
            tbl_data.append([str(v) for v in row.values])
        t = Table(tbl_data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a3a5c')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 7.5),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,1), (-1,-1), 7.5),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f0f4f8')]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d0d7de')),
            ('ALIGN', (1,0), (-1,-1), 'CENTER'),
            ('ALIGN', (0,0), (0,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ]))
        return t

    # ── Cover page ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("ANALYSIS OF COVARIANCE (ANCOVA)", title_style))
    story.append(Paragraph("SPSS-Equivalent Statistical Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#0077b6')))
    story.append(Spacer(1, 8))

    meta = [
        ["Dependent Variable:", dep_var],
        ["Fixed Factor:", group_var],
        ["Covariate(s):", ", ".join(covariates)],
        ["Sample Size:", str(len(df))],
        ["Significance Level:", f"α = {alpha}"],
        ["Analysis Date:", datetime.now().strftime("%B %d, %Y  %H:%M")],
    ]
    mt = Table(meta, colWidths=[4*cm, 12*cm])
    mt.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (1,0), (1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('TEXTCOLOR', (0,0), (0,-1), colors.HexColor('#1a3a5c')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(mt)
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#d0d7de')))

    # ── Model Summary ─────────────────────────────────────────────────────────
    section("1. MODEL SUMMARY")
    ms = results["model_summary"]
    ms_df = pd.DataFrame([{
        "R": fmt(ms['R'],3), "R²": fmt(ms['R²'],3), "Adj R²": fmt(ms['Adj R²'],3),
        "Std. Error": fmt(ms['Std Error Estimate'],3),
        "F": fmt(ms['F'],3), "df1": int(ms['df1']), "df2": int(ms['df2']),
        "Sig.": fmt_p(ms['p'])
    }])
    story.append(df_to_pdf_table(ms_df))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Note. R² values indicate the proportion of variance explained by the model.", note_style))

    # ── Between-Subjects Effects ──────────────────────────────────────────────
    section("2. TESTS OF BETWEEN-SUBJECTS EFFECTS")
    story.append(Paragraph(f"Dependent Variable: {dep_var}", body_style))
    bse = results["between_subjects"].copy()
    bse_fmt = []
    for _, row in bse.iterrows():
        bse_fmt.append({
            "Source": row["Source"],
            "Type III SS": fmt(row["SS"],3),
            "df": str(int(row["df"])) if not np.isnan(row["df"]) else "—",
            "MS": fmt(row.get("MS",np.nan),3),
            "F": fmt(row.get("F",np.nan),3),
            "Sig.": fmt_p(row.get("Sig.",np.nan)),
            "Partial η²": fmt(row.get("Partial η²",np.nan),3),
            "Obs. Power": fmt(row.get("Observed Power",np.nan),3),
        })
    story.append(df_to_pdf_table(pd.DataFrame(bse_fmt)))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Note. Computed using α = " + str(alpha) + ". Type III Sum of Squares.", note_style))

    # ── Descriptive Statistics ────────────────────────────────────────────────
    section("3. DESCRIPTIVE STATISTICS")
    desc = results["descriptives"].copy()
    desc_fmt = desc.copy()
    for col in ["Mean","Std. Deviation","Std. Error","Min","Max","95% CI Lower","95% CI Upper","Skewness","Kurtosis"]:
        desc_fmt[col] = desc[col].apply(lambda x: fmt(x,3))
    story.append(df_to_pdf_table(desc_fmt))

    # ── EMM ───────────────────────────────────────────────────────────────────
    section("4. ESTIMATED MARGINAL MEANS")
    ci_l = [c for c in results["emm"].columns if "Lower" in c][0]
    ci_u = [c for c in results["emm"].columns if "Upper" in c][0]
    emm_fmt = results["emm"].copy()
    for col in ["Mean","Std. Error",ci_l,ci_u]:
        emm_fmt[col] = emm_fmt[col].apply(lambda x: fmt(x,3))
    story.append(Paragraph(f"Dependent Variable: {dep_var}", body_style))
    story.append(df_to_pdf_table(emm_fmt))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Note. Covariates appearing in the model are evaluated at the following values: "
                            + ", ".join([f"{c} = {df[c].mean():.3f}" for c in covariates]), note_style))

    # ── Pairwise Comparisons ──────────────────────────────────────────────────
    section("5. PAIRWISE COMPARISONS (BONFERRONI)")
    story.append(Paragraph(f"Dependent Variable: {dep_var}", body_style))
    pw = results["pairwise"].copy()
    pw_fmt = pw.copy()
    for col in ["Mean Diff (I-J)","Std. Error","95% CI Lower","95% CI Upper"]:
        pw_fmt[col] = pw[col].apply(lambda x: fmt(x,3))
    pw_fmt["Sig. (Bonferroni)"] = pw["Sig. (Bonferroni)"].apply(fmt_p)
    story.append(df_to_pdf_table(pw_fmt))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Note. Based on estimated marginal means. Adjustment for multiple comparisons: Bonferroni.", note_style))

    # ── Assumption Tests ──────────────────────────────────────────────────────
    section("6. ASSUMPTION TESTS")
    story.append(Paragraph("a) Normality of Residuals — Shapiro-Wilk Test", h2_style))
    sw = results["shapiro"]
    sw_df = pd.DataFrame([{"Statistic (W)": fmt(sw['stat'],3), "Sig.": fmt_p(sw['p']),
                            "Result": "Assumption Met ✓" if sw['pass'] else "Assumption Violated ✗"}])
    story.append(df_to_pdf_table(sw_df))

    story.append(Paragraph("b) Homogeneity of Variances — Levene's Test", h2_style))
    lev = results["levene"]
    lev_df = pd.DataFrame([{"Levene Statistic": fmt(lev['stat'],3), "Sig.": fmt_p(lev['p']),
                              "Result": "Assumption Met ✓" if lev['pass'] else "Assumption Violated ✗"}])
    story.append(df_to_pdf_table(lev_df))

    if results.get("homog_slopes"):
        story.append(Paragraph("c) Homogeneity of Regression Slopes", h2_style))
        hs_rows = [{"Term":r['term'], "F":fmt(r['F'],3), "df":str(r['df']), "Sig.":fmt_p(r['p']),
                    "Result":"Met ✓" if r['pass'] else "Violated ✗"} for r in results["homog_slopes"]]
        story.append(df_to_pdf_table(pd.DataFrame(hs_rows)))

    if "vif" in results:
        story.append(Paragraph("d) Multicollinearity — Variance Inflation Factors (VIF)", h2_style))
        vif_fmt = results["vif"].copy()
        vif_fmt["VIF"] = vif_fmt["VIF"].apply(lambda x: fmt(x,3))
        story.append(df_to_pdf_table(vif_fmt))

    # ── Interpretation ────────────────────────────────────────────────────────
    section("7. STATISTICAL INTERPRETATION")
    for line in interpretations:
        clean = line.replace('<b>','').replace('</b>','').replace('<i>','').replace('</i>','')
        story.append(Paragraph(clean, interp_style))
        story.append(Spacer(1, 4))

    # ── Plots ─────────────────────────────────────────────────────────────────
    section("8. FIGURES")
    try:
        fig_emm   = plot_emm(results, dep_var, group_var, alpha)
        emm_bytes = fig_to_bytes(fig_emm)
        plt.close(fig_emm)
        story.append(Image(io.BytesIO(emm_bytes), width=14*cm, height=8.5*cm))
        story.append(Paragraph("Figure 1. Estimated Marginal Means with 95% confidence intervals.", note_style))
        story.append(Spacer(1, 8))

        fig_res   = plot_residuals(results)
        res_bytes = fig_to_bytes(fig_res)
        plt.close(fig_res)
        story.append(Image(io.BytesIO(res_bytes), width=17*cm, height=5*cm))
        story.append(Paragraph("Figure 2. Residual diagnostic plots.", note_style))
    except:
        pass

    # ── Footer note ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#d0d7de')))
    story.append(Paragraph("Generated by ANCOVA Analysis App · Results follow SPSS General Linear Model (GLM) conventions · "
                            "Type III Sum of Squares · Bonferroni correction applied to pairwise comparisons", note_style))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ════════════════════════════════════════════════════════════════════════════════
# UI RENDERING FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def render_table(df, title=None):
    if title:
        st.markdown(f'<div class="section-title">📋 {title}</div>', unsafe_allow_html=True)
    html = '<table class="spss-table"><thead><tr>'
    for col in df.columns:
        html += f'<th>{col}</th>'
    html += '</tr></thead><tbody>'
    for _, row in df.iterrows():
        html += '<tr>'
        for val in row.values:
            html += f'<td>{val}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    st.markdown(html, unsafe_allow_html=True)

def render_interpretation(lines):
    for line in lines:
        st.markdown(f'<div class="interpretation-box">{line}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 ANCOVA Analysis Suite</h1>
        <p>Analysis of Covariance · SPSS-Equivalent Output · Comprehensive Statistical Reporting</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")

        # Sample data download
        with open("/mnt/user-data/outputs/ancova_sample_data.csv", "rb") as f:
            st.download_button("⬇️ Download Sample Data", f, "ancova_sample_data.csv", "text/csv",
                               help="Download sample CSV to test the app")
        st.markdown("---")

        uploaded = st.file_uploader("📂 Upload CSV Data", type=["csv"])
        if uploaded:
            sep_opt = st.selectbox("Delimiter", [",",";","\t","|"], index=0)
            try:
                df = pd.read_csv(uploaded, sep=sep_opt)
                st.success(f"✅ {len(df)} rows × {len(df.columns)} cols loaded")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                df = None
        else:
            df = pd.read_csv("/mnt/user-data/outputs/ancova_sample_data.csv")
            st.info("ℹ️ Using built-in sample data")

        if df is not None:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            all_cols = df.columns.tolist()

            st.markdown("---")
            dep_var   = st.selectbox("🎯 Dependent Variable (DV)", num_cols,
                                      index=num_cols.index("posttest") if "posttest" in num_cols else 0)
            group_var = st.selectbox("👥 Fixed Factor (Group)",
                                      [c for c in all_cols if c != dep_var],
                                      index=([c for c in all_cols if c != dep_var].index("group")
                                             if "group" in all_cols else 0))
            cov_candidates = [c for c in num_cols if c != dep_var]
            covariates = st.multiselect("📐 Covariate(s)", cov_candidates,
                                         default=["pretest"] if "pretest" in cov_candidates else (cov_candidates[:1] if cov_candidates else []))
            alpha = st.selectbox("α Level", [0.05, 0.01, 0.001], index=0)

            st.markdown("---")
            run_btn = st.button("🚀 Run ANCOVA", type="primary", use_container_width=True)
        else:
            run_btn = False

    if df is None:
        return

    # Preview
    with st.expander("🔍 Data Preview", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)
        c1,c2,c3 = st.columns(3)
        c1.metric("Rows", len(df)); c2.metric("Columns", len(df.columns)); c3.metric("Missing", df.isnull().sum().sum())

    if not run_btn and 'ancova_results' not in st.session_state:
        st.info("👈 Configure your variables in the sidebar and click **Run ANCOVA**.")
        return

    if run_btn:
        if not covariates:
            st.error("Please select at least one covariate."); return
        with st.spinner("Running ANCOVA analysis..."):
            results = run_ancova(df, dep_var, group_var, covariates, alpha)
        if "error" in results:
            st.error(f"Analysis failed: {results['error']}"); return
        st.session_state["ancova_results"] = results
        st.session_state["ancova_params"]  = (dep_var, group_var, covariates, alpha)

    results = st.session_state.get("ancova_results")
    params  = st.session_state.get("ancova_params")
    if results is None or params is None:
        return
    dep_var, group_var, covariates, alpha = params

    interpretations = interpret_ancova(results, dep_var, group_var, covariates, alpha)

    st.success("✅ ANCOVA completed successfully!")

    # Quick metrics
    ms = results["model_summary"]
    cols = st.columns(5)
    metrics = [
        (f"F({ms['df1']},{ms['df2']})", f"{ms['F']:.3f}", "Model F-statistic"),
        (fmt_p(ms["p"]), "", "Model Sig."),
        (fmt(ms["R²"],3), "", "R-squared"),
        (fmt(ms["Adj R²"],3), "", "Adjusted R²"),
        (fmt(results.get("factor_eta2",np.nan),3), "", f"Partial η² ({group_var})"),
    ]
    for col, (val, delta, label) in zip(cols, metrics):
        col.markdown(f'<div class="metric-card"><div class="value">{val}</div><div class="label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # TABS
    tabs = st.tabs(["📋 Between-Subjects","📊 Descriptives","🎯 Marginal Means",
                     "🔀 Pairwise","⚖️ Assumptions","📈 Plots","💬 Interpretation"])

    # ── Tab 1: Between-Subjects Effects ──────────────────────────────────────
    with tabs[0]:
        st.markdown(f'<div class="section-title">Tests of Between-Subjects Effects · Dependent Variable: {dep_var}</div>', unsafe_allow_html=True)
        bse = results["between_subjects"].copy()
        bse_disp = []
        for _, row in bse.iterrows():
            sig_val = row.get("Sig.", np.nan)
            sig_str = fmt_p(sig_val) if not np.isnan(sig_val) else "—"
            if not np.isnan(sig_val):
                badge = "sig-yes" if sig_val < alpha else "sig-no"
                sig_str += f' <span class="sig-badge {badge}">{"✓" if sig_val < alpha else "✗"}</span>'
            bse_disp.append({
                "Source": row["Source"],
                "Type III SS": fmt(row["SS"],3),
                "df": str(int(row["df"])) if not np.isnan(row["df"]) else "—",
                "MS": fmt(row.get("MS",np.nan),3),
                "F": fmt(row.get("F",np.nan),3),
                "Sig.": sig_str,
                "Partial η²": fmt(row.get("Partial η²",np.nan),3),
                "Observed Power": fmt(row.get("Observed Power",np.nan),3),
            })
        html = '<table class="spss-table"><thead><tr>'
        for col in bse_disp[0].keys():
            html += f'<th>{col}</th>'
        html += '</tr></thead><tbody>'
        for row in bse_disp:
            html += '<tr>'
            for val in row.values():
                html += f'<td>{val}</td>'
            html += '</tr>'
        html += '</tbody></table>'
        st.markdown(html, unsafe_allow_html=True)
        st.markdown(f"<small>Note: Computed using α = {alpha}. Type III Sum of Squares. R² = {fmt(ms['R²'],3)} (Adjusted R² = {fmt(ms['Adj R²'],3)})</small>", unsafe_allow_html=True)

    # ── Tab 2: Descriptives ───────────────────────────────────────────────────
    with tabs[1]:
        desc = results["descriptives"].copy()
        desc_disp = desc.copy()
        for col in ["Mean","Std. Deviation","Std. Error","Min","Max","95% CI Lower","95% CI Upper","Skewness","Kurtosis"]:
            desc_disp[col] = desc[col].apply(lambda x: fmt(x,3))
        render_table(desc_disp, "Descriptive Statistics")
        st.markdown(f"<small>Note: Descriptive statistics computed for all variables by {group_var}.</small>", unsafe_allow_html=True)

    # ── Tab 3: EMM ────────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown(f'<div class="section-title">Estimated Marginal Means · Dependent Variable: {dep_var}</div>', unsafe_allow_html=True)
        ci_l = [c for c in results["emm"].columns if "Lower" in c][0]
        ci_u = [c for c in results["emm"].columns if "Upper" in c][0]
        emm_disp = results["emm"].copy()
        for col in ["Mean","Std. Error",ci_l,ci_u]:
            emm_disp[col] = emm_disp[col].apply(lambda x: fmt(x,3))
        render_table(emm_disp)
        st.markdown(f"<small>Note: Covariates evaluated at mean values: {', '.join([f'{c} = {df[c].mean():.3f}' for c in covariates])}</small>", unsafe_allow_html=True)

    # ── Tab 4: Pairwise ───────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown(f'<div class="section-title">Pairwise Comparisons (Bonferroni) · {dep_var}</div>', unsafe_allow_html=True)
        pw = results["pairwise"].copy()
        pw_disp = pw.copy()
        for col in ["Mean Diff (I-J)","Std. Error","95% CI Lower","95% CI Upper"]:
            pw_disp[col] = pw[col].apply(lambda x: fmt(x,3))
        pw_disp["Sig. (Bonferroni)"] = pw["Sig. (Bonferroni)"].apply(fmt_p)
        render_table(pw_disp)
        st.markdown("<small>Note: Based on estimated marginal means. Adjustment for multiple comparisons: Bonferroni.</small>", unsafe_allow_html=True)

    # ── Tab 5: Assumptions ────────────────────────────────────────────────────
    with tabs[4]:
        c1, c2 = st.columns(2)
        with c1:
            sw = results["shapiro"]
            status = '<span class="assumption-pass">✓ MET</span>' if sw['pass'] else '<span class="assumption-fail">✗ VIOLATED</span>'
            st.markdown(f'<div class="section-title">Shapiro-Wilk Normality Test (Residuals)</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <table class="spss-table"><thead><tr><th>Statistic (W)</th><th>df</th><th>Sig.</th><th>Result</th></tr></thead>
            <tbody><tr><td>{fmt(sw['stat'],3)}</td><td>{len(df)-1}</td><td>{fmt_p(sw['p'])}</td><td>{status}</td></tr></tbody>
            </table>""", unsafe_allow_html=True)
            if not sw['pass']:
                st.markdown('<div class="warning-box">⚠️ Normality assumption violated. Consider data transformations or non-parametric alternatives.</div>', unsafe_allow_html=True)

        with c2:
            lev = results["levene"]
            status = '<span class="assumption-pass">✓ MET</span>' if lev['pass'] else '<span class="assumption-fail">✗ VIOLATED</span>'
            st.markdown('<div class="section-title">Levene\'s Test of Homogeneity of Variance</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <table class="spss-table"><thead><tr><th>Levene Statistic</th><th>df1</th><th>df2</th><th>Sig.</th><th>Result</th></tr></thead>
            <tbody><tr><td>{fmt(lev['stat'],3)}</td><td>{len(df[group_var].unique())-1}</td><td>{len(df)-len(df[group_var].unique())}</td><td>{fmt_p(lev['p'])}</td><td>{status}</td></tr></tbody>
            </table>""", unsafe_allow_html=True)
            if not lev['pass']:
                st.markdown('<div class="warning-box">⚠️ Homogeneity of variance violated. Results may not be reliable.</div>', unsafe_allow_html=True)

        if results.get("homog_slopes"):
            st.markdown('<div class="section-title">Homogeneity of Regression Slopes</div>', unsafe_allow_html=True)
            hs_rows = []
            for r in results["homog_slopes"]:
                status = '<span class="assumption-pass">✓ MET</span>' if r['pass'] else '<span class="assumption-fail">✗ VIOLATED</span>'
                hs_rows.append({"Interaction Term":r['term'],"F":fmt(r['F'],3),"df":r['df'],"Sig.":fmt_p(r['p']),"Result":status})
            html = '<table class="spss-table"><thead><tr>' + ''.join(f'<th>{c}</th>' for c in hs_rows[0].keys()) + '</tr></thead><tbody>'
            for row in hs_rows:
                html += '<tr>' + ''.join(f'<td>{v}</td>' for v in row.values()) + '</tr>'
            html += '</tbody></table>'
            st.markdown(html, unsafe_allow_html=True)
            st.markdown("<small>Note: Non-significant interaction terms indicate homogeneity of regression slopes (ANCOVA assumption met).</small>", unsafe_allow_html=True)

        if "vif" in results:
            st.markdown('<div class="section-title">Multicollinearity — Variance Inflation Factors (VIF)</div>', unsafe_allow_html=True)
            vif_df = results["vif"].copy()
            vif_df["VIF"] = vif_df["VIF"].apply(lambda x: fmt(x,3))
            vif_df["Interpretation"] = results["vif"]["VIF"].apply(
                lambda v: '<span class="assumption-pass">✓ Acceptable (&lt;5)</span>' if v<5
                else '<span class="assumption-fail">✗ Concerning (&gt;5)</span>')
            html = '<table class="spss-table"><thead><tr>' + ''.join(f'<th>{c}</th>' for c in vif_df.columns) + '</tr></thead><tbody>'
            for _, row in vif_df.iterrows():
                html += '<tr>' + ''.join(f'<td>{v}</td>' for v in row.values) + '</tr>'
            html += '</tbody></table>'
            st.markdown(html, unsafe_allow_html=True)

    # ── Tab 6: Plots ──────────────────────────────────────────────────────────
    with tabs[5]:
        c1, c2 = st.columns([1.3,1])
        with c1:
            fig_emm = plot_emm(results, dep_var, group_var, alpha)
            st.pyplot(fig_emm, use_container_width=True)
            plt.close(fig_emm)
        with c2:
            for cov in covariates:
                fig_sc = plot_scatter_cov(df, dep_var, group_var, cov)
                st.pyplot(fig_sc, use_container_width=True)
                plt.close(fig_sc)

        fig_res = plot_residuals(results)
        st.pyplot(fig_res, use_container_width=True)
        plt.close(fig_res)

    # ── Tab 7: Interpretation ─────────────────────────────────────────────────
    with tabs[6]:
        st.markdown("### 📝 Statistical Interpretation")
        st.markdown("*Automatically generated interpretations following APA 7th edition reporting guidelines:*")
        render_interpretation(interpretations)

        # APA write-up
        f_g  = results.get("factor_f", np.nan)
        p_g  = results.get("factor_p", np.nan)
        eta2 = results.get("factor_eta2", np.nan)
        df1  = results.get("factor_df1", "")
        df2  = results.get("factor_df2", "")
        apa  = (f"A one-way ANCOVA was conducted to examine the effect of {group_var} on {dep_var} "
                f"after controlling for {', '.join(covariates)}. "
                f"The ANCOVA was {'significant' if not np.isnan(p_g) and p_g < alpha else 'not significant'}, "
                f"F({df1}, {df2}) = {fmt(f_g,2)}, p {'< .001' if not np.isnan(p_g) and p_g < .001 else '= '+fmt_p(p_g)}, "
                f"partial η² = {fmt(eta2,3)} ({eta_sq_label(eta2)} effect).")
        st.markdown("**APA 7th Edition Write-Up:**")
        st.code(apa, language=None)

    # ── Download Section ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Export Results")
    dcol1, dcol2, dcol3 = st.columns(3)

    # PDF
    with dcol1:
        with st.spinner("Generating PDF..."):
            pdf_bytes = generate_pdf_report(results, df, dep_var, group_var, covariates, alpha, interpretations)
        st.download_button("📄 Download PDF Report", pdf_bytes,
                           f"ANCOVA_Report_{dep_var}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                           "application/pdf", use_container_width=True)

    # Excel
    with dcol2:
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
            results["descriptives"].to_excel(writer, "Descriptive Statistics", index=False)
            results["between_subjects"].to_excel(writer, "Between-Subjects Effects", index=False)
            results["emm"].to_excel(writer, "Estimated Marginal Means", index=False)
            results["pairwise"].to_excel(writer, "Pairwise Comparisons", index=False)
            pd.DataFrame([results["shapiro"]]).to_excel(writer, "Shapiro-Wilk", index=False)
            pd.DataFrame([results["levene"]]).to_excel(writer, "Levene Test", index=False)
        excel_buf.seek(0)
        st.download_button("📊 Download Excel", excel_buf.getvalue(),
                           f"ANCOVA_Tables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

    # CSV
    with dcol3:
        csv_all = "\n\n".join([
            "=== BETWEEN-SUBJECTS EFFECTS ===",
            results["between_subjects"].to_csv(index=False),
            "=== DESCRIPTIVE STATISTICS ===",
            results["descriptives"].to_csv(index=False),
            "=== ESTIMATED MARGINAL MEANS ===",
            results["emm"].to_csv(index=False),
            "=== PAIRWISE COMPARISONS ===",
            results["pairwise"].to_csv(index=False),
        ])
        st.download_button("📝 Download CSV Tables", csv_all.encode(),
                           f"ANCOVA_Tables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           "text/csv", use_container_width=True)


if __name__ == "__main__":
    main()
