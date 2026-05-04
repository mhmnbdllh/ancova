"""
ANCOVA Analysis App — SPSS-Equivalent Output
=============================================
Analysis of Covariance following SPSS General Linear Model (GLM) conventions.
Type III Sum of Squares · LSD / Bonferroni / Sidak post-hoc · Full assumption testing.

KEY FORMULA CORRECTIONS (SPSS-exact):
  • EMM SE  = sqrt( c'(X'X)^-1 c * MS_error )   — per-group contrast vector
  • Pairwise SE = sqrt( (c_i-c_j)'(X'X)^-1(c_i-c_j) * MS_error )
  • Bonferroni p = min(p_LSD * k, 1)  where k = number of pairs
  • Sidak      p = 1 - (1 - p_LSD)^k
  • Intercept F uses MS_error as denominator (Type III)
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import patsy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
import io
import warnings
import itertools
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                 Paragraph, Spacer, HRFlowable, Image, PageBreak)
from reportlab.lib.enums import TA_CENTER

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ANCOVA Analysis", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,600;0,700;1,400&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;}
.main-hdr{background:linear-gradient(135deg,#0f2027 0%,#203a43 50%,#2c5364 100%);
  padding:1.8rem 2.2rem;border-radius:12px;margin-bottom:1.5rem;border-left:5px solid #00d4ff;}
.main-hdr h1{color:#fff;font-size:2rem;font-weight:700;margin:0 0 .3rem 0;}
.main-hdr p{color:#a8d8ea;margin:0;font-size:.95rem;}
.sec-title{background:#1a3a5c;color:#fff;padding:8px 14px;border-radius:6px 6px 0 0;
  font-weight:700;font-size:.85rem;letter-spacing:.5px;margin-top:1.2rem;font-family:'IBM Plex Mono',monospace;}
.spss-wrap{overflow-x:auto;margin-bottom:.5rem;}
.spss-tbl{font-family:'IBM Plex Mono',monospace;font-size:.78rem;border-collapse:collapse;width:100%;min-width:500px;}
.spss-tbl th{background:#1a3a5c;color:#fff;padding:7px 11px;text-align:center;
  font-weight:600;border:1px solid #2d5a8a;font-size:.76rem;white-space:nowrap;}
.spss-tbl td{padding:5px 11px;border:1px solid #d0d7de;text-align:right;color:#1a1a2e;white-space:nowrap;}
.spss-tbl tr:nth-child(even) td{background:#f0f4f8;}
.spss-tbl tr:nth-child(odd) td{background:#fff;}
.spss-tbl td:first-child{text-align:left;font-weight:500;background:#f8fafc!important;}
.interp-box{background:linear-gradient(135deg,#e8f4f8,#f0f8ff);border-left:4px solid #0077b6;
  padding:.9rem 1.1rem;border-radius:0 8px 8px 0;margin:.6rem 0;font-size:.87rem;
  line-height:1.75;color:#1a1a2e;}
.interp-box b{color:#0077b6;}
.metric-card{background:#fff;border:1px solid #e1e8ed;border-radius:10px;padding:.9rem;
  text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.06);}
.metric-val{font-size:1.55rem;font-weight:700;color:#1a3a5c;font-family:'IBM Plex Mono',monospace;}
.metric-lbl{font-size:.72rem;color:#6c757d;margin-top:.25rem;text-transform:uppercase;letter-spacing:.5px;}
.warn-box{background:#fff3cd;border-left:4px solid #ffc107;padding:.7rem 1rem;
  border-radius:0 6px 6px 0;font-size:.82rem;color:#856404;margin:.4rem 0;}
.pass{color:#28a745;font-weight:700;} .fail{color:#dc3545;font-weight:700;}
.note-txt{font-size:.76rem;color:#6c757d;font-style:italic;margin-top:.3rem;}
</style>
""", unsafe_allow_html=True)

# ── Embedded sample data ───────────────────────────────────────────────────────
SAMPLE_CSV = """subject_id,group,pretest,posttest
1,Control,45,50
2,Control,50,55
3,Control,38,42
4,Control,60,63
5,Control,55,58
6,Control,42,47
7,Control,48,52
8,Control,53,56
9,Control,40,44
10,Control,58,61
11,Treatment_A,46,60
12,Treatment_A,52,68
13,Treatment_A,39,54
14,Treatment_A,61,75
15,Treatment_A,56,70
16,Treatment_A,43,58
17,Treatment_A,49,64
18,Treatment_A,54,69
19,Treatment_A,41,56
20,Treatment_A,59,73
21,Treatment_B,47,72
22,Treatment_B,51,78
23,Treatment_B,40,66
24,Treatment_B,62,88
25,Treatment_B,57,83
26,Treatment_B,44,70
27,Treatment_B,50,76
28,Treatment_B,55,81
29,Treatment_B,42,68
30,Treatment_B,60,86"""

# ── Utility ────────────────────────────────────────────────────────────────────
def _f(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "."
    return f"{v:.{d}f}"

def _p(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "."
    return ".000" if v < .001 else f"{v:.3f}"

def eta_label(e):
    if np.isnan(e): return "—"
    if e < .01: return "negligible"
    if e < .06: return "small"
    if e < .14: return "medium"
    return "large"

def obs_power(f_val, df1, df2, alpha=.05):
    from scipy.stats import ncf, f as fdist
    if np.isnan(f_val) or f_val <= 0:
        return np.nan
    crit = fdist.ppf(1 - alpha, df1, df2)
    ncp  = f_val * df1
    return float(np.clip(1 - ncf.cdf(crit, df1, df2, ncp), 0, 1))

# ── Build contrast vectors from model design matrix ────────────────────────────
def build_contrasts(model, df, grp, covs, groups):
    """
    Return dict {group_name -> contrast_vector} and XtX_inv.
    Contrast vector c evaluates the model at group g, covariates at their means.
    SPSS formula: Var(EMM_g) = c_g' (X'X)^{-1} c_g * MS_error
    """
    cov_means = {c: float(df[c].mean()) for c in covs}
    params    = model.params

    # Identify parameter names
    param_names = list(params.index)

    # Build design matrix to get (X'X)^-1
    X = np.array(model.model.exog)  # already the design matrix used in fit
    XtX_inv = np.linalg.inv(X.T @ X)

    # Map group -> contrast vector
    # We do this by predicting with a single-row dataframe at cov means
    # then extracting the corresponding row from patsy design matrix
    contrasts = {}
    for g in groups:
        row_df = pd.DataFrame({grp: [g]})
        for c in covs:
            row_df[c] = [cov_means[c]]
        # Use patsy to get the design row (same formula as model)
        dm = patsy.dmatrix(model.model.data.design_info, row_df, return_type='matrix')
        c_vec = np.array(dm)[0]
        contrasts[g] = c_vec

    return contrasts, XtX_inv, cov_means

# ── ANCOVA engine ──────────────────────────────────────────────────────────────
def run_ancova(df, dep, grp, covs, alpha=0.05):
    R = {}
    groups  = sorted(df[grp].unique())
    n_total = len(df)

    # ── Descriptive statistics ─────────────────────────────────────────────────
    desc = []
    for g in groups:
        sub = df[df[grp] == g]
        for var in [dep] + covs:
            v = sub[var].dropna()
            desc.append({
                "Group": g, "Variable": var, "N": len(v),
                "Mean": v.mean(), "Std. Deviation": v.std(ddof=1),
                "Std. Error": v.sem(),
                "95% CI Lower": v.mean() - 1.96 * v.sem(),
                "95% CI Upper": v.mean() + 1.96 * v.sem(),
                "Minimum": v.min(), "Maximum": v.max(),
                "Skewness": float(v.skew()), "Kurtosis": float(v.kurt()),
            })
    R["desc"] = pd.DataFrame(desc)

    # ── OLS model (Type III SS, SPSS GLM) ─────────────────────────────────────
    cov_str = " + ".join([f'Q("{c}")' for c in covs])
    formula = f'Q("{dep}") ~ C(Q("{grp}")) + {cov_str}'
    try:
        model = smf.ols(formula, data=df).fit()
    except Exception as e:
        return {"error": str(e)}
    R["model"] = model

    at3      = anova_lm(model, typ=3)
    ss_model = model.ess
    ss_error = model.ssr
    ss_ct    = ss_model + ss_error
    df_resid = int(model.df_resid)
    ms_error = ss_error / df_resid
    R["ms_error"] = ms_error
    R["df_resid"] = df_resid

    # ── Between-subjects effects table ─────────────────────────────────────────
    bse = []

    # Corrected Model
    df_mod = int(model.df_model)
    ms_mod = ss_model / df_mod
    f_mod  = ms_mod / ms_error
    p_mod  = float(1 - stats.f.cdf(f_mod, df_mod, df_resid))
    bse.append({"Source": "Corrected Model",
                "Type III SS": ss_model, "df": df_mod,
                "MS": ms_mod, "F": f_mod, "Sig.": p_mod,
                "Partial η²": ss_model / ss_ct,
                "Observed Power": obs_power(f_mod, df_mod, df_resid, alpha)})

    # Intercept  (SPSS Type III: SS_intercept / MS_error)
    if "Intercept" in at3.index:
        r    = at3.loc["Intercept"]
        ss_i = float(r["sum_sq"]); df_i = int(r["df"]); ms_i = ss_i / df_i
        # SPSS uses ms_i / ms_error  — same as statsmodels F column
        f_i  = float(r["F"]); p_i = float(r["PR(>F)"])
        bse.append({"Source": "Intercept",
                    "Type III SS": ss_i, "df": df_i, "MS": ms_i,
                    "F": f_i, "Sig.": p_i,
                    "Partial η²": ss_i / (ss_i + ss_error),
                    "Observed Power": obs_power(f_i, df_i, df_resid, alpha)})

    # Covariates
    for cov in covs:
        key = None
        for idx in at3.index:
            if cov in idx and "Intercept" not in idx and grp not in idx:
                key = idx; break
        if key:
            r    = at3.loc[key]
            ss_c = float(r["sum_sq"]); df_c = int(r["df"]); ms_c = ss_c / df_c
            f_c  = float(r["F"]); p_c  = float(r["PR(>F)"])
            bse.append({"Source": cov,
                        "Type III SS": ss_c, "df": df_c, "MS": ms_c,
                        "F": f_c, "Sig.": p_c,
                        "Partial η²": ss_c / (ss_c + ss_error),
                        "Observed Power": obs_power(f_c, df_c, df_resid, alpha)})

    # Factor (group)
    grp_key = None
    for idx in at3.index:
        if grp in idx and "Intercept" not in idx:
            grp_key = idx; break
    if grp_key:
        r    = at3.loc[grp_key]
        ss_g = float(r["sum_sq"]); df_g = int(r["df"]); ms_g = ss_g / df_g
        f_g  = float(r["F"]); p_g  = float(r["PR(>F)"])
        eta2_g = ss_g / (ss_g + ss_error)
        pw_g   = obs_power(f_g, df_g, df_resid, alpha)
        bse.append({"Source": grp,
                    "Type III SS": ss_g, "df": df_g, "MS": ms_g,
                    "F": f_g, "Sig.": p_g,
                    "Partial η²": eta2_g, "Observed Power": pw_g})
        R.update({"f_g": f_g, "p_g": p_g, "df1_g": df_g,
                  "df2_g": df_resid, "eta2_g": eta2_g, "pw_g": pw_g})
    else:
        R.update({"f_g": np.nan, "p_g": np.nan})

    # Error
    bse.append({"Source": "Error",
                "Type III SS": ss_error, "df": df_resid, "MS": ms_error,
                "F": np.nan, "Sig.": np.nan, "Partial η²": np.nan, "Observed Power": np.nan})
    # Corrected Total
    bse.append({"Source": "Corrected Total",
                "Type III SS": ss_ct, "df": n_total - 1,
                "MS": np.nan, "F": np.nan, "Sig.": np.nan,
                "Partial η²": np.nan, "Observed Power": np.nan})

    R["bse"] = pd.DataFrame(bse)
    R["model_summary"] = {
        "R": float(np.sqrt(model.rsquared)),
        "R²": float(model.rsquared),
        "Adj R²": float(model.rsquared_adj),
        "Std. Error of Estimate": float(np.sqrt(ms_error)),
        "F": float(model.fvalue), "df1": int(model.df_model),
        "df2": df_resid, "Sig.": float(model.f_pvalue), "N": n_total,
    }

    # ── Contrast vectors & (X'X)^-1  (SPSS-exact SE formula) ─────────────────
    try:
        contrasts, XtX_inv, cov_means = build_contrasts(model, df, grp, covs, groups)
    except Exception:
        # fallback: build manually via patsy
        dm_full = patsy.dmatrix(
            " + ".join([f'C(Q("{grp}"))'] + [f'Q("{c}")' for c in covs]),
            data=df, return_type="matrix")
        X_arr   = np.array(dm_full)
        XtX_inv = np.linalg.inv(X_arr.T @ X_arr)
        cov_means = {c: float(df[c].mean()) for c in covs}
        contrasts = {}
        for g in groups:
            row = pd.DataFrame({grp: [g], **{c: [cov_means[c]] for c in covs}})
            dm_row = patsy.dmatrix(dm_full.design_info, row, return_type="matrix")
            contrasts[g] = np.array(dm_row)[0]

    R["contrasts"]  = contrasts
    R["XtX_inv"]    = XtX_inv
    R["cov_means"]  = cov_means

    # ── Estimated Marginal Means  (SPSS-exact) ─────────────────────────────────
    t_crit = float(stats.t.ppf(1 - alpha / 2, df_resid))
    emm = []
    for g in groups:
        c_vec = contrasts[g]
        mu    = float(c_vec @ model.params.values)
        # SPSS SE: sqrt( c'(X'X)^-1 c * MS_error )
        se_g  = float(np.sqrt(c_vec @ XtX_inv @ c_vec * ms_error))
        emm.append({
            "Group": g,
            "Mean": mu,
            "Std. Error": se_g,
            f"{int((1-alpha)*100)}% CI Lower": mu - t_crit * se_g,
            f"{int((1-alpha)*100)}% CI Upper": mu + t_crit * se_g,
        })
    R["emm"] = pd.DataFrame(emm)

    # ── Pairwise comparisons: LSD / Bonferroni / Sidak  (SPSS Compare Main Effects) ──
    pairs = list(itertools.combinations(groups, 2))
    k     = len(pairs)

    pw_lsd  = []
    pw_bonf = []
    pw_sidak= []

    for g1, g2 in pairs:
        c1   = contrasts[g1]; c2 = contrasts[g2]
        diff = float((c1 - c2) @ model.params.values)
        c_d  = c1 - c2
        # SPSS SE for difference: sqrt( (c1-c2)'(X'X)^-1(c1-c2) * MS_error )
        se_d = float(np.sqrt(c_d @ XtX_inv @ c_d * ms_error))
        t_d  = diff / se_d if se_d > 0 else np.nan
        p_lsd_val  = float(2 * (1 - stats.t.cdf(abs(t_d), df_resid)))
        p_bonf_val = float(min(p_lsd_val * k, 1.0))
        p_sidak_val= float(1 - (1 - p_lsd_val) ** k)
        ci_l = diff - t_crit * se_d
        ci_u = diff + t_crit * se_d

        base = {"(I) Group": g1, "(J) Group": g2,
                "Mean Diff (I-J)": diff, "Std. Error": se_d,
                "95% CI Lower": ci_l, "95% CI Upper": ci_u}

        pw_lsd.append({**base, "Sig. (LSD)": p_lsd_val})
        pw_bonf.append({**base, "Sig. (Bonferroni)": p_bonf_val})
        pw_sidak.append({**base, "Sig. (Sidak)": p_sidak_val})

        # Also add reverse pair (j vs i) — SPSS shows both directions
        base_r = {"(I) Group": g2, "(J) Group": g1,
                  "Mean Diff (I-J)": -diff, "Std. Error": se_d,
                  "95% CI Lower": -ci_u, "95% CI Upper": -ci_l}
        pw_lsd.append({**base_r, "Sig. (LSD)": p_lsd_val})
        pw_bonf.append({**base_r, "Sig. (Bonferroni)": p_bonf_val})
        pw_sidak.append({**base_r, "Sig. (Sidak)": p_sidak_val})

    R["pw_lsd"]   = pd.DataFrame(pw_lsd).sort_values(["(I) Group","(J) Group"]).reset_index(drop=True)
    R["pw_bonf"]  = pd.DataFrame(pw_bonf).sort_values(["(I) Group","(J) Group"]).reset_index(drop=True)
    R["pw_sidak"] = pd.DataFrame(pw_sidak).sort_values(["(I) Group","(J) Group"]).reset_index(drop=True)

    # ── Assumption tests ───────────────────────────────────────────────────────
    resid = model.resid.values

    sw_s, sw_p = stats.shapiro(resid)
    R["shapiro"] = {"W": float(sw_s), "Sig.": float(sw_p), "pass": float(sw_p) > alpha}

    grp_vals = [df[df[grp] == g][dep].values for g in groups]
    lev_f, lev_p = stats.levene(*grp_vals)
    R["levene"] = {"F": float(lev_f), "Sig.": float(lev_p), "pass": float(lev_p) > alpha}

    # Homogeneity of regression slopes
    int_terms  = " + ".join([f'C(Q("{grp}")):Q("{c}")' for c in covs])
    formula_int = formula + " + " + int_terms
    try:
        model_int = smf.ols(formula_int, data=df).fit()
        at3_int   = anova_lm(model_int, typ=3)
        slopes = []
        for idx in at3_int.index:
            if ":" in idx and grp in idx:
                r = at3_int.loc[idx]
                slopes.append({"Term": idx, "F": float(r["F"]),
                                "df": int(r["df"]), "Sig.": float(r["PR(>F)"]),
                                "pass": float(r["PR(>F)"]) > alpha})
        R["slopes"] = slopes
    except Exception:
        R["slopes"] = []

    if len(covs) > 1:
        X_df = df[covs].dropna()
        Xc   = sm.add_constant(X_df)
        R["vif"] = pd.DataFrame([
            {"Variable": Xc.columns[i],
             "VIF": float(variance_inflation_factor(Xc.values, i))}
            for i in range(1, Xc.shape[1])])

    return R


# ── Interpretation ─────────────────────────────────────────────────────────────
def interpret(R, dep, grp, covs, alpha):
    lines = []
    ms  = R["model_summary"]
    sig_m = ms["Sig."] < alpha
    lines.append(
        f"<b>Overall Model:</b> The ANCOVA model was "
        f"{'statistically significant' if sig_m else 'not statistically significant'}, "
        f"F({ms['df1']}, {ms['df2']}) = {ms['F']:.3f}, "
        f"p {'< .001' if ms['Sig.'] < .001 else '= ' + _p(ms['Sig.'])}. "
        f"The model explained {ms['R²']*100:.1f}% of total variance in <i>{dep}</i> "
        f"(R² = {ms['R²']:.3f}, Adjusted R² = {ms['Adj R²']:.3f}), "
        f"a {'strong' if ms['R²']>.5 else 'moderate' if ms['R²']>.3 else 'weak'} fit."
    )

    f_g = R.get("f_g", np.nan); p_g = R.get("p_g", np.nan)
    if not np.isnan(f_g):
        eta2 = R.get("eta2_g", np.nan)
        lines.append(
            f"<b>Group Effect ({grp}):</b> After controlling for {', '.join(covs)}, "
            f"there was {'a statistically significant' if p_g < alpha else 'no statistically significant'} "
            f"effect of <i>{grp}</i> on <i>{dep}</i>, "
            f"F({R['df1_g']}, {R['df2_g']}) = {f_g:.3f}, "
            f"p {'< .001' if p_g < .001 else '= ' + _p(p_g)}, "
            f"partial η² = {eta2:.3f} ({eta_label(eta2)} effect). "
            + ("Groups differ significantly on adjusted means." if p_g < alpha
               else "No significant group differences after covariate adjustment.")
        )

    bse = R["bse"]
    for cov in covs:
        row = bse[bse["Source"] == cov]
        if not row.empty:
            fc = float(row["F"].iloc[0]); pc = float(row["Sig."].iloc[0])
            ec = float(row["Partial η²"].iloc[0])
            lines.append(
                f"<b>Covariate ({cov}):</b> <i>{cov}</i> was "
                f"{'a statistically significant' if pc < alpha else 'not a statistically significant'} "
                f"predictor of <i>{dep}</i>, F = {fc:.3f}, "
                f"p {'< .001' if pc < .001 else '= ' + _p(pc)}, "
                f"partial η² = {ec:.3f} ({eta_label(ec)} effect). "
                + ("The covariate reduced error variance and increased statistical power."
                   if pc < alpha else "The covariate did not significantly reduce error variance.")
            )

    pw_b = R["pw_bonf"]
    sig_b = pw_b[pw_b["Sig. (Bonferroni)"] < alpha]
    done  = set()
    pairs_txt = []
    for _, r in sig_b.iterrows():
        key = tuple(sorted([r["(I) Group"], r["(J) Group"]]))
        if key not in done:
            pairs_txt.append(
                f"{r['(I) Group']} vs. {r['(J) Group']} "
                f"(Δ = {r['Mean Diff (I-J)']:.3f}, p = {_p(r['Sig. (Bonferroni)'])})"
            )
            done.add(key)
    if pairs_txt:
        lines.append(f"<b>Pairwise Comparisons (Bonferroni):</b> Significant differences: {'; '.join(pairs_txt)}.")
    else:
        lines.append("<b>Pairwise Comparisons (Bonferroni):</b> No significant pairwise differences after Bonferroni correction.")

    sw = R["shapiro"]; lev = R["levene"]
    lines.append(
        f"<b>Assumption Checks:</b> "
        f"Residual normality (Shapiro-Wilk W = {sw['W']:.3f}, p = {_p(sw['Sig.'])}) was "
        f"{'satisfied ✓' if sw['pass'] else 'violated ✗ — interpret cautiously'}. "
        f"Homogeneity of variance (Levene F = {lev['F']:.3f}, p = {_p(lev['Sig.'])}) was "
        f"{'met ✓' if lev['pass'] else 'violated ✗'}."
    )
    slopes = R.get("slopes", [])
    if slopes:
        all_met = all(s["pass"] for s in slopes)
        lines.append(
            f"<b>Homogeneity of Regression Slopes:</b> The interaction between {grp} and "
            f"covariate(s) was {'non-significant — assumption met ✓' if all_met else 'significant — assumption violated ✗'}."
        )
    return lines


# ── Plots ──────────────────────────────────────────────────────────────────────
PAL = ["#1a3a5c","#0077b6","#00b4d8","#90e0ef","#caf0f8","#023e8a","#48cae4"]

def plot_emm(R, dep, grp, alpha):
    emm    = R["emm"]
    ci_l_c = [c for c in emm.columns if "Lower" in c][0]
    ci_u_c = [c for c in emm.columns if "Upper" in c][0]
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#f8fafc")
    ax.set_facecolor("#f8fafc")
    grps  = emm["Group"].values
    means = emm["Mean"].values
    lows  = emm[ci_l_c].values
    highs = emm[ci_u_c].values
    x     = np.arange(len(grps))
    bars  = ax.bar(x, means, color=[PAL[i % len(PAL)] for i in range(len(grps))],
                   alpha=0.88, edgecolor="white", linewidth=1.5, zorder=3)
    ax.errorbar(x, means, yerr=[means-lows, highs-means],
                fmt="none", color="#333", capsize=7, linewidth=1.5, zorder=4)
    for bar, m, h in zip(bars, means, highs):
        ax.text(bar.get_x()+bar.get_width()/2,
                h + (highs.max()-lows.min())*0.02,
                f"{m:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#1a1a2e")
    ax.set_xticks(x); ax.set_xticklabels(grps, fontsize=9)
    ax.set_ylabel(f"Adj. Mean of {dep}", fontsize=9)
    ax.set_xlabel(grp, fontsize=9)
    ax.set_title("Estimated Marginal Means", fontsize=11,
                 fontweight="bold", color="#1a3a5c", pad=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    return fig

def plot_scatter(df, dep, grp, cov):
    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor="#f8fafc")
    ax.set_facecolor("#f8fafc")
    for i, g in enumerate(sorted(df[grp].unique())):
        sub = df[df[grp]==g]
        ax.scatter(sub[cov], sub[dep], color=PAL[i%len(PAL)],
                   alpha=0.75, s=55, label=g, edgecolors="white", linewidths=0.5, zorder=3)
        m, b = np.polyfit(sub[cov].values, sub[dep].values, 1)
        xl = np.linspace(float(sub[cov].min()), float(sub[cov].max()), 100)
        ax.plot(xl, m*xl+b, color=PAL[i%len(PAL)], linewidth=2)
    ax.set_xlabel(cov, fontsize=9); ax.set_ylabel(dep, fontsize=9)
    ax.set_title(f"{dep} vs. {cov}", fontsize=11, fontweight="bold", color="#1a3a5c", pad=10)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=0.2, linestyle="--")
    plt.tight_layout()
    return fig

def plot_residuals(R):
    model  = R["model"]
    resid  = model.resid.values
    fitted = model.fittedvalues.values
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="#f8fafc")
    for ax in axes: ax.set_facecolor("#f8fafc")

    axes[0].scatter(fitted, resid, color=PAL[1], alpha=0.7, s=40,
                    edgecolors="white", linewidths=0.4, zorder=3)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("Fitted Values", fontsize=8)
    axes[0].set_ylabel("Residuals", fontsize=8)
    axes[0].set_title("Residuals vs. Fitted", fontsize=9, fontweight="bold", color="#1a3a5c")
    axes[0].spines[["top","right"]].set_visible(False)

    (osm, osr), (slope, intercept, _) = stats.probplot(resid)
    axes[1].plot(osm, osr, "o", color=PAL[1], alpha=0.7, markersize=5,
                 markeredgecolor="white", zorder=3)
    axes[1].plot(osm, slope*np.array(osm)+intercept, "r--", linewidth=1.5)
    axes[1].set_xlabel("Theoretical Quantiles", fontsize=8)
    axes[1].set_ylabel("Sample Quantiles", fontsize=8)
    axes[1].set_title("Normal Q-Q Plot", fontsize=9, fontweight="bold", color="#1a3a5c")
    axes[1].spines[["top","right"]].set_visible(False)

    axes[2].hist(resid, bins=10, color=PAL[1], alpha=0.8, edgecolor="white", linewidth=0.8)
    xr = np.linspace(resid.min(), resid.max(), 100)
    scale = len(resid)*(resid.max()-resid.min())/10
    axes[2].plot(xr, stats.norm.pdf(xr, resid.mean(), resid.std())*scale, "r--", linewidth=1.5)
    axes[2].set_xlabel("Residuals", fontsize=8)
    axes[2].set_ylabel("Frequency", fontsize=8)
    axes[2].set_title("Residual Distribution", fontsize=9, fontweight="bold", color="#1a3a5c")
    axes[2].spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    return fig

def fig_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ── PDF report ─────────────────────────────────────────────────────────────────
def build_pdf(R, df, dep, grp, covs, alpha, interps):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             rightMargin=1.8*cm, leftMargin=1.8*cm,
                             topMargin=2*cm, bottomMargin=2*cm)

    H1  = ParagraphStyle("H1", fontSize=13, fontName="Helvetica-Bold",
                          textColor=colors.white, backColor=colors.HexColor("#1a3a5c"),
                          spaceAfter=6, spaceBefore=12, borderPadding=(5,8,5,8))
    H2  = ParagraphStyle("H2", fontSize=10, fontName="Helvetica-Bold",
                          textColor=colors.HexColor("#1a3a5c"), spaceAfter=3, spaceBefore=8)
    BD  = ParagraphStyle("BD", fontSize=8.5, fontName="Helvetica", leading=13, spaceAfter=4)
    IT  = ParagraphStyle("IT", fontSize=8.5, fontName="Helvetica", leading=13,
                          backColor=colors.HexColor("#e8f4f8"),
                          borderPadding=(5,8,5,8), spaceAfter=5)
    NT  = ParagraphStyle("NT", fontSize=7.5, fontName="Helvetica-Oblique",
                          textColor=colors.HexColor("#6c757d"), spaceAfter=4)
    TIT = ParagraphStyle("TIT", fontSize=18, fontName="Helvetica-Bold",
                          textColor=colors.HexColor("#1a3a5c"), alignment=TA_CENTER)
    SUB = ParagraphStyle("SUB", fontSize=10, fontName="Helvetica",
                          textColor=colors.HexColor("#6c757d"),
                          alignment=TA_CENTER, spaceAfter=20)

    TS = TableStyle([
        ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR",(0,0),(-1,0), colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTNAME",(0,1),(-1,-1),"Helvetica"),
        ("FONTSIZE",(0,0),(-1,-1),7.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#f0f4f8")]),
        ("GRID",(0,0),(-1,-1),0.5, colors.HexColor("#d0d7de")),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),
        ("ALIGN",(0,0),(0,-1),"LEFT"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),4),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),
        ("LEFTPADDING",(0,0),(-1,-1),6),
        ("RIGHTPADDING",(0,0),(-1,-1),6),
    ])

    def mk_tbl(rows):
        t = Table(rows, repeatRows=1)
        t.setStyle(TS)
        return t

    def df_to_rows(data_df, num_cols=None, p_cols=None):
        num_cols = num_cols or []
        p_cols   = p_cols or []
        rows     = [list(data_df.columns)]
        for _, row in data_df.iterrows():
            r = []
            for col, val in zip(data_df.columns, row.values):
                if col in p_cols:
                    r.append(_p(float(val)) if not (isinstance(val,float) and np.isnan(val)) else ".")
                elif col in num_cols:
                    r.append(_f(float(val)) if not (isinstance(val,float) and np.isnan(val)) else ".")
                else:
                    r.append("." if (isinstance(val,float) and np.isnan(val)) else str(val))
            rows.append(r)
        return rows

    story = []
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("ANALYSIS OF COVARIANCE (ANCOVA)", TIT))
    story.append(Paragraph("SPSS-Equivalent Statistical Report", SUB))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0077b6")))
    story.append(Spacer(1, 6))

    meta = [["Dependent Variable:", dep], ["Fixed Factor:", grp],
            ["Covariate(s):", ", ".join(covs)], ["N:", str(len(df))],
            ["α:", str(alpha)], ["Date:", datetime.now().strftime("%B %d, %Y  %H:%M")]]
    mt = Table(meta, colWidths=[4*cm, 12*cm])
    mt.setStyle(TableStyle([("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
                             ("FONTNAME",(1,0),(1,-1),"Helvetica"),
                             ("FONTSIZE",(0,0),(-1,-1),8.5),
                             ("TEXTCOLOR",(0,0),(0,-1),colors.HexColor("#1a3a5c")),
                             ("TOPPADDING",(0,0),(-1,-1),3),
                             ("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    story.append(mt)
    story.append(Spacer(1,6))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#d0d7de")))

    # 1. Model summary
    story.append(Paragraph("  1. MODEL SUMMARY", H1))
    ms = R["model_summary"]
    story.append(mk_tbl(df_to_rows(pd.DataFrame([{
        "R": _f(ms["R"]), "R²": _f(ms["R²"]), "Adj R²": _f(ms["Adj R²"]),
        "Std. Error": _f(ms["Std. Error of Estimate"]),
        "F": _f(ms["F"]), "df1": ms["df1"], "df2": ms["df2"],
        "Sig.": _p(ms["Sig."])}]))))
    story.append(Paragraph("Note. R² = proportion of variance explained.", NT))

    # 2. Between-subjects effects
    story.append(Paragraph("  2. TESTS OF BETWEEN-SUBJECTS EFFECTS", H1))
    story.append(Paragraph(f"Dependent Variable: {dep}", BD))
    bse = R["bse"]
    nc  = ["Type III SS","MS","F","Partial η²","Observed Power"]
    story.append(mk_tbl(df_to_rows(bse, num_cols=nc, p_cols=["Sig."])))
    story.append(Paragraph(
        f"Note. Type III SS. α = {alpha}. "
        f"R² = {_f(ms['R²'])} (Adj R² = {_f(ms['Adj R²'])})", NT))

    # 3. Descriptives
    story.append(Paragraph("  3. DESCRIPTIVE STATISTICS", H1))
    nd = ["Mean","Std. Deviation","Std. Error","95% CI Lower","95% CI Upper",
          "Minimum","Maximum","Skewness","Kurtosis"]
    story.append(mk_tbl(df_to_rows(R["desc"], num_cols=nd)))

    # 4. EMM
    story.append(Paragraph("  4. ESTIMATED MARGINAL MEANS", H1))
    story.append(Paragraph(f"Dependent Variable: {dep}", BD))
    emm_nc = [c for c in R["emm"].columns if c != "Group"]
    story.append(mk_tbl(df_to_rows(R["emm"], num_cols=emm_nc)))
    cov_note = "; ".join([f"{c} = {v:.3f}" for c,v in R["cov_means"].items()])
    story.append(Paragraph(f"Note. Covariates at mean values: {cov_note}", NT))

    # 5. Pairwise — LSD
    story.append(Paragraph("  5a. PAIRWISE COMPARISONS — LSD (No Adjustment)", H1))
    story.append(Paragraph(f"Dependent Variable: {dep}", BD))
    pw_nc = ["Mean Diff (I-J)","Std. Error","95% CI Lower","95% CI Upper"]
    story.append(mk_tbl(df_to_rows(R["pw_lsd"], num_cols=pw_nc, p_cols=["Sig. (LSD)"])))
    story.append(Paragraph("Note. No adjustment for multiple comparisons.", NT))

    # 5b. Bonferroni
    story.append(Paragraph("  5b. PAIRWISE COMPARISONS — BONFERRONI", H1))
    story.append(Paragraph(f"Dependent Variable: {dep}", BD))
    story.append(mk_tbl(df_to_rows(R["pw_bonf"], num_cols=pw_nc, p_cols=["Sig. (Bonferroni)"])))
    story.append(Paragraph("Note. Adjustment: Bonferroni.", NT))

    # 5c. Sidak
    story.append(Paragraph("  5c. PAIRWISE COMPARISONS — SIDAK", H1))
    story.append(Paragraph(f"Dependent Variable: {dep}", BD))
    story.append(mk_tbl(df_to_rows(R["pw_sidak"], num_cols=pw_nc, p_cols=["Sig. (Sidak)"])))
    story.append(Paragraph("Note. Adjustment: Šidák.", NT))

    # 6. Assumptions
    story.append(Paragraph("  6. ASSUMPTION TESTS", H1))
    sw = R["shapiro"]
    story.append(Paragraph("a) Shapiro-Wilk (Normality of Residuals)", H2))
    story.append(mk_tbl(df_to_rows(pd.DataFrame([{
        "W": _f(sw["W"]), "Sig.": _p(sw["Sig."]),
        "Result": "Met" if sw["pass"] else "Violated"}]))))
    lev = R["levene"]
    story.append(Paragraph("b) Levene's Test (Homogeneity of Variances)", H2))
    story.append(mk_tbl(df_to_rows(pd.DataFrame([{
        "F": _f(lev["F"]), "Sig.": _p(lev["Sig."]),
        "Result": "Met" if lev["pass"] else "Violated"}]))))
    if R.get("slopes"):
        story.append(Paragraph("c) Homogeneity of Regression Slopes", H2))
        story.append(mk_tbl(df_to_rows(pd.DataFrame([
            {"Term":s["Term"],"F":_f(s["F"]),"df":s["df"],
             "Sig.":_p(s["Sig."]),"Result":"Met" if s["pass"] else "Violated"}
            for s in R["slopes"]]))))
    if "vif" in R:
        story.append(Paragraph("d) VIF (Multicollinearity)", H2))
        vif = R["vif"].copy(); vif["VIF"] = vif["VIF"].apply(_f)
        story.append(mk_tbl(df_to_rows(vif)))

    # 7. Interpretation
    story.append(Paragraph("  7. INTERPRETATION", H1))
    for line in interps:
        clean = (line.replace("<b>","").replace("</b>","")
                     .replace("<i>","").replace("</i>","")
                     .replace("✓","").replace("✗",""))
        story.append(Paragraph(clean, IT))

    # 8. Figures
    story.append(PageBreak())
    story.append(Paragraph("  8. FIGURES", H1))
    try:
        f1 = plot_emm(R, dep, grp, alpha)
        b1 = fig_bytes(f1); plt.close(f1)
        story.append(Image(io.BytesIO(b1), width=14*cm, height=8.5*cm))
        story.append(Paragraph("Figure 1. Estimated Marginal Means with 95% CI.", NT))
        story.append(Spacer(1,8))
        f2 = plot_residuals(R)
        b2 = fig_bytes(f2); plt.close(f2)
        story.append(Image(io.BytesIO(b2), width=17*cm, height=5*cm))
        story.append(Paragraph("Figure 2. Residual diagnostic plots.", NT))
    except Exception:
        pass

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#d0d7de")))
    story.append(Paragraph(
        "Generated by ANCOVA Analysis App · SPSS GLM conventions · "
        "Type III SS · SE via c'(X'X)⁻¹c·MS_error", NT))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ── HTML table renderer ────────────────────────────────────────────────────────
def html_tbl(df_in):
    html = '<div class="spss-wrap"><table class="spss-tbl"><thead><tr>'
    for col in df_in.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    for _, row in df_in.iterrows():
        html += "<tr>"
        for val in row.values:
            html += f"<td>{val}</td>"
        html += "</tr>"
    html += "</tbody></table></div>"
    return html

def fmt_bse(df_in):
    out = []
    for _, r in df_in.iterrows():
        out.append({
            "Source": r["Source"],
            "Type III SS": _f(r["Type III SS"]),
            "df": str(int(r["df"])) if not (isinstance(r["df"],float) and np.isnan(r["df"])) else ".",
            "MS": _f(r["MS"]), "F": _f(r["F"]), "Sig.": _p(r["Sig."]),
            "Partial η²": _f(r["Partial η²"]),
            "Observed Power": _f(r["Observed Power"]),
        })
    return pd.DataFrame(out)

def fmt_desc(df_in):
    d = df_in.copy()
    for c in ["Mean","Std. Deviation","Std. Error","95% CI Lower","95% CI Upper",
              "Minimum","Maximum","Skewness","Kurtosis"]:
        d[c] = d[c].apply(_f)
    return d

def fmt_emm(df_in):
    d = df_in.copy()
    for c in [col for col in d.columns if col != "Group"]:
        d[c] = d[c].apply(_f)
    return d

def fmt_pw(df_in, sig_col):
    d = df_in.copy()
    for c in ["Mean Diff (I-J)","Std. Error","95% CI Lower","95% CI Upper"]:
        d[c] = d[c].apply(_f)
    d[sig_col] = d[sig_col].apply(_p)
    return d


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="main-hdr">
        <h1>📊 ANCOVA Analysis Suite</h1>
        <p>Analysis of Covariance · SPSS General Linear Model (GLM) · Type III Sum of Squares</p>
    </div>""", unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Setup")
        st.markdown("---")
        st.markdown("**📄 Sample Data Format**")
        st.markdown("""
Your CSV needs:
- **1 numeric** dependent variable column
- **1 categorical** group/factor column
- **1+ numeric** covariate column(s)

*Example:* `group, pretest, posttest`
""")
        st.download_button("⬇️ Download Sample CSV", SAMPLE_CSV.encode(),
                           "ancova_sample_data.csv", "text/csv", use_container_width=True)
        st.markdown("---")

        uploaded = st.file_uploader("📂 Upload Your CSV", type=["csv"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.success(f"✅ {len(df)} rows × {len(df.columns)} cols")
            except Exception as e:
                st.error(f"Read error: {e}"); df = None
        else:
            df = pd.read_csv(io.StringIO(SAMPLE_CSV))
            st.info("ℹ️ Using built-in sample data")

        if df is not None:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            all_cols = df.columns.tolist()
            st.markdown("---")
            dep_var = st.selectbox("🎯 Dependent Variable", num_cols,
                                    index=num_cols.index("posttest") if "posttest" in num_cols else 0)
            grp_cands = [c for c in all_cols if c != dep_var]
            group_var = st.selectbox("👥 Fixed Factor (Group)", grp_cands,
                                      index=grp_cands.index("group") if "group" in grp_cands else 0)
            cov_cands = [c for c in num_cols if c != dep_var]
            covariates = st.multiselect("📐 Covariate(s)", cov_cands,
                                         default=["pretest"] if "pretest" in cov_cands
                                         else cov_cands[:1] if cov_cands else [])
            alpha = st.selectbox("α Level", [0.05, 0.01, 0.001], index=0)
            st.markdown("---")
            run_btn = st.button("🚀 Run ANCOVA", type="primary", use_container_width=True)
        else:
            run_btn = False

    if df is None:
        return

    with st.expander("🔍 Data Preview", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Rows", len(df)); c2.metric("Cols", len(df.columns))
        c3.metric("Groups", df[group_var].nunique() if "group_var" in dir() else "—")
        c4.metric("Missing", int(df.isnull().sum().sum()))

    if not run_btn and "R" not in st.session_state:
        st.info("👈 Select variables in the sidebar, then click **Run ANCOVA**.")
        return

    if run_btn:
        if not covariates:
            st.error("⚠️ Select at least one covariate."); return
        with st.spinner("Running ANCOVA…"):
            R = run_ancova(df, dep_var, group_var, covariates, alpha)
        if "error" in R:
            st.error(f"Analysis error: {R['error']}"); return
        st.session_state["R"]      = R
        st.session_state["params"] = (dep_var, group_var, covariates, alpha)
        st.session_state["df"]     = df

    R      = st.session_state.get("R")
    params = st.session_state.get("params")
    df     = st.session_state.get("df", df)
    if R is None or params is None: return
    dep_var, group_var, covariates, alpha = params

    interps = interpret(R, dep_var, group_var, covariates, alpha)
    ms = R["model_summary"]

    st.success("✅ ANCOVA complete!")

    # Quick metrics
    cols = st.columns(5)
    for col, (val, lbl) in zip(cols, [
        (f"{ms['F']:.3f}", f"Model F({ms['df1']},{ms['df2']})"),
        (_p(ms["Sig."]),   "Model Sig."),
        (f"{ms['R²']:.3f}","R-squared"),
        (f"{ms['Adj R²']:.3f}", "Adj R²"),
        (_f(R.get("eta2_g",np.nan)), f"Partial η² ({group_var})"),
    ]):
        col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div>'
                     f'<div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    t1,t2,t3,t4,t5,t6,t7,t8 = st.tabs([
        "📋 Between-Subjects","📊 Descriptives","🎯 Marginal Means",
        "🔀 Pairwise: LSD","🔀 Pairwise: Bonferroni","🔀 Pairwise: Sidak",
        "⚖️ Assumptions","📈 Plots & Interpretation"
    ])

    # Tab 1 — Between-Subjects Effects
    with t1:
        st.markdown(f'<div class="sec-title">Tests of Between-Subjects Effects · Dependent Variable: {dep_var}</div>',
                    unsafe_allow_html=True)
        st.markdown(html_tbl(fmt_bse(R["bse"])), unsafe_allow_html=True)
        st.markdown(f'<p class="note-txt">Type III SS. α={alpha}. '
                    f'R²={_f(ms["R²"])} (Adj R²={_f(ms["Adj R²"])})</p>', unsafe_allow_html=True)

        st.markdown('<div class="sec-title">Model Summary</div>', unsafe_allow_html=True)
        st.markdown(html_tbl(pd.DataFrame([{
            "R":_f(ms["R"]),"R²":_f(ms["R²"]),"Adj R²":_f(ms["Adj R²"]),
            "Std. Error of Estimate":_f(ms["Std. Error of Estimate"]),
            "F":_f(ms["F"]),"df1":ms["df1"],"df2":ms["df2"],
            "Sig.":_p(ms["Sig."]),"N":ms["N"]}])), unsafe_allow_html=True)

    # Tab 2 — Descriptives
    with t2:
        st.markdown('<div class="sec-title">Descriptive Statistics</div>', unsafe_allow_html=True)
        st.markdown(html_tbl(fmt_desc(R["desc"])), unsafe_allow_html=True)

    # Tab 3 — EMM
    with t3:
        st.markdown(f'<div class="sec-title">Estimated Marginal Means · Dependent Variable: {dep_var}</div>',
                    unsafe_allow_html=True)
        st.markdown(html_tbl(fmt_emm(R["emm"])), unsafe_allow_html=True)
        cov_note = "; ".join([f"{c} = {v:.3f}" for c,v in R["cov_means"].items()])
        st.markdown(f'<p class="note-txt">Covariates at mean values: {cov_note}</p>', unsafe_allow_html=True)

    # Tab 4 — LSD
    with t4:
        st.markdown(f'<div class="sec-title">Pairwise Comparisons — LSD (No Adjustment) · {dep_var}</div>',
                    unsafe_allow_html=True)
        st.markdown(html_tbl(fmt_pw(R["pw_lsd"],"Sig. (LSD)")), unsafe_allow_html=True)
        st.markdown('<p class="note-txt">Based on estimated marginal means. No adjustment for multiple comparisons.</p>',
                    unsafe_allow_html=True)

    # Tab 5 — Bonferroni
    with t5:
        st.markdown(f'<div class="sec-title">Pairwise Comparisons — Bonferroni · {dep_var}</div>',
                    unsafe_allow_html=True)
        st.markdown(html_tbl(fmt_pw(R["pw_bonf"],"Sig. (Bonferroni)")), unsafe_allow_html=True)
        st.markdown('<p class="note-txt">Adjustment: Bonferroni. p_adj = min(p_LSD × k, 1) where k = number of pairs.</p>',
                    unsafe_allow_html=True)

    # Tab 6 — Sidak
    with t6:
        st.markdown(f'<div class="sec-title">Pairwise Comparisons — Šidák · {dep_var}</div>',
                    unsafe_allow_html=True)
        st.markdown(html_tbl(fmt_pw(R["pw_sidak"],"Sig. (Sidak)")), unsafe_allow_html=True)
        st.markdown('<p class="note-txt">Adjustment: Šidák. p_adj = 1 − (1 − p_LSD)^k.</p>',
                    unsafe_allow_html=True)

    # Tab 7 — Assumptions
    with t7:
        c1,c2 = st.columns(2)
        with c1:
            sw = R["shapiro"]
            res = '<span class="pass">✓ Met</span>' if sw["pass"] else '<span class="fail">✗ Violated</span>'
            st.markdown('<div class="sec-title">Shapiro-Wilk — Normality of Residuals</div>', unsafe_allow_html=True)
            st.markdown(f'<table class="spss-tbl"><thead><tr><th>W</th><th>Sig.</th><th>Result</th></tr></thead>'
                        f'<tbody><tr><td>{_f(sw["W"])}</td><td>{_p(sw["Sig."])}</td>'
                        f'<td style="text-align:left">{res}</td></tr></tbody></table>',
                        unsafe_allow_html=True)
            if not sw["pass"]:
                st.markdown('<div class="warn-box">⚠️ Normality violated. Consider transformations.</div>',
                            unsafe_allow_html=True)
        with c2:
            lev = R["levene"]
            res = '<span class="pass">✓ Met</span>' if lev["pass"] else '<span class="fail">✗ Violated</span>'
            n_g = df[group_var].nunique()
            st.markdown('<div class="sec-title">Levene\'s Test — Homogeneity of Variances</div>', unsafe_allow_html=True)
            st.markdown(f'<table class="spss-tbl"><thead><tr><th>F</th><th>df1</th><th>df2</th>'
                        f'<th>Sig.</th><th>Result</th></tr></thead>'
                        f'<tbody><tr><td>{_f(lev["F"])}</td><td>{n_g-1}</td><td>{len(df)-n_g}</td>'
                        f'<td>{_p(lev["Sig."])}</td><td style="text-align:left">{res}</td>'
                        f'</tr></tbody></table>', unsafe_allow_html=True)

        slopes = R.get("slopes",[])
        if slopes:
            st.markdown('<div class="sec-title">Homogeneity of Regression Slopes</div>', unsafe_allow_html=True)
            rows = []
            for s in slopes:
                r = '<span class="pass">✓ Met</span>' if s["pass"] else '<span class="fail">✗ Violated</span>'
                rows.append({"Term":s["Term"],"F":_f(s["F"]),"df":s["df"],
                             "Sig.":_p(s["Sig."]),"Result":r})
            st.markdown(html_tbl(pd.DataFrame(rows)), unsafe_allow_html=True)
            st.markdown('<p class="note-txt">Non-significant interaction → equal slopes across groups (ANCOVA assumption met).</p>',
                        unsafe_allow_html=True)

        if "vif" in R:
            st.markdown('<div class="sec-title">Multicollinearity — VIF</div>', unsafe_allow_html=True)
            vd = R["vif"].copy()
            vd["VIF"] = vd["VIF"].apply(_f)
            vd["Verdict"] = R["vif"]["VIF"].apply(
                lambda v: '<span class="pass">Acceptable (&lt;5)</span>'
                if v<5 else '<span class="fail">Concerning (&gt;5)</span>')
            st.markdown(html_tbl(vd), unsafe_allow_html=True)

    # Tab 8 — Plots & Interpretation
    with t8:
        pc1,pc2 = st.columns([1.2,1])
        with pc1:
            f1 = plot_emm(R,dep_var,group_var,alpha)
            st.pyplot(f1, use_container_width=True); plt.close(f1)
        with pc2:
            for cov in covariates:
                f2 = plot_scatter(df,dep_var,group_var,cov)
                st.pyplot(f2, use_container_width=True); plt.close(f2)
        f3 = plot_residuals(R)
        st.pyplot(f3, use_container_width=True); plt.close(f3)

        st.markdown("---")
        st.markdown("### 📝 Statistical Interpretation")
        for line in interps:
            st.markdown(f'<div class="interp-box">{line}</div>', unsafe_allow_html=True)

        f_g=R.get("f_g",np.nan); p_g=R.get("p_g",np.nan)
        eta2=R.get("eta2_g",np.nan)
        apa = (f"A one-way ANCOVA examined the effect of {group_var} on {dep_var} "
               f"after controlling for {', '.join(covariates)}. "
               f"The result was {'significant' if not np.isnan(p_g) and p_g<alpha else 'not significant'}, "
               f"F({R.get('df1_g','?')}, {R.get('df2_g','?')}) = {_f(f_g,2)}, "
               f"p {'< .001' if not np.isnan(p_g) and p_g<.001 else '= '+_p(p_g)}, "
               f"partial η² = {_f(eta2)} ({eta_label(eta2)} effect).")
        st.markdown("**APA 7th Write-Up:**")
        st.code(apa, language=None)

    # ── Downloads ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Download Results")
    dc1,dc2,dc3 = st.columns(3)

    with dc1:
        with st.spinner("Building PDF…"):
            pdf_data = build_pdf(R,df,dep_var,group_var,covariates,alpha,interps)
        st.download_button("📄 PDF Report", pdf_data,
                           f"ANCOVA_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                           "application/pdf", use_container_width=True)
    with dc2:
        xbuf = io.BytesIO()
        with pd.ExcelWriter(xbuf, engine="openpyxl") as writer:
            R["bse"].to_excel(writer,  sheet_name="Between-Subjects Effects",  index=False)
            R["desc"].to_excel(writer, sheet_name="Descriptive Statistics",    index=False)
            R["emm"].to_excel(writer,  sheet_name="Estimated Marginal Means",  index=False)
            R["pw_lsd"].to_excel(writer,   sheet_name="Pairwise LSD",          index=False)
            R["pw_bonf"].to_excel(writer,  sheet_name="Pairwise Bonferroni",   index=False)
            R["pw_sidak"].to_excel(writer, sheet_name="Pairwise Sidak",        index=False)
            pd.DataFrame([R["shapiro"]]).to_excel(writer, sheet_name="Shapiro-Wilk", index=False)
            pd.DataFrame([R["levene"]]).to_excel(writer,  sheet_name="Levene",       index=False)
            if R.get("slopes"):
                pd.DataFrame(R["slopes"]).to_excel(writer, sheet_name="Regression Slopes", index=False)
        xbuf.seek(0)
        st.download_button("📊 Excel Workbook", xbuf.getvalue(),
                           f"ANCOVA_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    with dc3:
        csv_parts = [
            "=== BETWEEN-SUBJECTS EFFECTS ===\n"  + R["bse"].to_csv(index=False),
            "=== DESCRIPTIVE STATISTICS ===\n"     + R["desc"].to_csv(index=False),
            "=== ESTIMATED MARGINAL MEANS ===\n"   + R["emm"].to_csv(index=False),
            "=== PAIRWISE LSD ===\n"               + R["pw_lsd"].to_csv(index=False),
            "=== PAIRWISE BONFERRONI ===\n"        + R["pw_bonf"].to_csv(index=False),
            "=== PAIRWISE SIDAK ===\n"             + R["pw_sidak"].to_csv(index=False),
        ]
        st.download_button("📝 CSV Tables", "\n\n".join(csv_parts).encode(),
                           f"ANCOVA_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           "text/csv", use_container_width=True)


if __name__ == "__main__":
    main()
