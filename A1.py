import pandas as pd
from scipy.optimize import brentq
import numpy as np
import matplotlib.pyplot as plt
import math

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

def bond_list(df):
    selected_isins = [
    "CA135087L518",
    "CA135087L930",
    "CA135087M847",
    "CA135087N837",
    "CA135087P576",
    "CA135087Q491",
    "CA135087Q988",
    "CA135087R895",
    "CA135087S471",
    "CA135087T388",
    "CA135087T792"
]
    df["ISIN"] = df["ISIN"].astype(str).str.strip()
    df_selected = (
    df[df["ISIN"].isin(selected_isins)]
    .copy()
    .sort_values(["ISIN", "Date"])
    .reset_index(drop=True)
)
    return df_selected

def last_next_coupon_dates(settle, maturity, months=6):
    settle = pd.Timestamp(settle)
    maturity = pd.Timestamp(maturity)

    next_cp = maturity
    while next_cp > settle:
        last_cp = next_cp - pd.DateOffset(months=months)
        if last_cp <= settle:
            return last_cp, next_cp
        next_cp = last_cp
    return pd.NaT, pd.NaT

def dirty_price_and_f(clean_price, coupon_rate, settle, maturity, F=100.0):
    settle = pd.Timestamp(settle)
    maturity = pd.Timestamp(maturity)

    last_cp, next_cp = last_next_coupon_dates(settle, maturity)
    if pd.isna(last_cp) or pd.isna(next_cp):
        return np.nan, np.nan

    C = F * coupon_rate / 2.0
    days_since = (settle - last_cp).days
    days_period = (next_cp - last_cp).days

    f = (next_cp - settle).days / days_period
    AI = C * (days_since / days_period)

    return clean_price + AI, f

def ytm_from_dirty_price(clean_price, coupon_rate, settle, maturity, F=100.0):
    settle = pd.Timestamp(settle)
    maturity = pd.Timestamp(maturity)

    C = F * coupon_rate / 2.0
    dirty_price, f = dirty_price_and_f(clean_price, coupon_rate, settle, maturity, F=F)
    if np.isnan(dirty_price) or np.isnan(f):
        return np.nan
    
    _ , next_cp = last_next_coupon_dates(settle, maturity)
    N = 0
    d = next_cp
    while d <= maturity:
        N += 1
        d = d + pd.DateOffset(months=6)

    def price_from_y(y):
        pv = 0.0
        for k in range(N):
            pv += C / (1 + y/2.0)**(f + k)
        pv += F / (1 + y/2.0)**(f + N - 1)
        return pv

    def root(y):
        return price_from_y(y) - dirty_price

    return brentq(root, 1e-10, 1.0), dirty_price, f

def ytm(df, F=100.0):
    df = df.copy()
    ytm_list = []
    dirty_list = []
    f_list = []

    for _, r in df.iterrows():
        clean = float(r['Close']) * F
        coupon = float(r["Coupon"])
        settle = r["Date"]
        mat = r["Maturity Date"]
        y, dirty_price, f = ytm_from_dirty_price(clean, coupon, settle, mat, F=F)
        ytm_list.append(y)
        dirty_list.append(dirty_price)
        f_list.append(f)

    df["YTM"] = ytm_list
    df['Dirty'] = dirty_list
    df['Fraction'] = f_list

    return df

def linear_interp_maturity_ytm(df_with_ytm,
                                       targets_years=(1, 2, 3, 4, 5),
                                       date_col="Date",
                                       mat_col="Maturity Date",
                                       ytm_col="YTM"):
    
    dff = df_with_ytm.copy()
    dff[date_col] = pd.to_datetime(dff[date_col], errors="coerce")
    dff[mat_col] = pd.to_datetime(dff[mat_col], errors="coerce")

    dff["t_years"] = (dff[mat_col] - dff[date_col]).dt.days / 365.25

    dff = dff.dropna(subset=[date_col, "t_years", ytm_col])
    dff = dff[(dff["t_years"] > 0) & np.isfinite(dff[ytm_col])]

    targets = np.array(targets_years, dtype=float)

    rows = []
    for day, g in dff.groupby(dff[date_col].dt.date):
        g = g.sort_values("t_years")

        t = g["t_years"].to_numpy(dtype=float)
        y = g[ytm_col].to_numpy(dtype=float)

        if targets.min() < t.min() or targets.max() > t.max():
            continue

        y_target = np.interp(targets, t, y) 

        row = {"Date": pd.to_datetime(str(day))}
        for T, val in zip(targets, y_target):
            row[f"YTM_{int(T)}Y"] = val
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("Date").set_index("Date")
    return out

def plot_constant_maturity_ytm_curves(ytm_curve_1to5, filename="ytm_curves.png"):
    plt.figure(figsize=(8, 4))
    x = np.array([1, 2, 3, 4, 5], dtype=float)

    for day, row in ytm_curve_1to5.iterrows():
        y = row[[f"YTM_{k}Y" for k in range(1, 6)]].to_numpy(dtype=float)
        plt.plot(x, y, marker="o", label=str(day.date()))

    plt.xlabel("Maturity (years)")
    plt.ylabel("YTM (annual)")
    plt.title("Interpolated YTM Curves (1-5Y) for each day")
    plt.xticks([1,2,3,4,5], ["1Y","2Y","3Y","4Y","5Y"])
    plt.grid(True)
    plt.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def plot_ytm_curves(df_with_ytm):
    dff = df_with_ytm.dropna(subset=["Maturity Date", "YTM"]).copy()

    dff["Date"] = pd.to_datetime(dff["Date"], errors="coerce")
    dff["Maturity Date"] = pd.to_datetime(dff["Maturity Date"], errors="coerce")

    plt.figure()
    for day, g in dff.groupby(dff["Date"].dt.date):
        g = g.sort_values("Maturity Date")
        plt.plot(g["Maturity Date"], g["YTM"], marker="o", label=str(day))

    plt.xlabel("Maturity Date")
    plt.ylabel("Yield to Maturity")
    plt.title("YTM Curves by Maturity Date for each day")
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.show()

def bootstrapping_from_dirty_price(bonds, F=100.0):
    spot_rates = []

    for i, (price, coupon_rate, maturity, f) in enumerate(sorted(bonds, key=lambda x: x[2])):
        periods = math.floor(maturity) 
        coupon = F * coupon_rate / 2
        cash_flows = np.array([coupon] * periods)
        time_periods = np.arange(f, periods + f)

        if i == 0:
            spot_rate = 2 * (((coupon + F)/price)**(1 / f) - 1)
        else:
            discounted_cash_flows = [cf / (1 + (spot_rates[j] / 2))**time_periods[j] for j, cf in enumerate(cash_flows)]
            discounted_cash_flows = np.sum(discounted_cash_flows)
            residual = price - discounted_cash_flows
            spot_rate = 2 * (((coupon + F) / residual)**(1 / maturity) - 1)

        spot_rates.append(spot_rate)
    return spot_rates

def boostrapping_data(df, F=100):

    df = df.copy()
    df["ToM"] = (df["Maturity Date"] - df["Date"]).dt.days / 365.25

    for _, g in df.groupby(df["Date"].dt.date):
        g = g.sort_values("Maturity Date")
    
        bonds = list(zip(
            g["Dirty"].astype(float),
            g["Coupon"].astype(float),
            2*g["ToM"].astype(float),
            g['Fraction'].astype(float)
        ))
        
        spot_rates = bootstrapping_from_dirty_price(bonds, F)
        df.loc[g.index, "Spot Rate"] = spot_rates
    
    return df

def plot_sr_curves(df_with_sr):
    dff = df_with_sr.dropna(subset=["Maturity Date", "Spot Rate"]).copy()

    dff["Date"] = pd.to_datetime(dff["Date"], errors="coerce")
    dff["Maturity Date"] = pd.to_datetime(dff["Maturity Date"], errors="coerce")

    plt.figure()
    for day, g in dff.groupby(dff["Date"].dt.date):
        g = g.sort_values("Maturity Date")
        plt.plot(g["Maturity Date"], g["Spot Rate"], marker="o", label=str(day))

    plt.xlabel("Maturity Date")
    plt.ylabel("Spot Rate")
    plt.title("Spot Rate by Maturity Date for each day")
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.show()

def plot_interpolated_spot_curves(interp_spot_df, filename="spot_curves.png"):
    if interp_spot_df.empty:
        print("No interpolated spot points to plot.")
        return

    plt.figure(figsize=(8, 4))
    for day, g in interp_spot_df.groupby(interp_spot_df["Day"].dt.date):
        g = g.sort_values("Maturity (years)")
        plt.plot(g["Maturity (years)"], g["Spot Rate (interp)"],
                 marker="o", label=str(day))

    plt.xlabel("Maturity (years)")
    plt.ylabel("Spot Rate (annual)")
    plt.title("Interpolated Spot Curves (1-5Y) for each day")
    plt.xticks([1, 2, 3, 4, 5], ["1Y", "2Y", "3Y", "4Y", "5Y"])
    plt.grid(True)
    plt.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

def interp_spot_and_forward(df_with_sr,
                                   settle_col="Date",
                                   maturity_col="Maturity Date",
                                   spot_col="Spot Rate",
                                   targets=(1., 2., 3., 4., 5.)):
    """
    For each day:
      1) build (t_years, spot(t)) points from bootstrapped spot rates
      2) linearly interpolate spot to t = 1,2,3,4,5
      3) convert to DF(t) = 1/(1+spot/2)^(2t)  (semiannual comp convention)
      4) compute 1Y forwards: f(t,t+1) = DF(t)/DF(t+1) - 1 for t=1..4

    Returns:
      fwd_df: tidy DataFrame [Day, Start (years), End (years), 1Y Forward Rate]
      spot_interp_df: tidy DataFrame [Day, Maturity (years), Spot Rate (interp)]
    """
    dff = df_with_sr.dropna(subset=[spot_col, maturity_col, settle_col]).copy()
    dff[settle_col] = pd.to_datetime(dff[settle_col], errors="coerce")
    dff[maturity_col] = pd.to_datetime(dff[maturity_col], errors="coerce")

    targets = np.array(targets, dtype=float)  # default 1..5
    fwd_out = []
    spot_out = []

    for day, g in dff.groupby(dff[settle_col].dt.date):
        t_years = (g[maturity_col] - g[settle_col]).dt.days / 365.25
        sr = g[spot_col].astype(float)

        order = np.argsort(t_years.to_numpy())
        t_sorted = t_years.to_numpy()[order]
        sr_sorted = sr.to_numpy()[order]

        if targets.min() < t_sorted.min() or targets.max() > t_sorted.max():
            continue

        sr_t = np.interp(targets, t_sorted, sr_sorted)

        spot_tmp = pd.DataFrame({
            "Day": pd.to_datetime(str(day)),
            "Maturity (years)": targets,
            "Spot Rate (interp)": sr_t
        })
        spot_out.append(spot_tmp)

        DF = 1.0 / (1.0 + sr_t/2.0) ** (2.0 * targets)

        fwd = DF[:-1] / DF[1:] - 1.0
        starts = targets[:-1]           
        ends = starts + 1.0     

        fwd_tmp = pd.DataFrame({
            "Day": pd.to_datetime(str(day)),
            "Start (years)": starts,
            "End (years)": ends,
            "1Y Forward Rate": fwd
        })
        fwd_out.append(fwd_tmp)

    fwd_df = (pd.concat(fwd_out, ignore_index=True)
              if fwd_out else pd.DataFrame(columns=["Day","Start (years)","End (years)","1Y Forward Rate"]))

    spot_interp_df = (pd.concat(spot_out, ignore_index=True)
                      if spot_out else pd.DataFrame(columns=["Day","Maturity (years)","Spot Rate (interp)"]))

    return fwd_df, spot_interp_df

def plot_forward_curves(fwd_df, filename="forward_curves.png"):
    if fwd_df.empty:
        print("No forward curve points to plot.")
        return

    plt.figure(figsize=(8, 4))
    for day, g in fwd_df.groupby(fwd_df["Day"].dt.date):
        g = g.sort_values("Start (years)")
        plt.plot(g["Start (years)"], g["1Y Forward Rate"], marker="o", label=str(day))

    plt.xlabel("Forward rate between each year (1->2, 2->3, 3->4 and 4->5)")
    plt.ylabel("1-year forward rate")
    plt.title("1-Year Forward Curves for each day")
    plt.xticks([1, 2, 3, 4], ["1yr-1yr", "1yr-2yr", "1yr-3yr", "1yr-4yr"])
    plt.grid(True)
    plt.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

def log_returns_from_levels(R):
    """
    X_{i,j} = log(r_{i,j+1} / r_{i,j})
    Returns X with one fewer row than R.
    """
    # require strictly positive levels for log
    if (R <= 0).any().any():
        raise ValueError("Found non-positive levels; cannot compute log-returns.")
    X = np.log(R / R.shift(1)).dropna()
    return X

def covariance_matrix(X):
    """Sample covariance of columns."""
    return X.cov()

def eig_full_table(Cov):
    """
    Input:
      Cov : pandas DataFrame (symmetric covariance matrix)

    Output:
      eval_df : DataFrame with eigenvalues and variance explained
      evec_df : DataFrame with eigenvectors (columns = v1, v2, ...)
    """
    A = Cov.to_numpy(dtype=float)
    labels = list(Cov.columns)

    # Eigen-decomposition
    evals, evecs = np.linalg.eigh(A)          
    idx = np.argsort(evals)[::-1]             
    evals = evals[idx]
    evecs = evecs[:, idx]                 

    # Variance explained
    var_exp = evals / evals.sum()

    # Table of eigenvalues
    eval_df = pd.DataFrame({
        "eigenvalue": evals,
        "variance_explained": var_exp,
    }, index=[f"v{k+1}" for k in range(len(evals))])

    # Table of eigenvectors
    evec_df = pd.DataFrame(
        evecs,
        index=labels,
        columns=[f"v{k+1}" for k in range(len(evals))]
    )

    return eval_df, evec_df

def build_forward_levels_wide(fwd_df,
                              day_col="Day",
                              start_col="Start (years)",
                              fwd_col="1Y Forward Rate"):
    """
    Builds wide levels matrix for forwards:
      cols = ["1Y1Y","2Y1Y","3Y1Y","4Y1Y"]
    """
    dff = fwd_df.copy()
    dff[day_col] = pd.to_datetime(dff[day_col], errors="coerce")

    label_map = {1.0:"1Y1Y", 2.0:"2Y1Y", 3.0:"3Y1Y", 4.0:"4Y1Y"}
    dff["Tenor"] = dff[start_col].map(label_map)

    F = (dff.pivot_table(index=day_col, columns="Tenor", values=fwd_col, aggfunc="first")
           .sort_index())

    F = F[["1Y1Y","2Y1Y","3Y1Y","4Y1Y"]].dropna(axis=0, how="any")
    return F

df = pd.read_excel('/Users/bonita/Downloads/APM466/Cleaned Canadian Bond.xlsx')
df["Maturity Date"] = pd.to_datetime(df["Maturity Date"], errors="coerce")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

selected_bonds = bond_list(df)
df_all = ytm(selected_bonds)
ytm_curve_1to5 = linear_interp_maturity_ytm(df_all)
plot_ytm_curves(df_all)
plot_constant_maturity_ytm_curves(ytm_curve_1to5)

df_all = boostrapping_data(df_all)
fwd_df, spot_interp_df = interp_spot_and_forward(df_all)
plot_sr_curves(df_all)
plot_interpolated_spot_curves(spot_interp_df)
plot_forward_curves(fwd_df)

print(df_all)

R_y = (ytm_curve_1to5
       .copy()
       .rename(columns={f"YTM_{k}Y": f"{k}Y" for k in range(1, 6)}))

R_y = R_y[["1Y","2Y","3Y","4Y","5Y"]].dropna().sort_index()
X_y = log_returns_from_levels(R_y)
Cov_y = covariance_matrix(X_y)

print("Yield log-return covariance (5x5) [interpolated 1Y..5Y]:")
print(Cov_y)

R_f = build_forward_levels_wide(fwd_df).dropna().sort_index()
X_f = log_returns_from_levels(R_f)
Cov_f = covariance_matrix(X_f)

print("\nForward log-return covariance (4x4):")
print(Cov_f)

# Yields
evals_y_df, evecs_y_df = eig_full_table(Cov_y)

print("YIELD eigenvalues + variance explained:")
print(evals_y_df)

print("\nYIELD eigenvectors:")
print(evecs_y_df)

# Forwards
evals_f_df, evecs_f_df = eig_full_table(Cov_f)

print("\nFWD eigenvalues + variance explained:")
print(evals_f_df)

print("\nFWD eigenvectors:")
print(evecs_f_df)