# ==============================================================================
# STUDENT PERFORMANCE ANALYSIS (Compatible with Google Colab, Jupyter & Terminal)
# ==============================================================================

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Automatically detect interactive notebook environment (Colab / Jupyter)
IN_NOTEBOOK = "ipykernel" in sys.modules or "google.colab" in sys.modules

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ==============================================================================
# 1. SYNTHETIC DATA GENERATOR (NUMPY RANDOM DISTRIBUTIONS)
# ==============================================================================

SUBJECT_CREDITS = {
    "math_score": 4,
    "science_score": 4,
    "english_score": 3,
    "history_score": 3,
    "computer_science_score": 4,
}
SUBJECTS = list(SUBJECT_CREDITS.keys())


def generate_student_dataset(n_samples: int = 300, seed: int = 42, include_anomalies: bool = True) -> pd.DataFrame:
    """Generate realistic student dataset using NumPy probability distributions."""
    rng = np.random.default_rng(seed)

    # Identifiers & Demographics
    student_ids = [f"STD-{1000 + i}" for i in range(1, n_samples + 1)]
    first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Pat", "Riley", "Casey", "Aarav", "Priya", "Mei"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Patel", "Chen"]
    names = [f"{rng.choice(first_names)} {rng.choice(last_names)}" for _ in range(n_samples)]
    sections = rng.choice(["Section A", "Section B", "Section C"], size=n_samples, p=[0.35, 0.35, 0.30])
    gender = rng.choice(["Male", "Female", "Non-Binary"], size=n_samples, p=[0.48, 0.48, 0.04])

    # Behavioral Factors: Normal distribution for study hours, Beta distribution for attendance
    study_hours = np.clip(rng.normal(loc=14.0, scale=5.0, size=n_samples), 1.0, 38.0).round(1)
    attendance = np.clip(rng.beta(a=7.0, b=2.0, size=n_samples) * 100.0, 35.0, 100.0).round(1)
    ability = rng.normal(loc=0.0, scale=1.0, size=n_samples)

    # Realistic Correlated Exam Marks
    math = np.clip(52.0 + (study_hours * 1.3) + (attendance * 0.18) + (ability * 8.0) + rng.normal(0, 7, n_samples), 15, 99).round(1)
    science = np.clip(48.0 + (math * 0.35) + (study_hours * 0.9) + (attendance * 0.15) + (ability * 6.0) + rng.normal(0, 6.5, n_samples), 18, 99).round(1)
    english = np.clip(58.0 + (attendance * 0.22) + (ability * 5.5) + rng.normal(0, 7.5, n_samples), 20, 99).round(1)
    history = np.clip(55.0 + (study_hours * 0.8) + (attendance * 0.25) + rng.normal(0, 8, n_samples), 20, 99).round(1)
    cs = np.clip(50.0 + (math * 0.3) + (study_hours * 1.1) + (ability * 6.0) + rng.normal(0, 7, n_samples), 15, 100).round(1)

    df = pd.DataFrame({
        "student_id": student_ids,
        "name": names,
        "gender": gender,
        "section": sections,
        "study_hours_per_week": study_hours,
        "attendance_rate": attendance,
        "math_score": math,
        "science_score": science,
        "english_score": english,
        "history_score": history,
        "computer_science_score": cs,
    })

    # Inject realistic anomalies to demonstrate data cleaning
    if include_anomalies:
        n = len(df)
        dirty_idx = rng.choice(n, size=max(2, int(n * 0.04)), replace=False)
        for idx in dirty_idx:
            df.at[idx, "section"] = f"  {df.at[idx, 'section'].lower()}  "
            df.at[idx, "name"] = df.at[idx, "name"].lower()

        # Out-of-range mark typos (>100 or negative)
        df.at[rng.choice(n), "math_score"] = 108.0
        df.at[rng.choice(n), "science_score"] = -5.0
        df.at[rng.choice(n), "attendance_rate"] = 104.0

        # Random missing values
        for col in ["math_score", "science_score", "attendance_rate", "study_hours_per_week"]:
            df.loc[rng.choice(n, size=3, replace=False), col] = np.nan

        # Duplicate records
        duplicates = df.iloc[rng.choice(n, size=3, replace=False)].copy()
        df = pd.concat([df, duplicates], ignore_index=True)

    return df


# ==============================================================================
# 2. DATA CLEANING & VECTORIZED PREPROCESSING (PANDAS & NUMPY)
# ==============================================================================

def clean_data(df: pd.DataFrame, subjects: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean dataset: remove duplicates, normalize text, clamp ranges, and impute missing values."""
    audit: dict[str, Any] = {}
    cleaned = df.copy()

    # 1. Deduplication based on student_id
    initial_len = len(cleaned)
    cleaned["student_id"] = cleaned["student_id"].astype(str).str.strip().str.upper()
    cleaned = cleaned.drop_duplicates(subset=["student_id"], keep="first").reset_index(drop=True)
    audit["duplicates_removed"] = initial_len - len(cleaned)

    # 2. Text Normalization
    cleaned["name"] = cleaned["name"].astype(str).str.strip().str.title()
    cleaned["section"] = cleaned["section"].astype(str).str.strip().str.title()
    cleaned["gender"] = cleaned["gender"].astype(str).str.strip().str.title()
    cleaned = cleaned.replace(r"(?i)^(nan|none|null|\s*)$", np.nan, regex=True)

    # 3. Missing Value Imputation (Section Median)
    audit["imputed_columns"] = {}
    num_cols = subjects + ["attendance_rate", "study_hours_per_week"]
    for col in num_cols:
        if col in cleaned.columns:
            missing_count = int(cleaned[col].isna().sum())
            if missing_count > 0:
                sec_median = cleaned.groupby("section")[col].transform("median")
                cleaned[col] = cleaned[col].fillna(sec_median).fillna(cleaned[col].median())
                audit["imputed_columns"][col] = missing_count

    # 4. Out-of-Range Clamping using np.clip [0.0, 100.0]
    audit["clamped_columns"] = {}
    for col in subjects:
        out_of_bounds = int(((cleaned[col] < 0.0) | (cleaned[col] > 100.0)).sum())
        if out_of_bounds > 0:
            cleaned[col] = np.clip(cleaned[col].to_numpy(dtype=float), 0.0, 100.0)
            audit["clamped_columns"][col] = out_of_bounds

    if "attendance_rate" in cleaned.columns:
        cleaned["attendance_rate"] = np.clip(cleaned["attendance_rate"].to_numpy(dtype=float), 0.0, 100.0)
    if "study_hours_per_week" in cleaned.columns:
        cleaned["study_hours_per_week"] = np.clip(cleaned["study_hours_per_week"].to_numpy(dtype=float), 0.0, 80.0)

    return cleaned, audit


def engineer_features(df: pd.DataFrame, subjects: list[str], weights: dict[str, int]) -> pd.DataFrame:
    """NumPy vectorized feature engineering."""
    df = df.copy()

    # Total Marks and Percentage
    matrix = df[subjects].to_numpy(dtype=float)
    df["total_marks"] = np.sum(matrix, axis=1).round(1)
    df["percentage"] = ((df["total_marks"] / (len(subjects) * 100.0)) * 100.0).round(2)

    # Letter Grade Assignment via np.select
    pct = df["percentage"].to_numpy()
    grade_conditions = [pct >= 90.0, pct >= 80.0, pct >= 70.0, pct >= 60.0, pct >= 50.0]
    grade_choices = ["A+", "A", "B", "C", "D"]
    df["grade"] = np.select(grade_conditions, grade_choices, default="F")

    # Weighted GPA (4.0 scale) via np.select
    total_weights = sum(weights.get(s, 3) for s in subjects)
    weighted_pts = np.zeros(len(df), dtype=float)

    for subj in subjects:
        m = df[subj].to_numpy()
        gp_conds = [m >= 90.0, m >= 80.0, m >= 70.0, m >= 60.0, m >= 50.0, m >= 40.0]
        gp_choices = [4.0, 3.5, 3.0, 2.5, 2.0, 1.0]
        pts = np.select(gp_conds, gp_choices, default=0.0)
        weighted_pts += pts * weights.get(subj, 3)

    df["weighted_gpa"] = np.round(weighted_pts / total_weights, 2)

    # Pass/Fail & At-Risk Flags via np.where / np.select
    subj_passed = matrix >= 40.0
    df["pass_status"] = np.where(np.all(subj_passed, axis=1) & (pct >= 50.0), "Pass", "Fail")
    df["failed_subjects_count"] = np.sum(~subj_passed, axis=1)

    # At-risk classifications
    att = df["attendance_rate"].to_numpy()
    high_risk = (att < 75.0) & ((pct < 50.0) | (df["failed_subjects_count"] >= 2))
    mod_risk = (att < 75.0) | (pct < 60.0) | (df["failed_subjects_count"] >= 1)
    df["at_risk_level"] = np.select([high_risk, mod_risk], ["High Risk", "Moderate Risk"], default="Low Risk")
    df["is_at_risk"] = np.where(df["at_risk_level"] != "Low Risk", True, False)

    # Z-scores and Percentiles
    for s in subjects:
        std_val = float(df[s].std(ddof=1)) or 1.0
        df[f"{s}_zscore"] = np.round((df[s].to_numpy() - float(df[s].mean())) / std_val, 3)
        df[f"{s}_percentile"] = np.round(df[s].rank(pct=True).to_numpy() * 100.0, 1)

    return df


# ==============================================================================
# 3. STATISTICAL & COMPARATIVE ANALYSIS
# ==============================================================================

def run_analysis(df: pd.DataFrame, subjects: list[str]) -> dict[str, Any]:
    """Calculate summary statistics, difficulty indices, and correlations."""
    # Subject Statistics
    subj_stats = []
    for s in subjects:
        arr = df[s].to_numpy(dtype=float)
        q25, q75 = np.percentile(arr, [25, 75])
        subj_stats.append({
            "subject": s.replace("_score", "").title(),
            "subject_col": s,
            "mean": round(float(np.mean(arr)), 2),
            "median": round(float(np.median(arr)), 2),
            "std": round(float(np.std(arr, ddof=1)), 2),
            "q25": round(float(q25), 1),
            "q75": round(float(q75), 1),
            "iqr": round(float(q75 - q25), 1),
            "pass_rate_pct": round(float(np.mean(arr >= 40.0) * 100.0), 1),
        })
    subj_df = pd.DataFrame(subj_stats)

    # Composite Subject Difficulty Index
    diff_records = []
    for _, row in subj_df.iterrows():
        fail_rate = 100.0 - row["pass_rate_pct"]
        score = (0.50 * (100.0 - row["mean"])) + (0.30 * (fail_rate * 2.5)) + (0.20 * min(row["std"] * 2.0, 40.0))
        tier = "High" if score >= 35.0 else ("Moderate" if score >= 20.0 else "Low")
        diff_records.append({
            "subject": row["subject"],
            "mean_score": row["mean"],
            "difficulty_index": round(score, 2),
            "difficulty_tier": tier,
        })
    diff_df = pd.DataFrame(diff_records).sort_values(by="difficulty_index", ascending=False).reset_index(drop=True)
    diff_df["difficulty_rank"] = np.arange(1, len(diff_df) + 1)

    # Classroom Section Grouping
    sec_df = df.groupby("section").agg(
        student_count=("student_id", "count"),
        avg_percentage=("percentage", lambda x: round(x.mean(), 2)),
        avg_gpa=("weighted_gpa", lambda x: round(x.mean(), 2)),
        pass_rate_pct=("pass_status", lambda s: round((s == "Pass").mean() * 100.0, 1)),
        avg_attendance=("attendance_rate", lambda x: round(x.mean(), 1)),
        at_risk_count=("is_at_risk", "sum"),
    ).reset_index()

    top_df = df.sort_values(by=["percentage", "weighted_gpa"], ascending=[False, False]).head(5)
    at_risk_df = df[df["is_at_risk"]].sort_values(by="percentage", ascending=True)

    corr_cols = ["attendance_rate", "study_hours_per_week", "percentage", "weighted_gpa"] + subjects
    corr_matrix = df[corr_cols].corr().round(3)

    return {
        "subject_summary": subj_df,
        "difficulty_ranking": diff_df,
        "section_summary": sec_df,
        "top_performers": top_df,
        "at_risk_students": at_risk_df,
        "correlation_matrix": corr_matrix,
    }


# ==============================================================================
# 4. REPORTING & VISUALIZATION
# ==============================================================================

def print_report(df: pd.DataFrame, report: dict[str, Any], audit: dict[str, Any]) -> None:
    """Print formatted summary report."""
    sep = "=" * 76
    print("\n" + sep)
    print("                    STUDENT PERFORMANCE ANALYSIS REPORT                    ")
    print(sep)
    print(f"\n[1] DATA CLEANING AUDIT: {audit.get('duplicates_removed', 0)} duplicates removed | {len(audit.get('imputed_columns', {}))} columns imputed")
    print(f"\n[2] OVERVIEW: {len(df)} Students | Overall Pass Rate: {(df['pass_status'] == 'Pass').mean() * 100:.1f}%")
    print(f"    Mean Percentage: {df['percentage'].mean():.2f}% | Mean GPA: {df['weighted_gpa'].mean():.2f} / 4.0")
    print(f"    At-Risk Students: {df['is_at_risk'].sum()} ({df['is_at_risk'].mean() * 100:.1f}%)")

    print("\n[3] SUBJECT DIFFICULTY RANKING:")
    print(f"    {'Rank':<5} {'Subject':<18} {'Mean':>7} {'Diff.Index':>12} {'Tier':>10}")
    print("    " + "-" * 56)
    for _, r in report["difficulty_ranking"].iterrows():
        print(f"    #{r['difficulty_rank']:<4} {r['subject']:<18} {r['mean_score']:>7.2f} {r['difficulty_index']:>12.2f} {r['difficulty_tier']:>10}")

    print("\n[4] SECTION COMPARISON:")
    print(f"    {'Section':<14} {'Students':>9} {'Avg %':>8} {'Avg GPA':>8} {'Pass%':>8}")
    print("    " + "-" * 50)
    for _, r in report["section_summary"].iterrows():
        print(f"    {r['section']:<14} {r['student_count']:>9} {r['avg_percentage']:>8.2f} {r['avg_gpa']:>8.2f} {r['pass_rate_pct']:>7.1f}%")

    print("\n[5] TOP 3 ACHIEVERS:")
    for i, (_, r) in enumerate(report["top_performers"].head(3).iterrows(), 1):
        print(f"    #{i}. {r['name']} ({r['student_id']}) - {r['percentage']}% | GPA: {r['weighted_gpa']} | Grade: {r['grade']}")
    print(sep + "\n")


def export_and_visualize(df: pd.DataFrame, report: dict[str, Any], out_dir: Path) -> None:
    """Save cleaned datasets, tables, and render charts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "cleaned_student_performance.csv", index=False)
    report["subject_summary"].to_csv(out_dir / "subject_summary.csv", index=False)
    report["difficulty_ranking"].to_csv(out_dir / "subject_difficulty.csv", index=False)
    report["section_summary"].to_csv(out_dir / "section_summary.csv", index=False)
    report["at_risk_students"].to_csv(out_dir / "at_risk_students.csv", index=False)
    report["correlation_matrix"].to_csv(out_dir / "correlation_matrix.csv")

    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    if MATPLOTLIB_AVAILABLE:
        # Chart 1: Grade Bar Chart
        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        grade_order = ["A+", "A", "B", "C", "D", "F"]
        counts = df["grade"].value_counts()
        vals = [counts.get(g, 0) for g in grade_order]
        ax.bar(grade_order, vals, color="#2A9D8F")
        ax.set_title("Overall Grade Distribution", fontweight="bold")
        ax.set_ylabel("Student Count")
        fig.tight_layout()
        fig.savefig(charts_dir / "grade_distribution.png")
        if IN_NOTEBOOK:
            plt.show()
        plt.close(fig)

        # Chart 2: Attendance vs Percentage Scatter Plot
        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        ax.scatter(df["attendance_rate"], df["percentage"], c="#457B9D", alpha=0.7)
        ax.set_title("Attendance Rate vs Academic Percentage", fontweight="bold")
        ax.set_xlabel("Attendance Rate (%)")
        ax.set_ylabel("Overall Percentage (%)")
        fig.tight_layout()
        fig.savefig(charts_dir / "attendance_vs_scores.png")
        if IN_NOTEBOOK:
            plt.show()
        plt.close(fig)

        print(f"[Visuals] Matplotlib charts saved to: {charts_dir}")


# ==============================================================================
# 5. ENTRYPOINT (SAFE FOR GOOGLE COLAB, JUPYTER & TERMINAL)
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Student Performance Analysis")
    parser.add_argument("--data-path", "-d", type=str, default=None, help="Custom CSV path")
    parser.add_argument("--samples", "-n", type=int, default=300, help="Synthetic records count")
    parser.add_argument("--output-dir", "-o", type=str, default="outputs", help="Output directory")

    # KEY FIX: Use parse_known_args() so Google Colab/Jupyter kernel flags (-f ...) are ignored
    args, _ = parser.parse_known_args()

    # Load custom dataset or generate synthetic fallback
    if args.data_path and Path(args.data_path).exists():
        print(f"[Data] Loading custom CSV: {args.data_path}")
        raw_df = pd.read_csv(args.data_path)
    else:
        print(f"[Data] Generating synthetic fallback dataset ({args.samples} samples)...")
        raw_df = generate_student_dataset(n_samples=args.samples)

    # Pipeline execution
    cleaned_df, audit = clean_data(raw_df, SUBJECTS)
    processed_df = engineer_features(cleaned_df, SUBJECTS, SUBJECT_CREDITS)
    report = run_analysis(processed_df, SUBJECTS)

    # Print summary report & export data
    print_report(processed_df, report, audit)
    export_and_visualize(processed_df, report, Path(args.output_dir))
    print(f"[Done] Cleaned data and reports exported to: {Path(args.output_dir).resolve()}\n")


if __name__ == "__main__":
    main()