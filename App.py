import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from datetime import datetime, time, timedelta

st.set_page_config(page_title="Workflow Analysis Hartac Hector Morales", layout="wide")

st.title("Workflow Analysis Dashboard - Dev: Hector M")

# Sidebar menu
option = st.sidebar.radio(
    "Choose Analysis:",
    ("Sales Order Workflow", "Work Order Workflow")
)

# -------------------
# Helper functions
# -------------------
def calculate_business_hours_duration(
    start_datetime,
    end_datetime,
    work_start_time=time(6, 30),
    work_end_time=time(17, 0)
):
    if pd.isna(start_datetime) or pd.isna(end_datetime):
        return pd.NaT

    if isinstance(start_datetime, str):
        start_datetime = pd.to_datetime(start_datetime)

    if isinstance(end_datetime, str):
        end_datetime = pd.to_datetime(end_datetime)

    if end_datetime <= start_datetime:
        return timedelta(0)

    total_business_time = timedelta(0)
    current_date = start_datetime.date()
    end_date = end_datetime.date()

    while current_date <= end_date:
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue

        day_start = datetime.combine(current_date, work_start_time)
        day_end = datetime.combine(current_date, work_end_time)

        actual_start = max(start_datetime, day_start) if current_date == start_datetime.date() else day_start
        actual_end = min(end_datetime, day_end) if current_date == end_datetime.date() else day_end

        if actual_start < actual_end:
            total_business_time += actual_end - actual_start

        current_date += timedelta(days=1)

    return total_business_time


def calculate_status_durations(df):
    df = df.copy()

    # Keep this column numeric from the beginning.
    # Do not initialise with pd.NaT because that can make pandas infer a datetime/timedelta dtype.
    df["Status_Duration_Hours"] = np.nan
    df["Status_Duration_Hours"] = df["Status_Duration_Hours"].astype("float64")

    for doc_num, group in df.groupby("Document Number"):
        group = group.sort_values("Date", ascending=False)

        for i in range(len(group) - 1):
            current_row_idx = group.index[i]
            next_row_idx = group.index[i + 1]

            start_time = group.loc[next_row_idx, "Date"]
            end_time = group.loc[current_row_idx, "Date"]

            duration = calculate_business_hours_duration(start_time, end_time)

            if not pd.isna(duration):
                df.at[next_row_idx, "Status_Duration_Hours"] = duration.total_seconds() / 3600

    return df


# -------------------
# Sales Order Workflow
# -------------------
if option == "Sales Order Workflow":
    st.header("Sales Order Workflow Analysis")

    uploaded_file = st.file_uploader(
        "Upload CSV from Statuses Change Bottle Neck Analysis Saved Search",
        type="csv",
        key="so"
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        required_columns = [
            "Date",
            "Created From",
            "Document Number",
            "Old Value",
            "New Value"
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.stop()

        # Convert Date to datetime
        df["Date"] = pd.to_datetime(
            df["Date"],
            format="%d/%m/%Y %I:%M %p",
            errors="coerce"
        )

        df = df.dropna(subset=["Date"]).copy()

        # Clean and transform
        df["Created From"] = df["Created From"].astype(str).str.replace("Quote #", "", regex=False)
        df.sort_values(by=["Document Number", "Date"], ascending=False, inplace=True)

        df.rename(
            columns={
                "Old Value": "Finishing",
                "New Value": "Starting"
            },
            inplace=True
        )

        # Remove duplicate document numbers within Created From groups
        grouped_unique = df.groupby("Created From")["Document Number"].apply(
            lambda x: sorted(list(x.unique()))
        )

        multiple_different_detail = grouped_unique[grouped_unique.apply(len) > 1]

        documents_to_delete = []

        for doc_list in multiple_different_detail:
            documents_to_delete.extend(doc_list[1:])

        df_cleaned = df.loc[~df["Document Number"].isin(documents_to_delete)].copy()

        # Replace Document Number with Created From where available
        df_cleaned.loc[
            df_cleaned["Created From"].notna(),
            "Document Number"
        ] = df_cleaned.loc[
            df_cleaned["Created From"].notna(),
            "Created From"
        ]

        df_cleaned.sort_values(by=["Document Number", "Date"], ascending=False, inplace=True)

        # Fill missing Finishing values from next Starting
        df_cleaned["Finishing"] = df_cleaned.groupby("Document Number")["Starting"].shift(-1).where(
            df_cleaned["Finishing"].isna(),
            df_cleaned["Finishing"]
        )

        # Remove documents with weekend activity
        df_cleaned["is_weekend"] = df_cleaned["Date"].dt.weekday >= 5
        weekend_docs = df_cleaned[df_cleaned["is_weekend"]]["Document Number"].unique()
        df_cleaned = df_cleaned.loc[~df_cleaned["Document Number"].isin(weekend_docs)].copy()
        df_cleaned.drop(["is_weekend"], axis=1, inplace=True)

        # Remove documents with activity outside working hours
        df_cleaned["hour"] = df_cleaned["Date"].dt.hour
        df_cleaned["minute"] = df_cleaned["Date"].dt.minute

        outside_hours = df_cleaned[
            (df_cleaned["hour"] < 6) |
            (df_cleaned["hour"] > 17) |
            ((df_cleaned["hour"] == 6) & (df_cleaned["minute"] < 30))
        ]

        outside_docs = outside_hours["Document Number"].unique()
        df_cleaned = df_cleaned.loc[~df_cleaned["Document Number"].isin(outside_docs)].copy()
        df_cleaned.drop(["hour", "minute"], axis=1, inplace=True)

        # Apply durations
        df_with_durations = calculate_status_durations(df_cleaned)

        filtered_df = df_with_durations.loc[
            df_with_durations["Status_Duration_Hours"].notna()
        ].copy()

        filtered_df["Status_Duration_Hours"] = pd.to_numeric(
            filtered_df["Status_Duration_Hours"],
            errors="coerce"
        )

        # Add Year-Month and filter
        filtered_df["Year-Month"] = filtered_df["Date"].dt.to_period("M").astype(str)

        filtered_df = filtered_df.loc[
            (filtered_df["Starting"] != "Reviewed by Purchasing") &
            (filtered_df["Starting"] != "Quote Sent") &
            (filtered_df["Year-Month"] != "2025-03")
        ].copy()

        if filtered_df.empty:
            st.warning("No valid Sales Order workflow records found after filtering.")
        else:
            # -------------------
            # Transactions over 2 hours per status
            # -------------------
            threshold_hours = 2

            over_threshold_df = filtered_df.loc[
                filtered_df["Status_Duration_Hours"] > threshold_hours
            ].copy()

            st.subheader("Sales Orders Over 2 Hours by Status")

            if over_threshold_df.empty:
                st.success("No Sales Order status tasks exceeded 2 business hours.")
            else:
                over_threshold_df["Status_Duration_Hours"] = over_threshold_df[
                    "Status_Duration_Hours"
                ].round(2)

                summary_over_threshold = (
                    over_threshold_df
                    .groupby("Starting")
                    .agg(
                        Transactions_Over_1_5h=("Document Number", "nunique"),
                        Transaction_Numbers=(
                            "Document Number",
                            lambda x: ", ".join(sorted(x.astype(str).unique()))
                        ),
                        Average_Hours=("Status_Duration_Hours", "mean"),
                        Max_Hours=("Status_Duration_Hours", "max")
                    )
                    .reset_index()
                    .rename(
                        columns={
                            "Starting": "Status"
                        }
                    )
                )

                summary_over_threshold["Average_Hours"] = summary_over_threshold[
                    "Average_Hours"
                ].round(2)

                summary_over_threshold["Max_Hours"] = summary_over_threshold[
                    "Max_Hours"
                ].round(2)

                st.dataframe(
                    summary_over_threshold,
                    use_container_width=True,
                    hide_index=True
                )

                with st.expander("Show transaction-level detail over 1.5 hours"):
                    detail_over_threshold = over_threshold_df[
                        [
                            "Document Number",
                            "Starting",
                            "Finishing",
                            "Date",
                            "Status_Duration_Hours",
                            "Year-Month"
                        ]
                    ].sort_values(
                        by=["Starting", "Status_Duration_Hours"],
                        ascending=[True, False]
                    )

                    detail_over_threshold = detail_over_threshold.rename(
                        columns={
                            "Document Number": "Transaction Number",
                            "Starting": "Status",
                            "Finishing": "Next Status",
                            "Status_Duration_Hours": "Duration Hours"
                        }
                    )

                    st.dataframe(
                        detail_over_threshold,
                        use_container_width=True,
                        hide_index=True
                    )

            # -------------------
            # Pivot
            # -------------------
            result = (
                filtered_df
                .groupby(["Year-Month", "Starting"])["Status_Duration_Hours"]
                .mean()
                .reset_index()
            )

            pivot = result.pivot(
                index="Year-Month",
                columns="Starting",
                values="Status_Duration_Hours"
            )

            st.subheader("Pivot Table (SO)")
            st.dataframe(pivot)

            if pivot.empty or pivot.dropna(how="all").empty:
                st.warning("Pivot table has no numeric data to plot.")
            else:
                fig, ax = plt.subplots(figsize=(12, 6))
                pivot.plot(kind="bar", ax=ax)

                plt.title("Average Status Duration Hours per Month per Status")
                plt.xlabel("Year-Month")
                plt.ylabel("Average Status Duration (Hours)")
                plt.xticks(rotation=45)
                plt.legend(title="Status")
                plt.tight_layout()

                for container in ax.containers:
                    ax.bar_label(
                        container,
                        fmt="%.2f",
                        rotation=90,
                        padding=2,
                        fontsize=8
                    )

                st.pyplot(fig)


# -------------------
# Work Order Workflow
# -------------------
elif option == "Work Order Workflow":
    st.header("WO Creation - SO Job Card Printing")

    uploaded_file = st.file_uploader(
        "Upload CSV from Production station change Bottle Neck Analysis Saved Search",
        type="csv",
        key="wo"
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        required_columns = [
            "Document Number",
            "New Value",
            "Date",
            "End Date",
            "Date.1"
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.stop()

        # Sort by Document Number
        df.sort_values(by=["Document Number"], ascending=False, inplace=True)

        # Clean New Value before mapping.
        # This fixes cases where the CSV reads values as 1, 100, 1.0, 100.0, or strings.
        df["New Value Clean"] = pd.to_numeric(df["New Value"], errors="coerce").astype("Int64")

        stage_map = {
            1: "Planning",
            100: "Finishing"
        }

        df["Stage Number"] = df["New Value Clean"].map(stage_map)

        # Helpful diagnostics in Streamlit
        with st.expander("WO data check"):
            st.write("Rows by mapped Stage Number:")
            st.write(df["Stage Number"].value_counts(dropna=False))

            st.write("Raw New Value samples:")
            st.write(df["New Value"].dropna().astype(str).value_counts().head(20))

        # Convert dates safely
        df["WO created"] = pd.to_datetime(
            df["Date"],
            format="%d/%m/%Y",
            errors="coerce"
        )

        df["Production Due Date"] = pd.to_datetime(
            df["End Date"],
            format="%d/%m/%Y",
            errors="coerce"
        )

        df["Job Card Printed"] = pd.to_datetime(
            df["Date.1"],
            dayfirst=True,
            errors="coerce"
        ).dt.normalize()

        # Keep only Finishing rows
        df = df.loc[df["Stage Number"] == "Finishing"].copy()

        # Remove rows where any required dates could not be parsed
        df = df.dropna(
            subset=[
                "WO created",
                "Production Due Date",
                "Job Card Printed"
            ]
        ).copy()

        if df.empty:
            st.warning(
                "No valid Finishing rows found after filtering. "
                "Check whether New Value contains 100 for Finishing and whether Date, End Date, and Date.1 are valid."
            )
        else:
            # Business-day calculations.
            # np.busday_count counts business days from start date to end date.
            df["Days from Printed to Due date"] = np.busday_count(
                df["Job Card Printed"].values.astype("datetime64[D]"),
                df["Production Due Date"].values.astype("datetime64[D]")
            )

            df["Days from Created to Due date"] = np.busday_count(
                df["WO created"].values.astype("datetime64[D]"),
                df["Production Due Date"].values.astype("datetime64[D]")
            )

            df["Days from Created to Printed"] = np.busday_count(
                df["WO created"].values.astype("datetime64[D]"),
                df["Job Card Printed"].values.astype("datetime64[D]")
            )

            # Create Month from WO created date
            df["Month"] = df["WO created"].dt.to_period("M").astype(str)

            pivot = df.pivot_table(
                index="Month",
                values=[
                    "Days from Printed to Due date",
                    "Days from Created to Due date",
                    "Days from Created to Printed"
                ],
                aggfunc="mean"
            ).reset_index()

            st.subheader("Pivot Table (WO)")
            st.dataframe(pivot)

            plot_df = pivot.set_index("Month")

            # Force numeric just before plotting to avoid:
            # TypeError: no numeric data to plot
            plot_df = plot_df.apply(pd.to_numeric, errors="coerce")

            if plot_df.empty or plot_df.dropna(how="all").empty:
                st.warning("Pivot table has no numeric data to plot.")
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_df.plot(kind="bar", ax=ax)

                ax.set_xlabel("WO Created Month")
                ax.set_ylabel("Average Business Days")
                ax.set_title("Average Business-Day Intervals per Month")

                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()

                for container in ax.containers:
                    ax.bar_label(container, fmt="%.2f")

                st.pyplot(fig)

            # Optional detailed data table
            with st.expander("Show cleaned WO detail data"):
                st.dataframe(df)
