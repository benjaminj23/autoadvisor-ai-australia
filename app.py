import re
from datetime import datetime
from html import escape
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import streamlit as st

try:
    import requests
except ImportError:  # Streamlit usually includes requests, but deployments enjoy drama.
    requests = None


st.set_page_config(page_title="AutoAdvisor AI MVP v20", page_icon="🚗", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("data/cars_master_dataset.csv")


MODEL_LAUNCH_YEARS = {
    # Launch/availability guards for the seed dataset.
    # This keeps the MVP from recommending impossible combinations like "2012 Volvo XC40".
    ("Volvo", "XC40"): 2018,
    ("MG", "ZS EV"): 2020,
    ("BYD", "Atto 3"): 2022,
    ("BYD", "Seal"): 2023,
    ("GWM", "Ora"): 2023,
    ("Polestar", "2"): 2021,
    ("Tesla", "Model 3"): 2019,
    ("Tesla", "Model Y"): 2022,
    ("Toyota", "C-HR"): 2017,
    ("Toyota", "Yaris Cross"): 2020,
    ("Kia", "Seltos"): 2019,
    ("Hyundai", "Kona"): 2017,
    ("Mitsubishi", "Eclipse Cross"): 2018,
    ("Mazda", "CX-8"): 2018,
    ("Skoda", "Kodiaq"): 2017,
    ("LDV", "D90"): 2017,
}


def normalise_model_for_launch(model):
    model = str(model).strip()

    suffixes = [
        " Hybrid",
        " Electric",
        " Petrol",
        " Diesel",
        " PHEV",
        " Petrol Hybrid",
        " Petrol/Electric",
    ]

    for suffix in suffixes:
        if model.endswith(suffix):
            return model[: -len(suffix)].strip()

    return model


def get_model_launch_year(make, model):
    key = (str(make).strip(), str(model).strip())

    if key in MODEL_LAUNCH_YEARS:
        return MODEL_LAUNCH_YEARS[key]

    normalised_key = (str(make).strip(), normalise_model_for_launch(model))

    return MODEL_LAUNCH_YEARS.get(normalised_key)


def year_range_bounds(year_range):
    years = [int(x) for x in re.findall(r"\d{4}", str(year_range))]

    if len(years) >= 2:
        return years[0], years[1]

    if len(years) == 1:
        return years[0], years[0]

    return None, None


def clean_vehicle_dataset(raw_df):
    cleaned = raw_df.copy()

    rows_to_keep = []
    display_ranges = []

    for _, row in cleaned.iterrows():
        start_year, end_year = year_range_bounds(row.get("year_range"))
        launch_year = get_model_launch_year(row.get("make"), row.get("model"))

        if start_year is None or end_year is None:
            rows_to_keep.append(True)
            display_ranges.append(str(row.get("year_range", "")))
            continue

        valid_start = max(start_year, launch_year) if launch_year else start_year
        valid_end = end_year

        if valid_start > valid_end:
            rows_to_keep.append(False)
            display_ranges.append(str(row.get("year_range", "")))
        else:
            rows_to_keep.append(True)
            if valid_start == start_year:
                display_ranges.append(str(row.get("year_range", "")))
            else:
                display_ranges.append(f"{valid_start}-{valid_end}")

    cleaned["display_year_range"] = display_ranges
    cleaned = cleaned.loc[rows_to_keep].reset_index(drop=True)

    return cleaned


df = clean_vehicle_dataset(load_data())

CURRENT_YEAR = datetime.now().year
FUEL_PRICE_PER_LITRE = 2.00
ELECTRICITY_COST_PER_KWH = 0.35
EV_KWH_PER_100KM_ASSUMPTION = 16
REGO_ANNUAL = 900
INSURANCE_ANNUAL_BASE = 1200


def money(x):
    return f"${x:,.0f}"


def minmax_score(series, higher_is_better=True):
    s = series.astype(float)

    if s.max() == s.min():
        return pd.Series([7.0] * len(s), index=s.index)

    scaled = 1 + 9 * (s - s.min()) / (s.max() - s.min())

    if not higher_is_better:
        scaled = 10 - (scaled - 1)

    return scaled


def parse_year_midpoint(year_range):
    nums = [int(x) for x in re.findall(r"\d{4}", str(year_range))]

    if len(nums) >= 2:
        return int(round((nums[0] + nums[1]) / 2))

    if len(nums) == 1:
        return nums[0]

    return CURRENT_YEAR - 5


def year_to_range(year):
    if year is None:
        return None

    if 2012 <= year <= 2015:
        return "2012-2015"

    if 2016 <= year <= 2019:
        return "2016-2019"

    if 2020 <= year <= 2024:
        return "2020-2024"

    if year >= 2025:
        return "2020-2024"

    return None


def clean_text_for_matching(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_price(text):
    cleaned = str(text)

    price_patterns = [
        r"\$\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{4,6})",
        r"\b([0-9]{1,3}(?:,[0-9]{3})+)\s*(?:aud|dollars|ono|negotiable)?\b",
        r"\b([1-9][0-9]{3,5})\s*(?:aud|dollars|ono|negotiable)\b",
    ]

    candidates = []

    for pattern in price_patterns:
        for match in re.findall(pattern, cleaned, flags=re.IGNORECASE):
            value = int(str(match).replace(",", ""))
            if 1000 <= value <= 250000:
                candidates.append(value)

    if candidates:
        return candidates[0]

    return None


def extract_kilometres(text):
    cleaned = str(text).lower()

    km_patterns = [
        r"([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{4,6})\s*(?:km|kms|kilometres|kilometers)\b",
        r"(?:odometer|odo|mileage)\s*[:\-]?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{4,6})",
    ]

    candidates = []

    for pattern in km_patterns:
        for match in re.findall(pattern, cleaned, flags=re.IGNORECASE):
            value = int(str(match).replace(",", ""))
            if 0 <= value <= 500000:
                candidates.append(value)

    if candidates:
        return candidates[0]

    return None


def extract_year(text):
    years = [int(x) for x in re.findall(r"\b(19[8-9][0-9]|20[0-3][0-9])\b", str(text))]

    valid_years = [year for year in years if 1990 <= year <= CURRENT_YEAR + 1]

    if valid_years:
        return valid_years[0]

    return None


def extract_service_history(text):
    cleaned = clean_text_for_matching(text)

    if any(term in cleaned for term in ["full service history", "full logbook", "full log book", "fsh"]):
        return "Full"

    if any(term in cleaned for term in ["partial service", "some service", "logbook missing"]):
        return "Partial"

    if any(term in cleaned for term in ["no service history", "no logbook", "missing service"]):
        return "No / missing"

    return None


def extract_accident_status(text):
    cleaned = clean_text_for_matching(text)

    if any(term in cleaned for term in ["write off", "written off", "repairable writeoff", "repairable write off", "accident history"]):
        return "Known accident/write-off concern"

    if any(term in cleaned for term in ["no accident", "never accident", "no write off", "not written off", "clean title"]):
        return "No concern reported"

    return None


def extract_make_model(text, base_df):
    cleaned = clean_text_for_matching(text)

    # Prefer longer model names first so "Model 3" beats generic "3", etc.
    candidates = (
        base_df[["make", "model"]]
        .drop_duplicates()
        .assign(model_len=lambda x: x["model"].str.len())
        .sort_values("model_len", ascending=False)
    )

    # Only return make + model if BOTH are clearly found in the listing text.
    for _, row in candidates.iterrows():
        make = str(row["make"])
        model = str(row["model"])

        make_clean = clean_text_for_matching(make)
        model_clean = clean_text_for_matching(model)

        if make_clean in cleaned and model_clean in cleaned:
            return make, model

    # Make-only fallback: return make, but do NOT guess a model.
    for make in sorted(base_df["make"].unique(), key=len, reverse=True):
        make_clean = clean_text_for_matching(make)

        if make_clean in cleaned:
            return make, None

    return None, None


def extract_listing_details(raw_text, base_df):
    make, model = extract_make_model(raw_text, base_df)
    year = extract_year(raw_text)
    price = extract_price(raw_text)
    kilometres = extract_kilometres(raw_text)
    service_history = extract_service_history(raw_text)
    accident_status = extract_accident_status(raw_text)
    year_range = year_to_range(year)

    return {
        "make": make,
        "model": model,
        "year": year,
        "year_range": year_range,
        "price": price,
        "kilometres": kilometres,
        "service_history": service_history,
        "accident_status": accident_status,
    }



def years_from_year_range(year_range):
    nums = [int(x) for x in re.findall(r"\d{4}", str(year_range))]

    if len(nums) >= 2:
        start_year, end_year = nums[0], nums[1]
        if start_year <= end_year and (end_year - start_year) <= 10:
            return list(range(start_year, end_year + 1))

    if len(nums) == 1:
        return [nums[0]]

    return []


def years_from_year_range(year_range):
    nums = [int(x) for x in re.findall(r"\d{4}", str(year_range))]

    if len(nums) >= 2:
        start_year, end_year = nums[0], nums[1]
        if start_year <= end_year and (end_year - start_year) <= 10:
            return list(range(start_year, end_year + 1))

    if len(nums) == 1:
        return [nums[0]]

    return []


def build_search_query(make, model, year_range=None, max_price=None, state=None, city=None, selected_year=None):
    parts = []

    if selected_year and str(selected_year) != "Any":
        parts.append(str(selected_year))
    elif year_range and "Select" not in str(year_range):
        years = years_from_year_range(year_range)
        if years:
            parts.append(f"{years[0]}-{years[-1]}")

    parts.extend([str(make), str(model)])

    if max_price:
        parts.append(f"under {int(max_price)}")

    if city:
        parts.append(str(city))
    elif state:
        parts.append(str(state))

    return " ".join([part for part in parts if part and str(part).strip()])


def canonical_model_for_marketplace(model):
    # Our dataset sometimes has recommendation names that do not match marketplace URL slugs.
    # Example: our dataset says "Mazda3", but Carsales uses /mazda/3/.
    model = str(model).strip()

    model_overrides = {
        "Mazda3": "3",
        "MG3": "mg3",
        "Model 3": "model-3",
        "Model Y": "model-y",
        "C-Class": "c-class",
        "3 Series": "3-series",
        "CX-3": "cx-3",
        "CX-5": "cx-5",
        "CX-8": "cx-8",
        "CX-9": "cx-9",
        "CX-30": "cx-30",
        "Yaris Cross": "yaris-cross",
        "Pajero Sport": "pajero-sport",
    }

    if model in model_overrides:
        return model_overrides[model]

    suffixes_to_remove = [
        " hybrid",
        " electric",
        " petrol",
        " diesel",
        " phev",
        " petrol hybrid",
        " petrol diesel",
        " diesel petrol",
    ]

    model_lower = model.lower()

    for suffix in suffixes_to_remove:
        if model_lower.endswith(suffix):
            return model[: -len(suffix)].strip()

    return model


def slugify_for_url(value):
    value = canonical_model_for_marketplace(value)
    value = str(value).lower().strip()
    value = value.replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value


def state_to_default_city(state):
    mapping = {
        "NSW": "Sydney",
        "VIC": "Melbourne",
        "QLD": "Brisbane",
        "WA": "Perth",
        "SA": "Adelaide",
        "TAS": "Hobart",
        "ACT": "Canberra",
        "NT": "Darwin",
    }

    return mapping.get(str(state).upper(), "Sydney")


def city_to_state(city):
    mapping = {
        "Sydney": "NSW",
        "Newcastle": "NSW",
        "Melbourne": "VIC",
        "Brisbane": "QLD",
        "Gold Coast": "QLD",
        "Perth": "WA",
        "Adelaide": "SA",
        "Canberra": "ACT",
        "Hobart": "TAS",
        "Darwin": "NT",
    }

    return mapping.get(str(city), None)


def state_to_carsales_slug(state):
    mapping = {
        "NSW": "new-south-wales-state",
        "VIC": "victoria-state",
        "QLD": "queensland-state",
        "WA": "western-australia-state",
        "SA": "south-australia-state",
        "TAS": "tasmania-state",
        "ACT": "australian-capital-territory-state",
        "NT": "northern-territory-state",
    }

    return mapping.get(str(state).upper(), None)


def city_to_slug(city):
    if not city:
        return "sydney"

    return slugify_for_url(city)


def build_listing_links(make, model, year_range=None, max_price=None, state=None, city=None, selected_year=None):
    query = build_search_query(make, model, year_range, max_price, state, city, selected_year)
    encoded_query = quote_plus(query)

    make_slug = slugify_for_url(make)
    model_slug = slugify_for_url(model)
    city_slug = city_to_slug(city or state_to_default_city(state))

    inferred_state = state or city_to_state(city)
    state_slug = state_to_carsales_slug(inferred_state)

    # Carsales supports year/make/model URL paths.
    if selected_year and str(selected_year) != "Any":
        if state_slug:
            carsales_url = f"https://www.carsales.com.au/cars/{selected_year}/{make_slug}/{model_slug}/{state_slug}/"
        else:
            carsales_url = f"https://www.carsales.com.au/cars/{selected_year}/{make_slug}/{model_slug}/"
    else:
        if state_slug:
            carsales_url = f"https://www.carsales.com.au/cars/{make_slug}/{model_slug}/{state_slug}/"
        else:
            carsales_url = f"https://www.carsales.com.au/cars/{make_slug}/{model_slug}/"

    # Gumtree and Facebook use keyword search, so the selected year is included in the query.
    gumtree_url = f"https://www.gumtree.com.au/s-cars-vans-utes/k0c18320?keywords={encoded_query}"
    facebook_url = f"https://www.facebook.com/marketplace/search/?query={encoded_query}"

    # Cars24 does not have a consistently portable year URL. Use a keyword search-style URL with year in the query.
    # If Cars24 ignores the query, the fallback is still its inventory search page. Humanity trembles.
    cars24_url = f"https://www.cars24.com.au/buy-used-cars/?search={encoded_query}"

    return {
        "Carsales": carsales_url,
        "Gumtree": gumtree_url,
        "Facebook Marketplace": facebook_url,
        "Cars24": cars24_url,
    }


def render_listing_search_buttons(make, model, year_range=None, max_price=None, state=None, city=None, key_suffix=None):
    years = years_from_year_range(year_range)
    launch_year = get_model_launch_year(make, model)

    if launch_year:
        years = [year for year in years if int(year) >= int(launch_year)]

    if not key_suffix:
        key_suffix = f"{make}_{model}_{year_range}_{max_price}_{state}_{city}"

    safe_key = re.sub(r"[^A-Za-z0-9_]+", "_", str(key_suffix))

    if years:
        st.caption("Choose one or more years to search")

        all_years = st.checkbox(
            "All years in this range",
            value=True,
            key=f"search_all_years_{safe_key}",
        )

        selected_years = []

        if all_years:
            selected_years = years
        else:
            year_cols = st.columns(min(len(years), 4))

            for idx, year in enumerate(years):
                with year_cols[idx % len(year_cols)]:
                    checked = st.checkbox(
                        str(year),
                        value=False,
                        key=f"search_year_checkbox_{safe_key}_{year}",
                    )

                    if checked:
                        selected_years.append(year)

        if not selected_years:
            st.warning("Select at least one year to generate marketplace search buttons.")
            return

        # If one year is selected, show one clean row.
        # If multiple years are selected, show year-specific rows.
        for selected_year in selected_years:
            if len(selected_years) > 1:
                st.markdown(f"**{selected_year}**")

            links = build_listing_links(
                make,
                model,
                year_range,
                max_price=max_price,
                state=state,
                city=city,
                selected_year=selected_year,
            )

            cols = st.columns(len(links))

            for col, (platform, url) in zip(cols, links.items()):
                with col:
                    st.link_button(
                        f"Search {platform}",
                        url,
                        use_container_width=True,
                    )
    else:
        links = build_listing_links(
            make,
            model,
            year_range,
            max_price=max_price,
            state=state,
            city=city,
            selected_year=None,
        )

        cols = st.columns(len(links))

        for col, (platform, url) in zip(cols, links.items()):
            with col:
                st.link_button(f"Search {platform}", url, use_container_width=True)



def score_dataset(
    base_df,
    budget,
    annual_km,
    family_size,
    ownership_years,
    fuel_pref,
    body_pref,
    age_pref,
    reliability_w,
    running_cost_w,
    safety_w,
    space_w,
    comfort_w,
    tech_w,
    performance_w,
    resale_w,
):
    work = base_df.copy()

    # Age preference as a hard filter.
    if age_pref == "2020-2024 only":
        work = work[work["year_range"] == "2020-2024"].copy()
    elif age_pref == "2016-2024":
        work = work[work["year_range"].isin(["2016-2019", "2020-2024"])].copy()

    if work.empty:
        return work, "No vehicles match the selected age preference."

    # Price fit score.
    work["price_ratio"] = work["avg_used_price_aud"] / budget
    work["budget_fit_score"] = np.where(
        work["price_ratio"] <= 0.85,
        10,
        np.where(
            work["price_ratio"] <= 1.0,
            9 - (work["price_ratio"] - 0.85) * 10,
            np.maximum(1, 7 - (work["price_ratio"] - 1.0) * 15),
        ),
    ).clip(1, 10)

    # Running cost score.
    work["fuel_cost_annual_aud"] = np.where(
        work["fuel_type"].str.contains("Electric", case=False, na=False),
        annual_km / 100 * EV_KWH_PER_100KM_ASSUMPTION * ELECTRICITY_COST_PER_KWH,
        annual_km
        / 100
        * work["fuel_economy_l_per_100km"].replace(0, np.nan).fillna(6.5)
        * FUEL_PRICE_PER_LITRE,
    )

    work["fuel_cost_score"] = minmax_score(work["fuel_cost_annual_aud"], higher_is_better=False)
    work["running_cost_score"] = (work["fuel_cost_score"] * 0.6) + (work["maintenance_score"] * 0.4)

    # Family / space scoring.
    work["family_fit_score"] = work["space_score"]

    if family_size == "1-2":
        work["family_fit_score"] = (
            work["space_score"] * 0.5
            + work["budget_fit_score"] * 0.3
            + work["running_cost_score"] * 0.2
        )
    elif family_size == "3-4":
        work["family_fit_score"] = work["space_score"]
    elif family_size == "5+":
        work["family_fit_score"] = np.where(work["seats"] >= 5, work["space_score"], 3)
    elif family_size == "Need 7 seats":
        work["family_fit_score"] = np.where(work["seats"] >= 7, 10, 2)

    # Fuel preference score.
    work["fuel_preference_score"] = 10

    if fuel_pref != "Any":
        work["fuel_preference_score"] = np.where(
            work["fuel_type"].str.contains(fuel_pref, case=False, na=False),
            10,
            4,
        )

    # Body preference score.
    work["body_preference_score"] = 10

    if body_pref == "Hatch/Sedan":
        work["body_preference_score"] = np.where(
            work["body_type"].str.contains("Hatch|Sedan", case=False, na=False),
            10,
            5,
        )
    elif body_pref == "SUV":
        work["body_preference_score"] = np.where(
            work["body_type"].str.contains("SUV", case=False, na=False),
            10,
            5,
        )
    elif body_pref == "Ute":
        work["body_preference_score"] = np.where(
            work["body_type"].str.contains("Ute", case=False, na=False),
            10,
            4,
        )
    elif body_pref == "People mover":
        work["body_preference_score"] = np.where(
            work["body_type"].str.contains("People mover", case=False, na=False),
            10,
            4,
        )
    elif body_pref == "7-seat SUV":
        work["body_preference_score"] = np.where(
            (work["body_type"].str.contains("SUV", case=False, na=False))
            & (work["seats"] >= 7),
            10,
            3,
        )

    weights = {
        "budget_fit_score": 4,
        "reliability_score": reliability_w,
        "running_cost_score": running_cost_w,
        "safety_score": safety_w,
        "family_fit_score": space_w,
        "comfort_score": comfort_w,
        "technology_score": tech_w,
        "performance_score": performance_w,
        "resale_score": resale_w,
        "fuel_preference_score": 2,
        "body_preference_score": 2,
    }

    weight_sum = sum(weights.values())
    work["final_score"] = sum(work[col] * w for col, w in weights.items()) / weight_sum * 10

    # Ownership cost model.
    work["estimated_resale_value_aud"] = work["avg_used_price_aud"] * (
        0.30 + work["resale_score"] / 10 * 0.40
    )

    work["insurance_annual_aud"] = (
        INSURANCE_ANNUAL_BASE
        + np.where(work["avg_used_price_aud"] > 50000, 500, 0)
        + np.where(work["performance_score"] > 8, 300, 0)
        + np.where(work["year_range"] == "2012-2015", -150, 0)
    ).clip(800, 2500)

    work["running_cost_period_aud"] = (
        work["fuel_cost_annual_aud"]
        + work["annual_maintenance_aud"]
        + REGO_ANNUAL
        + work["insurance_annual_aud"]
    ) * ownership_years

    work["depreciation_cost_aud"] = (
        work["avg_used_price_aud"] - work["estimated_resale_value_aud"]
    )

    work["net_ownership_cost_aud"] = (
        work["running_cost_period_aud"] + work["depreciation_cost_aud"]
    )

    # Hard requirement filters.
    candidate_work = work.copy()

    if family_size == "Need 7 seats":
        candidate_work = candidate_work[candidate_work["seats"] >= 7].copy()

    if body_pref == "7-seat SUV":
        candidate_work = candidate_work[
            (candidate_work["seats"] >= 7)
            & (candidate_work["body_type"].str.contains("SUV", case=False, na=False))
        ].copy()
    elif body_pref == "People mover":
        candidate_work = candidate_work[
            candidate_work["body_type"].str.contains("People mover", case=False, na=False)
        ].copy()
    elif body_pref == "Ute":
        candidate_work = candidate_work[
            candidate_work["body_type"].str.contains("Ute", case=False, na=False)
        ].copy()

    if fuel_pref != "Any":
        candidate_work = candidate_work[
            candidate_work["fuel_type"].str.contains(fuel_pref, case=False, na=False)
        ].copy()

    if candidate_work.empty:
        return candidate_work, "No vehicles in the current dataset match your hard requirements."

    # Hard budget filter.
    affordable_work = candidate_work[candidate_work["avg_used_price_aud"] <= budget].copy()

    if affordable_work.empty:
        candidate_work["over_budget_amount"] = candidate_work["avg_used_price_aud"] - budget

        result = candidate_work.sort_values(
            ["over_budget_amount", "final_score"],
            ascending=[True, False],
        )

        return (
            result.reset_index(drop=True),
            "No vehicles match your requirements within this budget. Showing the closest matching vehicles above budget.",
        )

    result = affordable_work.sort_values("final_score", ascending=False)
    return result.reset_index(drop=True), None


def explain_recommendation(row, budget):
    reasons = []

    if row["avg_used_price_aud"] > budget:
        reasons.append("closest match above budget")

    if row["reliability_score"] >= 8.5:
        reasons.append("strong reliability")

    if row["running_cost_score"] >= 8:
        reasons.append("low running costs")

    if row["resale_score"] >= 8.5:
        reasons.append("strong resale value")

    if row["safety_score"] >= 9:
        reasons.append("high safety score")

    if row["family_fit_score"] >= 8:
        reasons.append("good practicality for your profile")

    if row["technology_score"] <= 6.2 and row["year_range"] == "2012-2015":
        reasons.append("older tech, so inspect features carefully")

    if not reasons:
        reasons.append("balanced overall fit")

    return "Recommended for " + ", ".join(reasons) + "."


def analyse_listing(
    base_df,
    make,
    model,
    year_range,
    asking_price,
    kilometres,
    seller_type,
    service_history,
    accident_concern,
    state,
):
    match = base_df[
        (base_df["make"] == make)
        & (base_df["model"] == model)
        & (base_df["year_range"] == year_range)
    ].copy()

    if match.empty:
        return None

    car = match.iloc[0].copy()

    premium_brands = [
        "Audi",
        "BMW",
        "Mercedes-Benz",
        "Volkswagen",
        "Peugeot",
        "Jeep",
        "Volvo",
        "Skoda",
    ]

    is_premium_brand = make in premium_brands
    is_older_car = year_range in ["2012-2015", "2016-2019"]

    midpoint_year = parse_year_midpoint(year_range)
    estimated_age = max(1, CURRENT_YEAR - midpoint_year)
    expected_km = estimated_age * 15000
    km_difference = kilometres - expected_km

    # Simple km adjustment:
    # If actual km is higher than expected, fair value goes down.
    # If actual km is lower than expected, fair value goes up.
    km_adjustment = (km_difference / 10000) * 600
    km_adjustment = float(np.clip(km_adjustment, -3500, 4500))

    base_market_price = float(car["avg_used_price_aud"])
    adjusted_fair_price = base_market_price - km_adjustment

    # Service history adjustment.
    if service_history == "No / missing":
        adjusted_fair_price -= 1200
    elif service_history == "Partial":
        adjusted_fair_price -= 600

    # Accident/write-off adjustment.
    if accident_concern == "Known accident/write-off concern":
        adjusted_fair_price -= 2500

    # Dealer asking prices are often slightly higher because of warranty/overheads.
    if seller_type == "Dealer":
        adjusted_fair_price += 700

    # Older premium cars need a maintenance-risk discount.
    if is_premium_brand and is_older_car:
        adjusted_fair_price -= 1000

    adjusted_fair_price = max(1000, adjusted_fair_price)

    low_fair = adjusted_fair_price * 0.94
    high_fair = adjusted_fair_price * 1.06

    discount_percent = (adjusted_fair_price - asking_price) / adjusted_fair_price
    premium_percent = (asking_price - adjusted_fair_price) / adjusted_fair_price

    # Price status logic.
    if discount_percent >= 0.15:
        price_status = "Significantly below fair range - verify carefully"
    elif asking_price < low_fair:
        price_status = "Below estimated fair range"
    elif asking_price <= high_fair:
        price_status = "Within estimated fair range"
    elif premium_percent >= 0.15:
        price_status = "Significantly above fair range"
    else:
        price_status = "Above estimated fair range"

    over_under = asking_price - adjusted_fair_price

    # Offer logic.
    if price_status == "Significantly below fair range - verify carefully":
        suggested_offer = asking_price * 0.98
    elif price_status == "Below estimated fair range":
        suggested_offer = asking_price * 0.98
    elif price_status == "Within estimated fair range":
        suggested_offer = min(asking_price * 0.96, adjusted_fair_price * 0.97)
    else:
        suggested_offer = min(asking_price * 0.92, adjusted_fair_price * 0.97)

    suggested_offer = max(1000, suggested_offer)

    green_flags = []
    red_flags = []

    # Price flags.
    if price_status == "Significantly below fair range - verify carefully":
        red_flags.append(
            "The asking price is significantly below the prototype fair-price estimate. "
            "This could be a bargain, but it could also indicate hidden mechanical, finance, accident, or write-off issues."
        )
    elif asking_price < low_fair:
        green_flags.append(
            "Asking price is below the prototype fair-price estimate, but still verify condition carefully."
        )
    elif asking_price <= high_fair:
        green_flags.append("Asking price is within the prototype fair-price range.")
    else:
        red_flags.append(
            "Asking price appears high compared with the prototype fair-price estimate."
        )

    # Kilometres flags.
    if kilometres > expected_km * 1.35:
        red_flags.append("Kilometres are very high for the estimated age.")
    elif kilometres > expected_km * 1.15:
        red_flags.append("Kilometres are slightly high for the estimated age.")
    elif kilometres < expected_km * 0.65:
        green_flags.append("Kilometres are low for the estimated age.")
        red_flags.append(
            "Very low kilometres can be positive, but verify service records and usage history. "
            "Low km alone does not guarantee good condition."
        )
    else:
        green_flags.append("Kilometres are around the expected range.")

    # Service history flags.
    if service_history == "Full":
        green_flags.append("Full service history reported.")
    elif service_history == "Partial":
        red_flags.append("Only partial service history reported.")
    else:
        red_flags.append("Missing service history is a meaningful risk.")

    # Accident/write-off flags.
    if accident_concern == "Known accident/write-off concern":
        red_flags.append("Known accident/write-off concern needs serious verification.")
    elif accident_concern == "Unknown":
        red_flags.append(
            "Accident/write-off status is unknown. Run a PPSR check before considering purchase."
        )
    else:
        green_flags.append("No accident/write-off concern reported.")

    # Brand / maintenance logic.
    if is_premium_brand and is_older_car:
        red_flags.append(
            f"{make} is a premium brand and this is an older vehicle. "
            "Maintenance and repair costs may be significantly higher than mainstream brands."
        )

    if car["annual_maintenance_aud"] >= 1400:
        red_flags.append(
            "Prototype maintenance estimate is high. Budget carefully for repairs and servicing."
        )

    if car["maintenance_score"] <= 6.3:
        red_flags.append(
            "Maintenance score is weak in the prototype dataset. Independent inspection is strongly recommended."
        )

    # Reliability flags.
    if car["reliability_score"] >= 8.5:
        green_flags.append("Model has a strong prototype reliability score.")
    elif car["reliability_score"] < 7:
        red_flags.append("Model has a weaker prototype reliability score, so inspect carefully.")

    # Risk scoring.
    risk_points = 0

    if price_status == "Significantly below fair range - verify carefully":
        risk_points += 2
    elif price_status == "Above estimated fair range":
        risk_points += 2
    elif price_status == "Significantly above fair range":
        risk_points += 3

    if kilometres > expected_km * 1.35:
        risk_points += 3
    elif kilometres > expected_km * 1.15:
        risk_points += 1

    if service_history == "Partial":
        risk_points += 1
    elif service_history == "No / missing":
        risk_points += 3

    if accident_concern == "Unknown":
        risk_points += 1
    elif accident_concern == "Known accident/write-off concern":
        risk_points += 4

    if is_premium_brand and is_older_car:
        risk_points += 2

    if car["maintenance_score"] <= 6.3:
        risk_points += 1

    if car["reliability_score"] < 7:
        risk_points += 1

    # Prevent suspiciously cheap older premium cars from being labelled low risk.
    if (
        price_status == "Significantly below fair range - verify carefully"
        and is_premium_brand
        and is_older_car
    ):
        risk_points = max(risk_points, 4)

    if risk_points <= 2:
        risk_level = "Low"
    elif risk_points <= 5:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Categorised warnings for cleaner output.
    critical_checks = []
    cost_warnings = []
    context_warnings = []

    # Critical checks.
    if accident_concern == "Known accident/write-off concern":
        critical_checks.append(
            "Known accident/write-off concern. Verify carefully before proceeding."
        )
    elif accident_concern == "Unknown":
        critical_checks.append(
            "Accident/write-off status is unknown. Run a PPSR check before considering purchase."
        )

    if service_history == "No / missing":
        critical_checks.append(
            "Missing service history. This is a major inspection and verification risk."
        )
    elif service_history == "Partial":
        critical_checks.append(
            "Only partial service history reported. Ask for invoices and servicing proof."
        )

    if asking_price > high_fair:
        critical_checks.append(
            "Asking price appears high compared with the prototype fair-price estimate."
        )

    if price_status == "Significantly below fair range - verify carefully":
        critical_checks.append(
            "Price is significantly below estimated fair value. Verify PPSR, accident history, finance owing, and mechanical condition."
        )

    # Cost warnings.
    if is_premium_brand and is_older_car:
        cost_warnings.append(
            f"{make} is a premium brand and this is an older vehicle. "
            "Maintenance and repair costs may be significantly higher than mainstream brands."
        )

    if car["annual_maintenance_aud"] >= 1400:
        cost_warnings.append(
            "Prototype maintenance estimate is high. Budget carefully for repairs and servicing."
        )

    if car["maintenance_score"] <= 6.3:
        cost_warnings.append(
            "Maintenance score is weak in the prototype dataset. Independent inspection is strongly recommended."
        )

    if car["reliability_score"] < 7:
        cost_warnings.append(
            "Model has a weaker prototype reliability score, so inspect carefully."
        )

    # Context warnings.
    if kilometres < expected_km * 0.65:
        context_warnings.append(
            "Kilometres are very low for the estimated age. That can be positive, but verify service records and usage history."
        )

    if kilometres > expected_km * 1.35:
        context_warnings.append("Kilometres are very high for the estimated age.")
    elif kilometres > expected_km * 1.15:
        context_warnings.append("Kilometres are slightly high for the estimated age.")

    # Final verdict logic.
    if risk_level == "Low":
        final_verdict = "Good candidate"
        verdict_reason = (
            "The listing looks reasonable based on the prototype estimate. "
            "Still complete PPSR, service history checks, and inspection before purchase."
        )
    elif risk_level == "Medium":
        final_verdict = "Proceed with caution"
        verdict_reason = (
            "The listing has attractive elements, but there are enough risks that you should verify documents, "
            "run a PPSR check, and arrange an independent inspection before negotiating."
        )
    else:
        final_verdict = "High risk - inspect carefully or avoid"
        verdict_reason = (
            "The listing has multiple risk factors. Do not proceed without strong evidence, PPSR verification, "
            "full service records, and an independent mechanical inspection."
        )

    questions = [
        "Can you provide the full service history and invoices?",
        "Has the car ever been in an accident or written off?",
        "Can I run a PPSR check using the VIN?",
        "Is there any finance owing on the car?",
        "Are there any warning lights, oil leaks, transmission issues, or recent repairs?",
        "When were the tyres, brakes, and battery last replaced?",
        "Has the timing belt/chain, transmission, or major service item been inspected recently?",
        "Can I arrange an independent pre-purchase inspection?",
    ]

    if is_premium_brand and is_older_car:
        questions.extend(
            [
                "Has the car been serviced by a specialist or dealer?",
                "Are there records for major repairs such as transmission, suspension, turbo, cooling system, or electronics?",
                "Are there any oil leaks, coolant leaks, or electrical faults?",
            ]
        )

    result = {
        "car": car,
        "expected_km": expected_km,
        "km_difference": km_difference,
        "base_market_price": base_market_price,
        "adjusted_fair_price": adjusted_fair_price,
        "low_fair": low_fair,
        "high_fair": high_fair,
        "price_status": price_status,
        "over_under": over_under,
        "suggested_offer": suggested_offer,
        "risk_level": risk_level,
        "green_flags": green_flags,
        "red_flags": red_flags,
        "critical_checks": critical_checks,
        "cost_warnings": cost_warnings,
        "context_warnings": context_warnings,
        "final_verdict": final_verdict,
        "verdict_reason": verdict_reason,
        "questions": questions,
        "state": state,
    }

    return result


PPSR_OFFICIAL_URL = "https://www.ppsr.gov.au/"


def validate_vin(vin):
    vin = str(vin).strip().upper().replace(" ", "")

    if not vin:
        return "Not provided", None

    # VINs are normally 17 characters and exclude I, O, Q.
    if len(vin) != 17:
        return "Check VIN length", "VIN should normally be 17 characters."

    if re.search(r"[IOQ]", vin):
        return "Check VIN characters", "VIN should not normally contain I, O, or Q."

    if not re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", vin):
        return "Check VIN format", "VIN contains unexpected characters."

    return "Looks valid", None


def apply_vehicle_history_checks(
    analysis,
    vin,
    rego,
    ppsr_done,
    finance_status,
    written_off_status,
    stolen_status,
    takata_status,
):
    analysis = analysis.copy()
    analysis["green_flags"] = list(analysis.get("green_flags", []))
    analysis["red_flags"] = list(analysis.get("red_flags", []))
    analysis["critical_checks"] = list(analysis.get("critical_checks", []))
    analysis["cost_warnings"] = list(analysis.get("cost_warnings", []))
    analysis["context_warnings"] = list(analysis.get("context_warnings", []))
    analysis["questions"] = list(analysis.get("questions", []))

    vin_status, vin_warning = validate_vin(vin)

    vehicle_history = {
        "vin": str(vin).strip().upper(),
        "rego": str(rego).strip().upper(),
        "vin_status": vin_status,
        "vin_warning": vin_warning,
        "ppsr_done": ppsr_done,
        "finance_status": finance_status,
        "written_off_status": written_off_status,
        "stolen_status": stolen_status,
        "takata_status": takata_status,
        "ppsr_url": PPSR_OFFICIAL_URL,
    }

    analysis["vehicle_history"] = vehicle_history

    risk_floor = analysis.get("risk_level", "Low")
    no_buy = False

    if vin_warning:
        analysis["context_warnings"].append(vin_warning)

    if not vehicle_history["vin"]:
        analysis["critical_checks"].append(
            "VIN not entered. Ask the seller for the VIN and run the official PPSR check before making any payment."
        )
        analysis["questions"].append("Can you send the VIN so I can run an official PPSR check before inspection/payment?")

    if ppsr_done == "No - not checked yet":
        analysis["critical_checks"].append(
            "PPSR has not been checked yet. This is mandatory before deposit or payment because it can reveal finance owing, stolen status, and write-off records."
        )
        analysis["questions"].append("Are you comfortable with me completing a PPSR check before any deposit or payment?")
        risk_floor = "Medium"

    elif ppsr_done == "Yes - checked":
        if finance_status == "No":
            analysis["green_flags"].append("PPSR finance/security-interest status reported clear.")
        elif finance_status == "Yes":
            analysis["critical_checks"].append(
                "PPSR indicates a security interest / finance owing. Do not pay the seller until the finance is discharged and written proof is provided."
            )
            analysis["questions"].append("Can you provide written payout/discharge confirmation for the finance owing before sale?")
            risk_floor = "High"
            no_buy = True
        else:
            analysis["critical_checks"].append(
                "PPSR finance/security-interest result is unclear. Treat as unresolved until the certificate is checked."
            )
            risk_floor = "Medium"

        if written_off_status == "No":
            analysis["green_flags"].append("PPSR written-off status reported clear.")
        elif written_off_status == "Yes":
            analysis["critical_checks"].append(
                "PPSR indicates the vehicle has a written-off record. Avoid unless you fully understand the category, repair evidence, insurance impact, and resale consequences."
            )
            analysis["questions"].append("Can you provide written-off category details, repair invoices, and engineering/inspection evidence?")
            risk_floor = "High"
            no_buy = True
        else:
            analysis["critical_checks"].append(
                "PPSR written-off result is unclear. Verify the certificate before proceeding."
            )
            risk_floor = "Medium"

        if stolen_status == "No":
            analysis["green_flags"].append("PPSR stolen status reported clear.")
        elif stolen_status == "Yes":
            analysis["critical_checks"].append(
                "PPSR indicates the vehicle is recorded as stolen. Do not proceed with the purchase."
            )
            analysis["questions"].append("Why is the vehicle showing as stolen on PPSR?")
            risk_floor = "High"
            no_buy = True
        else:
            analysis["critical_checks"].append(
                "PPSR stolen-status result is unclear. Do not proceed until this is resolved."
            )
            risk_floor = "Medium"

        if takata_status == "No":
            analysis["green_flags"].append("No Takata recall issue reported from the PPSR/vehicle-history check.")
        elif takata_status == "Yes":
            analysis["critical_checks"].append(
                "Takata airbag recall flagged. Confirm whether the recall repair has been completed before considering purchase."
            )
            analysis["questions"].append("Has the Takata airbag recall repair been completed? Can you provide proof?")
            risk_floor = "Medium"
        else:
            analysis["context_warnings"].append(
                "Takata recall status is unclear. Check recall status before purchase."
            )

    else:
        analysis["critical_checks"].append(
            "PPSR status is unclear. Run the official PPSR search using the VIN before any deposit or payment."
        )
        risk_floor = "Medium"

    # Apply risk floor.
    current_level = analysis.get("risk_level", "Low")
    order = {"Low": 0, "Medium": 1, "High": 2}

    if order.get(risk_floor, 0) > order.get(current_level, 0):
        analysis["risk_level"] = risk_floor

    if no_buy:
        analysis["final_verdict"] = "Do not proceed until vehicle-history issue is resolved"
        analysis["verdict_reason"] = (
            "The listing has a serious PPSR/vehicle-history issue. The car may have finance owing, "
            "a written-off record, or stolen status. Resolve this with official evidence before considering purchase."
        )
    elif analysis["risk_level"] == "High" and "PPSR" in " ".join(analysis["critical_checks"]):
        analysis["final_verdict"] = "High risk - verify PPSR before proceeding"
        analysis["verdict_reason"] = (
            "There are unresolved vehicle-history checks. Do not make a deposit or payment until the official PPSR result is clear."
        )
    elif analysis["risk_level"] == "Medium" and ppsr_done != "Yes - checked":
        analysis["final_verdict"] = "Needs PPSR check before decision"
        analysis["verdict_reason"] = (
            "The listing may be reasonable, but the vehicle-history status is not verified yet. Run the PPSR check before deciding."
        )

    return analysis



def classify_expert_next_step(analysis):
    risk_level = analysis.get("risk_level", "Medium")
    final_verdict = analysis.get("final_verdict", "")

    if risk_level == "High" or "Do not proceed" in final_verdict:
        return (
            "Talk to a car buying advisor first",
            "This listing has high-risk signals. Before booking an inspection or negotiating, get human advice on whether it is even worth pursuing."
        )

    if risk_level == "Medium":
        return (
            "Book a pre-purchase inspection",
            "This listing may still be worth considering, but it needs a mechanic/inspection check before negotiation or payment."
        )

    return (
        "Get negotiation help",
        "This listing looks lower risk, so the next useful step is checking the car physically and negotiating from a stronger position."
    )


def build_partner_lead_summary(
    name,
    email,
    phone,
    suburb,
    help_needed,
    preferred_time,
    make,
    model,
    year_range,
    asking_price,
    kilometres,
    state,
    city,
    listing_url,
    analysis,
):
    vehicle_history = analysis.get("vehicle_history", {})

    lines = []
    lines.append("# AutoAdvisor Partner Lead")
    lines.append("")
    lines.append("## Buyer details")
    lines.append(f"- Name: {name or 'Not provided'}")
    lines.append(f"- Email: {email or 'Not provided'}")
    lines.append(f"- Phone: {phone or 'Not provided'}")
    lines.append(f"- Suburb: {suburb or 'Not provided'}")
    lines.append(f"- Help needed: {help_needed}")
    lines.append(f"- Preferred contact time: {preferred_time or 'Not provided'}")
    lines.append("")
    lines.append("## Vehicle / listing")
    lines.append(f"- Vehicle: {make} {model} ({year_range})")
    lines.append(f"- Asking price: {money(asking_price)}")
    lines.append(f"- Kilometres: {int(kilometres):,} km")
    lines.append(f"- Location: {city}, {state}")
    lines.append(f"- Listing URL: {listing_url or 'Not provided'}")
    lines.append("")
    lines.append("## AutoAdvisor result")
    lines.append(f"- Risk level: {analysis.get('risk_level', 'Not calculated')}")
    lines.append(f"- Final verdict: {analysis.get('final_verdict', 'Not calculated')}")
    lines.append(f"- Verdict reason: {analysis.get('verdict_reason', 'Not calculated')}")
    lines.append(f"- Suggested offer: {money(analysis.get('suggested_offer', 0))}")
    lines.append(f"- Fair range: {money(analysis.get('fair_low', 0))} to {money(analysis.get('fair_high', 0))}")
    lines.append("")
    lines.append("## PPSR / vehicle history")
    lines.append(f"- VIN: {vehicle_history.get('vin') or 'Not provided'}")
    lines.append(f"- Rego: {vehicle_history.get('rego') or 'Not provided'}")
    lines.append(f"- PPSR checked: {vehicle_history.get('ppsr_done', 'Not recorded')}")
    lines.append(f"- Finance/security interest: {vehicle_history.get('finance_status', 'Not recorded')}")
    lines.append(f"- Written off: {vehicle_history.get('written_off_status', 'Not recorded')}")
    lines.append(f"- Stolen: {vehicle_history.get('stolen_status', 'Not recorded')}")
    lines.append("")
    lines.append("## Important notes")
    lines.append("- This is an MVP referral lead summary, not legal, financial, mechanical, or insurance advice.")
    lines.append("- Partner should independently verify details with the buyer and inspect relevant documents before advising.")
    lines.append("- Buyer should not pay deposit or final payment until PPSR, inspection, ownership, and seller identity checks are complete.")
    lines.append("")

    return "\\n".join(lines)


def render_expert_help_section(
    analysis,
    make,
    model,
    year_range,
    asking_price,
    kilometres,
    state,
    city,
    listing_url,
):
    recommended_action, action_reason = classify_expert_next_step(analysis)

    st.markdown("---")
    st.subheader("Get human expert help")
    st.caption(
        "AutoAdvisor can filter the risk, but a real human expert should still inspect, verify, and negotiate before money changes hands. "
        "Annoying, yes. Cheaper than buying a lemon with Bluetooth."
    )

    st.markdown(
        f"""
        <div class="expert-card">
            <div>
                <div class="expert-eyebrow">Recommended next step</div>
                <div class="expert-title">{escape(recommended_action)}</div>
                <div class="expert-copy">{escape(action_reason)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Book pre-purchase inspection**")
        st.caption("For checking mechanical condition, accident signs, leaks, tyres, brakes, and whether the car is worth buying.")

    with c2:
        st.markdown("**Talk to car buying advisor**")
        st.caption("For choosing between listings, avoiding risky cars, and getting help with the buying process.")

    with c3:
        st.markdown("**Get negotiation help**")
        st.caption("For using the report, fair range, PPSR status, and inspection findings to negotiate better.")

    st.markdown("### Request expert help")

    default_help_index = 0
    help_options = [
        "Book pre-purchase inspection",
        "Talk to car buying advisor",
        "Get negotiation help",
        "Not sure - tell me the best next step",
    ]

    if recommended_action in help_options:
        default_help_index = help_options.index(recommended_action)

    lead_col1, lead_col2 = st.columns(2)

    with lead_col1:
        buyer_name = st.text_input("Name", key="partner_buyer_name")
        buyer_email = st.text_input("Email", key="partner_buyer_email")
        buyer_phone = st.text_input("Phone", key="partner_buyer_phone")

    with lead_col2:
        buyer_suburb = st.text_input("Suburb", value=city or "", key="partner_buyer_suburb")
        preferred_time = st.selectbox(
            "Preferred contact time",
            ["Anytime", "Morning", "Afternoon", "Evening", "Weekend"],
            key="partner_preferred_time",
        )
        help_needed = st.selectbox(
            "Help needed",
            help_options,
            index=default_help_index,
            key="partner_help_needed",
        )

    consent = st.checkbox(
        "I agree for AutoAdvisor to share this request with a trusted inspection/advisor partner when partnerships are available.",
        key="partner_consent",
    )

    lead_summary = build_partner_lead_summary(
        buyer_name,
        buyer_email,
        buyer_phone,
        buyer_suburb,
        help_needed,
        preferred_time,
        make,
        model,
        year_range,
        asking_price,
        kilometres,
        state,
        city,
        listing_url,
        analysis,
    )

    if not consent:
        st.info(
            "For now, this creates a lead summary you can download. Once partners are added, this can become a proper referral form. "
            "Because apparently businesses need consent before flinging your details across the internet. Sensible, tragically."
        )

    st.download_button(
        "Download expert-help request",
        data=lead_summary,
        file_name=f"autoadvisor_expert_help_{make}_{model}.md".replace(" ", "_").lower(),
        mime="text/markdown",
        use_container_width=True,
    )

    if consent:
        if buyer_name and (buyer_email or buyer_phone):
            st.success(
                "Expert-help request prepared. In the MVP, download this summary and send it manually to an advisor or inspection partner."
            )
        else:
            st.warning("Add your name and either email or phone before this becomes a usable partner lead.")

    with st.expander("Future partner model"):
        st.write(
            """
            In the commercial version, this section can route leads to:
            - mobile pre-purchase inspection providers
            - car buying advisors / brokers
            - negotiation support partners
            - insurance or finance partners later

            AutoAdvisor would send a qualified lead with the car, risk score, PPSR status, budget context, and buyer details.
            """
        )



def build_listing_report(
    analysis,
    make,
    model,
    year_range,
    asking_price,
    kilometres,
    seller_type,
    service_history,
    accident_concern,
    state,
):
    lines = []

    lines.append("# AutoAdvisor AI Listing Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## Vehicle")
    lines.append(f"- Vehicle: {make} {model} ({year_range})")
    lines.append(f"- Asking price: {money(asking_price)}")
    lines.append(f"- Kilometres: {kilometres:,.0f} km")
    lines.append(f"- Seller type: {seller_type}")
    lines.append(f"- State/territory: {state}")
    lines.append(f"- Service history: {service_history}")
    lines.append(f"- Accident/write-off concern: {accident_concern}")

    vehicle_history = analysis.get("vehicle_history", {})
    if vehicle_history:
        lines.append("")
        lines.append("## Vehicle history / PPSR")
        lines.append(f"- VIN: {vehicle_history.get('vin') or 'Not provided'}")
        lines.append(f"- Rego: {vehicle_history.get('rego') or 'Not provided'}")
        lines.append(f"- VIN check: {vehicle_history.get('vin_status', 'Not checked')}")
        if vehicle_history.get("vin_warning"):
            lines.append(f"- VIN warning: {vehicle_history.get('vin_warning')}")
        lines.append(f"- PPSR completed: {vehicle_history.get('ppsr_done', 'Not recorded')}")
        lines.append(f"- Finance/security interest: {vehicle_history.get('finance_status', 'Not recorded')}")
        lines.append(f"- Written-off status: {vehicle_history.get('written_off_status', 'Not recorded')}")
        lines.append(f"- Stolen status: {vehicle_history.get('stolen_status', 'Not recorded')}")
        lines.append(f"- Takata recall status: {vehicle_history.get('takata_status', 'Not recorded')}")
        lines.append(f"- Official PPSR site: {vehicle_history.get('ppsr_url', PPSR_OFFICIAL_URL)}")

    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Risk level: {analysis['risk_level']}")
    lines.append(f"- Final verdict: {analysis['final_verdict']}")
    lines.append(f"- Verdict reason: {analysis['verdict_reason']}")
    lines.append(f"- Price status: {analysis['price_status']}")
    lines.append(f"- Prototype adjusted fair price estimate: {money(analysis['adjusted_fair_price'])}")
    lines.append(f"- Estimated fair range: {money(analysis['low_fair'])} - {money(analysis['high_fair'])}")
    lines.append(f"- Suggested first offer: {money(analysis['suggested_offer'])}")
    lines.append(f"- Expected kilometres for age band: {analysis['expected_km']:,.0f} km")
    lines.append(f"- Kilometres vs expected: {analysis['km_difference']:,.0f} km")
    lines.append(f"- Asking price difference vs adjusted fair estimate: {money(analysis['over_under'])}")
    lines.append("")
    lines.append("## Green flags")

    if analysis["green_flags"]:
        for item in analysis["green_flags"]:
            lines.append(f"- {item}")
    else:
        lines.append("- No major green flags identified.")

    lines.append("")
    lines.append("## Critical checks")

    if analysis["critical_checks"]:
        for item in analysis["critical_checks"]:
            lines.append(f"- {item}")
    else:
        lines.append("- No critical checks triggered.")

    lines.append("")
    lines.append("## Cost warnings")

    if analysis["cost_warnings"]:
        for item in analysis["cost_warnings"]:
            lines.append(f"- {item}")
    else:
        lines.append("- No major cost warnings triggered.")

    lines.append("")
    lines.append("## Context warnings")

    if analysis["context_warnings"]:
        for item in analysis["context_warnings"]:
            lines.append(f"- {item}")
    else:
        lines.append("- No extra context warnings triggered.")

    lines.append("")
    lines.append("## Questions to ask seller")

    for q in analysis["questions"]:
        lines.append(f"- {q}")

    lines.append("")
    lines.append("## Negotiation script")
    lines.append(
        f"Hi, I’m interested in the {make} {model}. Based on similar prototype market estimates, "
        f"the fair range appears to be around {money(analysis['low_fair'])} to {money(analysis['high_fair'])}, "
        f"depending on condition and service history. Would you consider {money(analysis['suggested_offer'])} "
        f"subject to inspection and PPSR check?"
    )
    lines.append("")
    lines.append("## Prototype disclaimer")
    lines.append(
        "This is a prototype estimate, not financial advice or a certified valuation. "
        "Verify with PPSR, inspection, service records, and real market data before purchase."
    )

    return "\n".join(lines)


def safe_filename(text):
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower()
    return text or "report"



def vehicle_visual_theme(row):
    body = str(row.get("body_type", "")).lower()
    fuel = str(row.get("fuel_type", "")).lower()

    if "electric" in fuel:
        accent = "#41d6ff"
        glow = "rgba(65, 214, 255, 0.35)"
    elif "hybrid" in fuel:
        accent = "#6ee7b7"
        glow = "rgba(110, 231, 183, 0.30)"
    elif "diesel" in fuel:
        accent = "#fbbf24"
        glow = "rgba(251, 191, 36, 0.28)"
    else:
        accent = "#a78bfa"
        glow = "rgba(167, 139, 250, 0.28)"

    if "ute" in body:
        shape = "UTE"
    elif "suv" in body:
        shape = "SUV"
    elif "people mover" in body:
        shape = "MPV"
    elif "hatch" in body:
        shape = "HATCH"
    elif "sedan" in body:
        shape = "SEDAN"
    else:
        shape = "AUTO"

    return accent, glow, shape


VEHICLE_IMAGE_OVERRIDES = {
    # These are representative images. Exact trim/year matching would require licensed listing/photo data.
    ("Volvo", "XC40"): "https://en.wikipedia.org/wiki/Special:Redirect/file/2018_Volvo_XC40_First_Edition_T5_AWD_Automatic_2.0_Front.jpg",
    ("MG", "ZS EV"): "https://commons.wikimedia.org/wiki/Special:Redirect/file/MG_ZS_EV_IMG_4206.jpg",
    ("BYD", "Atto 3"): "https://en.wikipedia.org/wiki/Special:Redirect/file/BYD_Atto_3_IMG_7536.jpg",
    ("BYD", "Seal"): "https://en.wikipedia.org/wiki/Special:Redirect/file/BYD_Seal_IMG_9417.jpg",
    ("GWM", "Ora"): "https://en.wikipedia.org/wiki/Special:Redirect/file/Ora_Good_Cat_001.jpg",
    ("Polestar", "2"): "https://en.wikipedia.org/wiki/Special:Redirect/file/Polestar_2_IMG_3775.jpg",
}


def get_override_vehicle_image(make, model):
    key = (str(make).strip(), str(model).strip())

    if key in VEHICLE_IMAGE_OVERRIDES:
        return VEHICLE_IMAGE_OVERRIDES[key]

    normalised_key = (str(make).strip(), normalise_model_for_launch(model))

    return VEHICLE_IMAGE_OVERRIDES.get(normalised_key)


@st.cache_data(ttl=60 * 60 * 24)
def get_vehicle_image_url(make, model):
    """Fetch a representative real vehicle image from Wikimedia/Wikipedia.

    This is best-effort. It avoids scraping marketplace images and keeps the MVP deployable.
    """
    if requests is None:
        return None

    make = str(make).strip()
    model = str(model).strip()

    override_url = get_override_vehicle_image(make, model)
    if override_url:
        return override_url

    model_core = canonical_model_for_marketplace(model)
    model_core = str(model_core).replace("-", " ").strip()

    search_terms = []

    # Try exact model first, then core model, then generic automobile wording.
    for term in [
        f"{make} {model} car",
        f"{make} {model_core} car",
        f"{make} {model_core} automobile",
    ]:
        if term not in search_terms:
            search_terms.append(term)

    endpoint = "https://en.wikipedia.org/w/api.php"
    headers = {
        "User-Agent": "AutoAdvisorAI-Australia-MVP/1.0 student portfolio project"
    }

    # Exact title lookup first. This catches pages such as "Volvo XC40" better than generic search.
    for title in [f"{make} {model}", f"{make} {model_core}"]:
        try:
            exact_response = requests.get(
                endpoint,
                params={
                    "action": "query",
                    "format": "json",
                    "prop": "pageimages",
                    "piprop": "original|thumbnail",
                    "pithumbsize": 900,
                    "redirects": 1,
                    "titles": title,
                    "origin": "*",
                },
                headers=headers,
                timeout=5,
            )
            exact_response.raise_for_status()
            exact_payload = exact_response.json()
            exact_pages = exact_payload.get("query", {}).get("pages", {})

            for page in exact_pages.values():
                image_url = page.get("original", {}).get("source") or page.get("thumbnail", {}).get("source")
                if image_url:
                    return image_url
        except Exception:
            pass

    for term in search_terms:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": term,
            "gsrlimit": 6,
            "prop": "pageimages",
            "pithumbsize": 900,
            "pilicense": "any",
            "origin": "*",
        }

        try:
            response = requests.get(endpoint, params=params, headers=headers, timeout=5)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            continue

        pages = payload.get("query", {}).get("pages", {})
        ordered_pages = sorted(pages.values(), key=lambda item: item.get("index", 999))

        for page in ordered_pages:
            thumbnail = page.get("thumbnail", {})
            image_url = thumbnail.get("source")
            title = str(page.get("title", "")).lower()

            if not image_url:
                continue

            # Avoid very unrelated results when the search API gets creative. Because naturally it does.
            if make.lower() in title or model_core.lower() in title or str(model).lower() in title:
                return image_url

        # If no title match survived, use the first thumbnail from this query.
        for page in ordered_pages:
            image_url = page.get("thumbnail", {}).get("source")
            if image_url:
                return image_url

    return None


def render_vehicle_photo(row):
    make = escape(str(row["make"]))
    model = escape(str(row["model"]))
    year_range = escape(str(row.get("display_year_range", row["year_range"])))
    fuel = escape(str(row.get("fuel_type", "")))
    body = escape(str(row.get("body_type", "")))

    image_url = get_vehicle_image_url(row["make"], row["model"])

    if image_url:
        safe_url = escape(image_url, quote=True)
        return f"""
<div class="vehicle-photo-card">
    <img src="{safe_url}" alt="{make} {model}" loading="lazy" />
    <div class="vehicle-photo-overlay"></div>
    <div class="vehicle-photo-meta">
        <div class="vehicle-photo-tags">
            <span>{body}</span>
            <span>{fuel}</span>
        </div>
        <h3>{make} {model}</h3>
        <p>{year_range}</p>
    </div>
</div>
"""

    return f"""
<div class="vehicle-photo-card vehicle-photo-fallback">
    <div class="vehicle-photo-tags">
        <span>{body}</span>
        <span>{fuel}</span>
    </div>
    <div class="fallback-main">Image unavailable</div>
    <h3>{make} {model}</h3>
    <p>{year_range}</p>
</div>
"""


def render_top_match_spotlight(row, ownership_years):
    art = render_vehicle_photo(row)
    make = escape(str(row["make"]))
    model = escape(str(row["model"]))
    year_range = escape(str(row.get("display_year_range", row["year_range"])))
    buyer_fit = escape(str(row.get("buyer_fit", "Good fit for your profile")))

    st.html(
        f"""
        <div class="spotlight-card">
            <div class="spotlight-copy">
                <div class="eyebrow">Top Match</div>
                <h2>{make} {model} <span>({year_range})</span></h2>
                <p>{buyer_fit}</p>
                <div class="spotlight-metrics">
                    <div><small>Score</small><strong>{row['final_score']:.1f}/100</strong></div>
                    <div><small>Used price</small><strong>{money(row['avg_used_price_aud'])}</strong></div>
                    <div><small>{ownership_years}-yr cost</small><strong>{money(row['net_ownership_cost_aud'])}</strong></div>
                </div>
            </div>
            {art}
        </div>
        """
    )


def render_css_and_hero():
    st.html(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at 8% 5%, rgba(56, 189, 248, 0.17), transparent 30%),
                    radial-gradient(circle at 88% 12%, rgba(167, 139, 250, 0.18), transparent 28%),
                    linear-gradient(135deg, #050816 0%, #0f172a 52%, #111827 100%);
            }

            [data-testid="stHeader"] {
                background: rgba(5, 8, 22, 0.15);
                backdrop-filter: blur(10px);
            }

            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 4rem;
                max-width: 1280px;
            }

            .hero-wrap {
                position: relative;
                overflow: hidden;
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 28px;
                min-height: 310px;
                padding: 34px;
                background:
                    linear-gradient(135deg, rgba(7, 11, 20, 0.90), rgba(15, 23, 42, 0.72)),
                    radial-gradient(circle at 70% 35%, rgba(56, 189, 248, 0.18), transparent 28%),
                    radial-gradient(circle at 82% 72%, rgba(167, 139, 250, 0.22), transparent 32%);
                box-shadow: 0 30px 90px rgba(0,0,0,0.45);
                margin-bottom: 24px;
            }

            .hero-grid {
                display: grid;
                grid-template-columns: 1.05fr 0.95fr;
                gap: 24px;
                align-items: center;
            }

            .hero-badge {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                color: #cffafe;
                background: rgba(14, 165, 233, 0.14);
                border: 1px solid rgba(125, 211, 252, 0.25);
                padding: 8px 12px;
                border-radius: 999px;
                font-size: 0.85rem;
                letter-spacing: 0.2px;
                margin-bottom: 14px;
            }

            .hero-title {
                color: white;
                font-size: clamp(2.1rem, 4.8vw, 4.4rem);
                line-height: 0.98;
                font-weight: 850;
                letter-spacing: -0.055em;
                margin: 0 0 16px 0;
            }

            .hero-title span {
                background: linear-gradient(90deg, #38bdf8, #a78bfa, #6ee7b7);
                -webkit-background-clip: text;
                color: transparent;
            }

            .hero-subtitle {
                color: #cbd5e1;
                font-size: 1.05rem;
                line-height: 1.6;
                max-width: 650px;
                margin-bottom: 22px;
            }

            .hero-pills {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }

            .hero-pill {
                color: #e5e7eb;
                background: rgba(255,255,255,0.07);
                border: 1px solid rgba(255,255,255,0.13);
                border-radius: 999px;
                padding: 9px 13px;
                font-size: 0.9rem;
            }

            .hero-visual {
                position: relative;
                min-height: 245px;
                border-radius: 24px;
                background:
                    linear-gradient(160deg, rgba(255,255,255,0.10), rgba(255,255,255,0.03)),
                    radial-gradient(circle at 50% 15%, rgba(56,189,248,0.28), transparent 35%);
                border: 1px solid rgba(255,255,255,0.13);
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: inset 0 0 40px rgba(255,255,255,0.03);
            }

            .hero-visual::before {
                content: "";
                position: absolute;
                inset: 16px;
                border-radius: 20px;
                background:
                    linear-gradient(90deg, transparent 0 10%, rgba(255,255,255,0.08) 10% 10.5%, transparent 10.5% 20%),
                    linear-gradient(rgba(255,255,255,0.06) 1px, transparent 1px);
                background-size: 42px 42px, 42px 42px;
                opacity: 0.35;
            }

            .handshake-scene {
                position: relative;
                z-index: 2;
                text-align: center;
                color: white;
            }

            .handshake-line {
                font-size: 4.8rem;
                filter: drop-shadow(0 0 28px rgba(56,189,248,0.45));
                margin-bottom: 5px;
            }

            .micro-car {
                width: 260px;
                height: 84px;
                margin: 0 auto;
                position: relative;
            }

            .micro-car .body {
                position: absolute;
                bottom: 18px;
                left: 22px;
                width: 216px;
                height: 42px;
                border-radius: 18px 44px 16px 16px;
                background: linear-gradient(90deg, #38bdf8, #a78bfa);
                box-shadow: 0 18px 40px rgba(56,189,248,0.22);
            }

            .micro-car .roof {
                position: absolute;
                bottom: 48px;
                left: 74px;
                width: 98px;
                height: 38px;
                border-radius: 55px 55px 8px 8px;
                background: rgba(255,255,255,0.16);
                border: 1px solid rgba(255,255,255,0.22);
            }

            .micro-car .w1, .micro-car .w2 {
                position: absolute;
                bottom: 6px;
                width: 34px;
                height: 34px;
                border-radius: 50%;
                background: #020617;
                border: 7px solid #e2e8f0;
            }

            .micro-car .w1 { left: 56px; }
            .micro-car .w2 { right: 56px; }

            .hero-caption {
                color: #cbd5e1;
                font-size: 0.92rem;
                margin-top: 12px;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 10px;
            }

            .stTabs [data-baseweb="tab"] {
                border-radius: 999px;
                padding: 10px 18px;
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.10);
            }

            .stTabs [aria-selected="true"] {
                background: linear-gradient(90deg, rgba(56,189,248,0.20), rgba(167,139,250,0.20));
                border: 1px solid rgba(125,211,252,0.35);
            }

            div[data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.055);
                border: 1px solid rgba(255, 255, 255, 0.10);
                border-radius: 18px;
                padding: 16px;
                box-shadow: 0 16px 40px rgba(0,0,0,0.18);
            }

            div[data-testid="stDataFrame"] {
                border-radius: 18px;
                overflow: hidden;
                border: 1px solid rgba(255,255,255,0.08);
            }

            .spotlight-card {
                display: grid;
                grid-template-columns: 1.1fr 0.9fr;
                gap: 26px;
                align-items: center;
                border-radius: 26px;
                padding: 28px;
                border: 1px solid rgba(255,255,255,0.13);
                background:
                    linear-gradient(135deg, rgba(15,23,42,0.92), rgba(30,41,59,0.65)),
                    radial-gradient(circle at 85% 30%, rgba(56,189,248,0.18), transparent 28%);
                box-shadow: 0 28px 80px rgba(0,0,0,0.32);
                margin: 18px 0 24px;
            }

            .eyebrow {
                display: inline-block;
                color: #67e8f9;
                text-transform: uppercase;
                letter-spacing: 0.14em;
                font-size: 0.75rem;
                font-weight: 800;
                margin-bottom: 8px;
            }

            .spotlight-copy h2 {
                color: white;
                font-size: clamp(1.7rem, 3vw, 2.7rem);
                line-height: 1.05;
                letter-spacing: -0.04em;
                margin: 0 0 10px;
            }

            .spotlight-copy h2 span {
                color: #cbd5e1;
                font-weight: 600;
                font-size: 0.7em;
            }

            .spotlight-copy p {
                color: #cbd5e1;
                font-size: 1rem;
                line-height: 1.55;
                margin-bottom: 18px;
            }

            .spotlight-metrics {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 12px;
            }

            .spotlight-metrics div {
                background: rgba(255,255,255,0.07);
                border: 1px solid rgba(255,255,255,0.10);
                padding: 13px 14px;
                border-radius: 16px;
            }

            .spotlight-metrics small {
                color: #94a3b8;
                display: block;
                margin-bottom: 3px;
            }

            .spotlight-metrics strong {
                color: #fff;
                font-size: 1.05rem;
            }

            .vehicle-art {
                position: relative;
                border-radius: 24px;
                min-height: 245px;
                padding: 18px;
                background:
                    radial-gradient(circle at 50% 30%, var(--glow), transparent 42%),
                    linear-gradient(145deg, rgba(255,255,255,0.10), rgba(255,255,255,0.035));
                border: 1px solid rgba(255,255,255,0.13);
                overflow: hidden;
            }

            .vehicle-art::after {
                content: "";
                position: absolute;
                inset: auto 18px 22px 18px;
                height: 4px;
                background: linear-gradient(90deg, transparent, var(--accent), transparent);
                filter: blur(1px);
                opacity: 0.7;
            }

            .vehicle-art-topline {
                display: flex;
                justify-content: space-between;
                color: #cbd5e1;
                font-size: 0.75rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                z-index: 2;
                position: relative;
            }

            .car-silhouette {
                width: 100%;
                height: 150px;
                position: relative;
                margin-top: 10px;
            }

            .car-body {
                position: absolute;
                left: 8%;
                right: 8%;
                bottom: 35px;
                height: 48px;
                background: linear-gradient(90deg, var(--accent), #e0e7ff);
                border-radius: 26px 58px 18px 18px;
                box-shadow: 0 16px 55px var(--glow);
            }

            .car-roof {
                position: absolute;
                left: 28%;
                right: 28%;
                bottom: 76px;
                height: 48px;
                background: rgba(255,255,255,0.18);
                border: 1px solid rgba(255,255,255,0.25);
                border-radius: 60px 60px 8px 8px;
                transform: skewX(-8deg);
            }

            .wheel {
                position: absolute;
                bottom: 18px;
                width: 42px;
                height: 42px;
                border-radius: 50%;
                background: #020617;
                border: 9px solid #e2e8f0;
                z-index: 3;
            }

            .wheel-left { left: 22%; }
            .wheel-right { right: 22%; }

            .vehicle-art-title {
                color: white;
                font-weight: 850;
                font-size: 1.35rem;
                letter-spacing: -0.03em;
                position: relative;
                z-index: 2;
            }

            .vehicle-art-subtitle {
                color: #94a3b8;
                position: relative;
                z-index: 2;
            }



            .vehicle-photo-card {
                position: relative;
                min-height: 278px;
                border-radius: 24px;
                overflow: hidden;
                border: 1px solid rgba(255,255,255,0.14);
                background: rgba(15,23,42,0.72);
                box-shadow: 0 24px 70px rgba(0,0,0,0.34);
            }

            .vehicle-photo-card img {
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                object-fit: cover;
                transform: scale(1.02);
                filter: saturate(1.06) contrast(1.02);
            }

            .vehicle-photo-overlay {
                position: absolute;
                inset: 0;
                background:
                    linear-gradient(180deg, rgba(2,6,23,0.04), rgba(2,6,23,0.88)),
                    linear-gradient(90deg, rgba(2,6,23,0.20), rgba(2,6,23,0.03));
            }

            .vehicle-photo-meta {
                position: absolute;
                left: 18px;
                right: 18px;
                bottom: 18px;
                z-index: 2;
            }

            .vehicle-photo-tags {
                display: flex;
                justify-content: space-between;
                gap: 10px;
                margin-bottom: 12px;
            }

            .vehicle-photo-tags span {
                color: #e0f2fe;
                background: rgba(2,6,23,0.58);
                border: 1px solid rgba(255,255,255,0.18);
                padding: 7px 10px;
                border-radius: 999px;
                font-size: 0.72rem;
                letter-spacing: 0.10em;
                text-transform: uppercase;
                backdrop-filter: blur(8px);
            }

            .vehicle-photo-card h3 {
                color: white;
                font-size: 1.45rem;
                line-height: 1.05;
                margin: 0 0 6px 0;
                letter-spacing: -0.04em;
                text-shadow: 0 2px 16px rgba(0,0,0,0.6);
            }

            .vehicle-photo-card p {
                color: #cbd5e1;
                margin: 0;
                text-shadow: 0 2px 14px rgba(0,0,0,0.6);
            }

            .vehicle-photo-fallback {
                padding: 20px;
                display: flex;
                flex-direction: column;
                justify-content: end;
                background:
                    radial-gradient(circle at 50% 25%, rgba(56,189,248,0.22), transparent 34%),
                    linear-gradient(145deg, rgba(15,23,42,0.95), rgba(30,41,59,0.82));
            }

            .fallback-main {
                color: #93c5fd;
                font-size: 1rem;
                margin-bottom: 16px;
            }

            .stButton > button, .stDownloadButton > button, a[data-testid="stLinkButton"] {
                border-radius: 999px !important;
                border: 1px solid rgba(125,211,252,0.28) !important;
                background: linear-gradient(90deg, rgba(56,189,248,0.18), rgba(167,139,250,0.18)) !important;
                color: #eef2ff !important;
                font-weight: 700 !important;
            }

            @media (max-width: 850px) {
                .hero-grid, .spotlight-card {
                    grid-template-columns: 1fr;
                }

                .spotlight-metrics {
                    grid-template-columns: 1fr;
                }
            }
        
.real-vehicle-card {
    position: relative;
    border-radius: 24px;
    min-height: 245px;
    padding: 18px;
    background:
        radial-gradient(circle at 50% 20%, rgba(56,189,248,0.18), transparent 42%),
        linear-gradient(145deg, rgba(255,255,255,0.10), rgba(255,255,255,0.035));
    border: 1px solid rgba(255,255,255,0.13);
    overflow: hidden;
}

.image-frame {
    width: 100%;
    height: 165px;
    border-radius: 18px;
    overflow: hidden;
    background: rgba(2, 6, 23, 0.35);
    border: 1px solid rgba(255,255,255,0.10);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 12px;
}

.real-car-image {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    filter: saturate(1.04) contrast(1.03);
}

.image-fallback {
    color: #93c5fd;
    text-align: center;
    font-weight: 700;
}

.fallback-icon {
    font-size: 3rem;
    margin-bottom: 8px;
}

.vehicle-chip-row {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 10px;
}

.vehicle-chip-row span {
    color: #dbeafe;
    background: rgba(15, 23, 42, 0.62);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 999px;
    padding: 7px 10px;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 800;
}


.expert-card {
    border-radius: 22px;
    border: 1px solid rgba(125, 211, 252, 0.22);
    background:
        linear-gradient(135deg, rgba(56,189,248,0.13), rgba(167,139,250,0.12)),
        rgba(15, 23, 42, 0.62);
    padding: 22px;
    margin: 12px 0 20px 0;
    box-shadow: 0 18px 55px rgba(0,0,0,0.22);
}
.expert-eyebrow {
    color: #67e8f9;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.76rem;
    font-weight: 800;
    margin-bottom: 6px;
}
.expert-title {
    color: #ffffff;
    font-size: 1.35rem;
    font-weight: 850;
    letter-spacing: -0.02em;
    margin-bottom: 7px;
}
.expert-copy {
    color: #cbd5e1;
    line-height: 1.55;
}

</style>

        <div class="hero-wrap">
            <div class="hero-grid">
                <div>
                    <div class="hero-badge">AI-assisted car buying • Australia MVP</div>
                    <h1 class="hero-title">Find the right car. <span>Skip the bad deal.</span></h1>
                    <div class="hero-subtitle">
                        AutoAdvisor AI recommends cars from your needs, estimates ownership cost,
                        analyses listing risk, and creates buyer reports before you message the seller.
                    </div>
                    <div class="hero-pills">
                        <div class="hero-pill">Recommendation engine</div>
                        <div class="hero-pill">Listing risk analysis</div>
                        <div class="hero-pill">Fair-price estimate</div>
                        <div class="hero-pill">Marketplace search</div>
                    </div>
                </div>

                <div class="hero-visual">
                    <div class="handshake-scene">
                        <div class="handshake-line">🤝</div>
                        <div class="micro-car">
                            <div class="roof"></div>
                            <div class="body"></div>
                            <div class="w1"></div>
                            <div class="w2"></div>
                        </div>
                        <div class="hero-caption">Human judgment + data-driven AI assistant</div>
                    </div>
                </div>
            </div>
        </div>
        """
    )




CAR_IMAGE_OVERRIDES = {
    # Stable Wikimedia Commons direct thumbnails for common MVP models.
    # These are representative model images, not guaranteed exact trim/year.
    ("Toyota", "Yaris"): "https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/2012_Toyota_Yaris_TR_VVT-I_1.33_Front.jpg/640px-2012_Toyota_Yaris_TR_VVT-I_1.33_Front.jpg",
    ("Toyota", "Corolla"): "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/2014_Toyota_Corolla_%28ZRE172R%29_Ascent_sedan_%282015-07-09%29_01.jpg/640px-2014_Toyota_Corolla_%28ZRE172R%29_Ascent_sedan_%282015-07-09%29_01.jpg",
    ("Toyota", "Corolla Hybrid"): "https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/2019_Toyota_Corolla_Hybrid_Icon_Tech_1.8_Front.jpg/640px-2019_Toyota_Corolla_Hybrid_Icon_Tech_1.8_Front.jpg",
    ("Toyota", "Camry"): "https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/2018_Toyota_Camry_Ascent_%28ASV70R%29_sedan_%282018-08-06%29_01.jpg/640px-2018_Toyota_Camry_Ascent_%28ASV70R%29_sedan_%282018-08-06%29_01.jpg",
    ("Toyota", "RAV4"): "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/2019_Toyota_RAV4_Design_HEV_CVT_2.5_Front.jpg/640px-2019_Toyota_RAV4_Design_HEV_CVT_2.5_Front.jpg",
    ("Toyota", "Kluger"): "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/2014_Toyota_Kluger_%28GSU50R%29_GXL_wagon_%282015-07-14%29_01.jpg/640px-2014_Toyota_Kluger_%28GSU50R%29_GXL_wagon_%282015-07-14%29_01.jpg",
    ("Toyota", "HiLux"): "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/2021_Toyota_Hilux_Icon_D-4D_2.4_Front.jpg/640px-2021_Toyota_Hilux_Icon_D-4D_2.4_Front.jpg",
    ("Toyota", "Prius"): "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/2016_Toyota_Prius_%28ZVW50R%29_Hybrid_liftback_%282016-04-02%29_01.jpg/640px-2016_Toyota_Prius_%28ZVW50R%29_Hybrid_liftback_%282016-04-02%29_01.jpg",

    ("Mazda", "Mazda3"): "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/2014_Mazda3_%28BM%29_Maxx_sedan_%282015-07-03%29_01.jpg/640px-2014_Mazda3_%28BM%29_Maxx_sedan_%282015-07-03%29_01.jpg",
    ("Mazda", "CX-5"): "https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/2017_Mazda_CX-5_%28KF%29_Maxx_Sport_wagon_%282018-08-06%29_01.jpg/640px-2017_Mazda_CX-5_%28KF%29_Maxx_Sport_wagon_%282018-08-06%29_01.jpg",
    ("Mazda", "CX-3"): "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/2015_Mazda_CX-3_%28DK%29_Maxx_Sport_wagon_%282018-10-01%29_01.jpg/640px-2015_Mazda_CX-3_%28DK%29_Maxx_Sport_wagon_%282018-10-01%29_01.jpg",
    ("Mazda", "CX-30"): "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d8/2020_Mazda_CX-30_Sport_Tech_SKYACTIV-X_2.0_Front.jpg/640px-2020_Mazda_CX-30_Sport_Tech_SKYACTIV-X_2.0_Front.jpg",

    ("Hyundai", "i30"): "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/2017_Hyundai_i30_SE_Nav_CRDi_1.6_Front.jpg/640px-2017_Hyundai_i30_SE_Nav_CRDi_1.6_Front.jpg",
    ("Hyundai", "Elantra"): "https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/2017_Hyundai_Elantra_%28AD%29_Active_sedan_%282018-08-06%29_01.jpg/640px-2017_Hyundai_Elantra_%28AD%29_Active_sedan_%282018-08-06%29_01.jpg",
    ("Hyundai", "Tucson"): "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/2016_Hyundai_Tucson_%28TL%29_Highlander_wagon_%282018-08-06%29_01.jpg/640px-2016_Hyundai_Tucson_%28TL%29_Highlander_wagon_%282018-08-06%29_01.jpg",
    ("Hyundai", "Santa Fe"): "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/2019_Hyundai_Santa_Fe_Premium_CRDi_2.2_Front.jpg/640px-2019_Hyundai_Santa_Fe_Premium_CRDi_2.2_Front.jpg",

    ("Kia", "Rio"): "https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/2017_Kia_Rio_2_ISG_1.25_Front.jpg/640px-2017_Kia_Rio_2_ISG_1.25_Front.jpg",
    ("Kia", "Cerato"): "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/2018_Kia_Cerato_%28BD%29_S_sedan_%282018-10-22%29_01.jpg/640px-2018_Kia_Cerato_%28BD%29_S_sedan_%282018-10-22%29_01.jpg",
    ("Kia", "Sportage"): "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/2016_Kia_Sportage_1_ISG_1.7_Front.jpg/640px-2016_Kia_Sportage_1_ISG_1.7_Front.jpg",
    ("Kia", "Sorento"): "https://upload.wikimedia.org/wikipedia/commons/thumb/6/61/2015_Kia_Sorento_KX-2_CRDi_2.2_Front.jpg/640px-2015_Kia_Sorento_KX-2_CRDi_2.2_Front.jpg",
    ("Kia", "Carnival"): "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/2019_Kia_Carnival_%28YP%29_Si_wagon_%282019-05-17%29_01.jpg/640px-2019_Kia_Carnival_%28YP%29_Si_wagon_%282019-05-17%29_01.jpg",

    ("Honda", "Civic"): "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/2016_Honda_Civic_%28FC%29_VTi-LX_sedan_%282018-10-01%29_01.jpg/640px-2016_Honda_Civic_%28FC%29_VTi-LX_sedan_%282018-10-01%29_01.jpg",
    ("Honda", "Jazz"): "https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/2015_Honda_Jazz_%28GF%29_VTi_hatchback_%282018-08-06%29_01.jpg/640px-2015_Honda_Jazz_%28GF%29_VTi_hatchback_%282018-08-06%29_01.jpg",
    ("Honda", "CR-V"): "https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/2018_Honda_CR-V_%28RW_MY18%29_VTi-LX_wagon_%282018-10-01%29_01.jpg/640px-2018_Honda_CR-V_%28RW_MY18%29_VTi-LX_wagon_%282018-10-01%29_01.jpg",

    ("Subaru", "Forester"): "https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/2019_Subaru_Forester_2.5i-S_%28AWD%29_wagon_%282018-10-22%29_01.jpg/640px-2019_Subaru_Forester_2.5i-S_%28AWD%29_wagon_%282018-10-22%29_01.jpg",
    ("Subaru", "Outback"): "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/2015_Subaru_Outback_2.5i_Premium_%28MY15%29_wagon_%282015-07-09%29_01.jpg/640px-2015_Subaru_Outback_2.5i_Premium_%28MY15%29_wagon_%282015-07-09%29_01.jpg",

    ("Suzuki", "Swift"): "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e2/2017_Suzuki_Swift_SZ5_Boosterjet_SHVS_1.0_Front.jpg/640px-2017_Suzuki_Swift_SZ5_Boosterjet_SHVS_1.0_Front.jpg",
    ("Suzuki", "Baleno"): "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/2016_Suzuki_Baleno_SZ5_Boosterjet_1.0_Front.jpg/640px-2016_Suzuki_Baleno_SZ5_Boosterjet_1.0_Front.jpg",
    ("Suzuki", "Vitara"): "https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/2015_Suzuki_Vitara_SZ5_DDiS_Allgrip_1.6_Front.jpg/640px-2015_Suzuki_Vitara_SZ5_DDiS_Allgrip_1.6_Front.jpg",

    ("Nissan", "X-Trail"): "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/2017_Nissan_X-Trail_N-Connecta_DCi_1.6_Front.jpg/640px-2017_Nissan_X-Trail_N-Connecta_DCi_1.6_Front.jpg",
    ("Nissan", "Qashqai"): "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/2018_Nissan_Qashqai_N-Connecta_DCi_1.5_Front.jpg/640px-2018_Nissan_Qashqai_N-Connecta_DCi_1.5_Front.jpg",
    ("Nissan", "Navara"): "https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/2017_Nissan_Navara_Tekna_DCi_2.3_Front.jpg/640px-2017_Nissan_Navara_Tekna_DCi_2.3_Front.jpg",

    ("Mitsubishi", "Outlander"): "https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/2018_Mitsubishi_Outlander_4h_PHEV_CVT_2.4_Front.jpg/640px-2018_Mitsubishi_Outlander_4h_PHEV_CVT_2.4_Front.jpg",
    ("Mitsubishi", "ASX"): "https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/2017_Mitsubishi_ASX_3_Di-D_4WD_2.2_Front.jpg/640px-2017_Mitsubishi_ASX_3_Di-D_4WD_2.2_Front.jpg",
    ("Mitsubishi", "Triton"): "https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/2019_Mitsubishi_L200_Barbarian_Di-D_2.4_Front.jpg/640px-2019_Mitsubishi_L200_Barbarian_Di-D_2.4_Front.jpg",

    ("Ford", "Ranger"): "https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/2019_Ford_Ranger_Wildtrak_EcoBlue_2.0_Front.jpg/640px-2019_Ford_Ranger_Wildtrak_EcoBlue_2.0_Front.jpg",
    ("Ford", "Focus"): "https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/2015_Ford_Focus_Zetec_S_TDCi_1.5_Front.jpg/640px-2015_Ford_Focus_Zetec_S_TDCi_1.5_Front.jpg",

    ("Volkswagen", "Golf"): "https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/2017_Volkswagen_Golf_SE_Navigation_TDi_1.6_Front.jpg/640px-2017_Volkswagen_Golf_SE_Navigation_TDi_1.6_Front.jpg",
    ("Volkswagen", "Tiguan"): "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/2017_Volkswagen_Tiguan_SEL_TDi_BMT_4Motion_2.0_Front.jpg/640px-2017_Volkswagen_Tiguan_SEL_TDi_BMT_4Motion_2.0_Front.jpg",

    ("Audi", "A4"): "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/2016_Audi_A4_S_Line_TDi_Ultra_S-A_2.0_Front.jpg/640px-2016_Audi_A4_S_Line_TDi_Ultra_S-A_2.0_Front.jpg",
    ("BMW", "3 Series"): "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/2016_BMW_320d_M_Sport_Automatic_2.0_Front.jpg/640px-2016_BMW_320d_M_Sport_Automatic_2.0_Front.jpg",
    ("Mercedes-Benz", "C-Class"): "https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/2015_Mercedes-Benz_C220_SE_BlueTEC_Automatic_2.1_Front.jpg/640px-2015_Mercedes-Benz_C220_SE_BlueTEC_Automatic_2.1_Front.jpg",
}


def normalise_model_for_image(model):
    model = str(model).strip()

    remove_suffixes = [
        " Hybrid",
        " Electric",
        " Petrol",
        " Diesel",
        " PHEV",
        " Petrol Hybrid",
    ]

    for suffix in remove_suffixes:
        if model.endswith(suffix):
            return model[: -len(suffix)].strip()

    return model


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def get_wikipedia_image(make, model):
    queries = [
        f"{make} {model}",
        f"{make} {normalise_model_for_image(model)}",
    ]

    for query in queries:
        try:
            response = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "format": "json",
                    "prop": "pageimages",
                    "piprop": "original|thumbnail",
                    "pithumbsize": 800,
                    "redirects": 1,
                    "titles": query,
                },
                timeout=4,
            )

            data = response.json()
            pages = data.get("query", {}).get("pages", {})

            for page in pages.values():
                image_url = page.get("original", {}).get("source") or page.get("thumbnail", {}).get("source")

                if image_url:
                    return image_url
        except Exception:
            continue

    return None


def get_car_image_url(make, model):
    key = (str(make), str(model))

    if key in CAR_IMAGE_OVERRIDES:
        return CAR_IMAGE_OVERRIDES[key]

    normalised_key = (str(make), normalise_model_for_image(model))

    if normalised_key in CAR_IMAGE_OVERRIDES:
        return CAR_IMAGE_OVERRIDES[normalised_key]

    return get_wikipedia_image(str(make), str(model))


def render_real_vehicle_card(row):
    make = escape(str(row["make"]))
    model = escape(str(row["model"]))
    year_range = escape(str(row["year_range"]))
    body_type = escape(str(row.get("body_type", "Car")))
    fuel_type = escape(str(row.get("fuel_type", "Fuel")))
    image_url = get_car_image_url(row["make"], row["model"])

    if image_url:
        image_html = f'<img class="real-car-image" src="{escape(image_url)}" alt="{make} {model}">'
    else:
        image_html = """
        <div class="image-fallback">
            <div class="fallback-icon">🚗</div>
            <div>Image unavailable</div>
        </div>
        """

    return f"""
    <div class="real-vehicle-card">
        <div class="image-frame">
            {image_html}
        </div>
        <div class="vehicle-chip-row">
            <span>{body_type}</span>
            <span>{fuel_type}</span>
        </div>
        <div class="vehicle-art-title">{make} {model}</div>
        <div class="vehicle-art-subtitle">{year_range}</div>
    </div>
    """



# -------------------------
# App UI
# -------------------------

render_css_and_hero()

tab1, tab2, tab3 = st.tabs(["Find My Best Car", "Analyse a Listing", "Dataset"])


with tab1:
    st.subheader("Buyer recommendation engine")

    c1, c2, c3 = st.columns(3)

    with c1:
        budget = st.number_input(
            "Budget (AUD)",
            min_value=5000,
            max_value=150000,
            value=35000,
            step=1000,
            key="rec_budget",
        )

        annual_km = st.number_input(
            "Estimated annual kilometres",
            min_value=1000,
            max_value=60000,
            value=15000,
            step=1000,
            key="rec_km",
        )

        family_size = st.selectbox(
            "Family size / regular passengers",
            ["1-2", "3-4", "5+", "Need 7 seats"],
            key="rec_family",
        )

    with c2:
        ownership_years = st.slider(
            "Ownership period (years)",
            1,
            10,
            5,
            key="rec_ownership",
        )

        fuel_pref = st.selectbox(
            "Fuel preference",
            ["Any", "Petrol", "Hybrid", "Electric", "Diesel"],
            key="rec_fuel",
        )

        body_pref = st.selectbox(
            "Body type preference",
            ["Any", "Hatch/Sedan", "SUV", "Ute", "People mover", "7-seat SUV"],
            key="rec_body",
        )

    with c3:
        age_pref = st.selectbox(
            "Age preference",
            ["Any", "2020-2024 only", "2016-2024", "2012-2024"],
            key="rec_age",
        )

        search_city = st.selectbox(
            "Listing search city",
            ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide", "Canberra", "Hobart", "Darwin", "Gold Coast", "Newcastle"],
            index=0,
            key="rec_search_city",
        )

        st.write(
            "Use the sliders below to tune priorities. "
            "The app will do math, which is apparently less popular than vibes but more useful."
        )

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        reliability_w = st.slider("Reliability", 1, 5, 5, key="rec_rel")
        running_cost_w = st.slider("Low running cost", 1, 5, 5, key="rec_cost")

    with p2:
        safety_w = st.slider("Safety", 1, 5, 4, key="rec_safety")
        space_w = st.slider("Space / practicality", 1, 5, 3, key="rec_space")

    with p3:
        comfort_w = st.slider("Comfort", 1, 5, 3, key="rec_comfort")
        tech_w = st.slider("Technology", 1, 5, 3, key="rec_tech")

    with p4:
        performance_w = st.slider("Performance", 1, 5, 2, key="rec_perf")
        resale_w = st.slider("Resale value", 1, 5, 4, key="rec_resale")

    work, message = score_dataset(
        df,
        budget,
        annual_km,
        family_size,
        ownership_years,
        fuel_pref,
        body_pref,
        age_pref,
        reliability_w,
        running_cost_w,
        safety_w,
        space_w,
        comfort_w,
        tech_w,
        performance_w,
        resale_w,
    )

    if message:
        if work.empty:
            st.error(message)
            st.stop()
        else:
            st.warning(message)

    top = work.head(10).copy()
    top["why_recommended"] = top.apply(
        lambda row: explain_recommendation(row, budget),
        axis=1,
    )

    render_top_match_spotlight(top.iloc[0], ownership_years)

    st.subheader("Top recommendations")

    display_cols = [
        "make",
        "model",
        "year_range",
        "body_type",
        "fuel_type",
        "seats",
        "avg_used_price_aud",
        "final_score",
        "net_ownership_cost_aud",
        "market_segment",
        "buyer_fit",
        "why_recommended",
    ]

    show = top[display_cols].copy()
    if "display_year_range" in top.columns and "year_range" in show.columns:
        show["year_range"] = top["display_year_range"].values

    show.insert(
        0,
        "photo",
        [get_vehicle_image_url(row["make"], row["model"]) or "" for _, row in top.iterrows()],
    )
    show["final_score"] = show["final_score"].round(1)
    show["avg_used_price_aud"] = show["avg_used_price_aud"].map(money)
    show["net_ownership_cost_aud"] = show["net_ownership_cost_aud"].map(money)

    try:
        st.dataframe(
            show,
            use_container_width=True,
            hide_index=True,
            column_config={"photo": st.column_config.ImageColumn("Photo", width="small")},
        )
    except Exception:
        st.dataframe(show.drop(columns=["photo"]), use_container_width=True, hide_index=True)

    best = top.iloc[0]

    st.markdown("---")
    st.subheader("Best match summary")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Best match", f"{best['make']} {best['model']} ({best.get('display_year_range', best['year_range'])})")

    with m2:
        st.metric("Final score", f"{best['final_score']:.1f}/100")

    with m3:
        st.metric(
            f"Estimated {ownership_years}-year net ownership cost",
            money(best["net_ownership_cost_aud"]),
        )

    st.write(explain_recommendation(best, budget))

    st.subheader("Find live listings for the best match")
    render_listing_search_buttons(
        best["make"],
        best["model"],
        best["year_range"],
        max_price=budget,
        city=search_city,
        key_suffix="best_match",
    )
    st.caption(
        "Choose one or more years, then open the marketplace links. Carsales uses year/make/model URLs. Gumtree, Facebook, and Cars24 include the selected year in the search query where possible."
    )

    with st.expander("Search links for top 5 recommendations"):
        for _, row in top.head(5).iterrows():
            st.markdown(f"**{row['make']} {row['model']} ({row.get('display_year_range', row['year_range'])})**")
            render_listing_search_buttons(
                row["make"],
                row["model"],
                row["year_range"],
                max_price=budget,
                city=search_city,
                key_suffix=f"top5_{row.name}",
            )

    st.subheader("Score breakdown for top 5")

    breakdown_cols = [
        "make",
        "model",
        "year_range",
        "budget_fit_score",
        "reliability_score",
        "running_cost_score",
        "safety_score",
        "family_fit_score",
        "comfort_score",
        "technology_score",
        "performance_score",
        "resale_score",
    ]

    breakdown = top.head(5)[breakdown_cols].copy()
    if "display_year_range" in top.columns and "year_range" in breakdown.columns:
        breakdown["year_range"] = top.head(5)["display_year_range"].values

    for col in breakdown_cols[3:]:
        breakdown[col] = breakdown[col].round(1)

    st.dataframe(breakdown, use_container_width=True, hide_index=True)


with tab2:
    st.subheader("Listing analyser")
    st.write(
        "Paste the important details from a listing, or enter them manually. "
        "This is not live scraping yet. We’re doing the responsible thing first, annoying as that is."
    )

    with st.expander("Paste raw listing text and auto-fill fields", expanded=True):
        raw_listing_text = st.text_area(
            "Raw listing text",
            height=140,
            placeholder=(
                "Example: 2015 Audi A4, 85,000km, full service history, $16,000, "
                "NSW, private seller, clean title..."
            ),
            key="raw_listing_text",
        )

        if st.button("Extract details from text"):
            extracted = extract_listing_details(raw_listing_text, df)

            extracted_messages = []

            if extracted["make"] in df["make"].unique():
                st.session_state["listing_make"] = extracted["make"]
                extracted_messages.append(f"Make: {extracted['make']}")

            if extracted["make"] and extracted["model"]:
                valid_models = df[df["make"] == extracted["make"]]["model"].unique()

                if extracted["model"] in valid_models:
                    st.session_state["listing_model"] = extracted["model"]
                    extracted_messages.append(f"Model: {extracted['model']}")
            elif extracted["make"] and not extracted["model"]:
                # Do not guess the model from make alone. That is how apps become confidently wrong.
                st.session_state["listing_model"] = "Select model"
                extracted_messages.append("Model not detected - please select it manually")

            if extracted["year_range"]:
                st.session_state["detected_year_range"] = extracted["year_range"]

            if extracted["make"] and extracted["model"] and extracted["year_range"]:
                valid_ranges = df[
                    (df["make"] == extracted["make"])
                    & (df["model"] == extracted["model"])
                ]["year_range"].unique()

                if extracted["year_range"] in valid_ranges:
                    st.session_state["listing_year"] = extracted["year_range"]
                    extracted_messages.append(
                        f"Year: {extracted['year']} → {extracted['year_range']}"
                    )
            elif extracted["year"] and extracted["year_range"]:
                extracted_messages.append(
                    f"Year detected: {extracted['year']} → {extracted['year_range']}"
                )

            if extracted["price"] is not None:
                st.session_state["listing_price"] = extracted["price"]
                extracted_messages.append(f"Price: {money(extracted['price'])}")

            if extracted["kilometres"] is not None:
                st.session_state["listing_kms"] = extracted["kilometres"]
                extracted_messages.append(f"Kilometres: {extracted['kilometres']:,.0f} km")

            if extracted["service_history"] is not None:
                st.session_state["listing_service"] = extracted["service_history"]
                extracted_messages.append(f"Service history: {extracted['service_history']}")

            if extracted["accident_status"] is not None:
                st.session_state["listing_accident"] = extracted["accident_status"]
                extracted_messages.append(f"Accident/write-off: {extracted['accident_status']}")

            if extracted_messages:
                st.success("Extracted: " + " | ".join(extracted_messages))
            else:
                st.warning(
                    "Could not confidently extract details. Try adding make, model, year, price and kilometres."
                )

    l1, l2, l3 = st.columns(3)

    with l1:
        makes = sorted(df["make"].unique())

        make_default = st.session_state.get("listing_make", makes[0])
        if make_default not in makes:
            make_default = makes[0]

        make = st.selectbox(
            "Make",
            makes,
            index=makes.index(make_default),
            key="listing_make",
        )

        models_for_make = sorted(df[df["make"] == make]["model"].unique())
        model_options = ["Select model"] + models_for_make

        model_default = st.session_state.get("listing_model", "Select model")
        if model_default not in model_options:
            model_default = "Select model"

        model = st.selectbox(
            "Model",
            model_options,
            index=model_options.index(model_default),
            key="listing_model",
        )

        if model == "Select model":
            year_range = "Select model first"
            st.selectbox(
                "Year range",
                ["Select model first"],
                index=0,
                key="listing_year_placeholder",
                disabled=True,
            )
        else:
            year_ranges = sorted(
                df[(df["make"] == make) & (df["model"] == model)]["year_range"].unique()
            )

            year_default = st.session_state.get(
                "listing_year",
                st.session_state.get("detected_year_range", year_ranges[0]),
            )

            if year_default not in year_ranges:
                year_default = st.session_state.get("detected_year_range", year_ranges[0])

            if year_default not in year_ranges:
                year_default = year_ranges[0]

            year_range = st.selectbox(
                "Year range",
                year_ranges,
                index=year_ranges.index(year_default),
                key="listing_year",
            )

    with l2:
        if model == "Select model":
            default_price = 10000
        else:
            default_price = int(
                df[
                    (df["make"] == make)
                    & (df["model"] == model)
                    & (df["year_range"] == year_range)
                ]["avg_used_price_aud"].iloc[0]
            )

        price_default = int(st.session_state.get("listing_price", default_price))
        price_default = max(1000, min(200000, price_default))

        asking_price = st.number_input(
            "Asking price (AUD)",
            min_value=1000,
            max_value=200000,
            value=price_default,
            step=500,
            key="listing_price",
        )

        km_default = int(st.session_state.get("listing_kms", 85000))
        km_default = max(0, min(500000, km_default))

        kilometres = st.number_input(
            "Kilometres",
            min_value=0,
            max_value=500000,
            value=km_default,
            step=5000,
            key="listing_kms",
        )

        state_options = ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"]

        state_default = st.session_state.get("listing_state", "NSW")
        if state_default not in state_options:
            state_default = "NSW"

        state = st.selectbox(
            "State / territory",
            state_options,
            index=state_options.index(state_default),
            key="listing_state",
        )

        city_options = ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide", "Canberra", "Hobart", "Darwin", "Gold Coast", "Newcastle"]

        city_default = st.session_state.get("listing_city", state_to_default_city(state))
        if city_default not in city_options:
            city_default = state_to_default_city(state)

        city = st.selectbox(
            "Listing search city",
            city_options,
            index=city_options.index(city_default),
            key="listing_city",
        )

        listing_url = st.text_input(
            "Original listing URL",
            value=st.session_state.get("listing_original_url", ""),
            key="listing_original_url",
            placeholder="Optional - paste Carsales/Gumtree/Cars24/Facebook link",
        )

    with l3:
        seller_options = ["Private", "Dealer"]

        seller_default = st.session_state.get("listing_seller", "Private")
        if seller_default not in seller_options:
            seller_default = "Private"

        seller_type = st.selectbox(
            "Seller type",
            seller_options,
            index=seller_options.index(seller_default),
            key="listing_seller",
        )

        service_options = ["Full", "Partial", "No / missing"]

        service_default = st.session_state.get("listing_service", "Full")
        if service_default not in service_options:
            service_default = "Full"

        service_history = st.selectbox(
            "Service history",
            service_options,
            index=service_options.index(service_default),
            key="listing_service",
        )

        accident_options = [
            "No concern reported",
            "Unknown",
            "Known accident/write-off concern",
        ]

        accident_default = st.session_state.get("listing_accident", "Unknown")
        if accident_default not in accident_options:
            accident_default = "Unknown"

        accident_concern = st.selectbox(
            "Accident/write-off concern",
            accident_options,
            index=accident_options.index(accident_default),
            key="listing_accident",
        )

    st.markdown("---")
    st.subheader("Vehicle history check")
    st.caption(
        "Use the official PPSR check before any deposit or payment. It is usually VIN-based, not just number-plate-based, because apparently trust is not a data field."
    )

    vh1, vh2 = st.columns(2)

    with vh1:
        vin = st.text_input(
            "VIN / chassis number",
            value=st.session_state.get("listing_vin", ""),
            max_chars=20,
            key="listing_vin",
            placeholder="Example: 17-character VIN",
        )

        vin_status, vin_warning = validate_vin(vin)
        if vin and vin_status == "Looks valid":
            st.success("VIN format looks valid.")
        elif vin and vin_warning:
            st.warning(vin_warning)

    with vh2:
        rego = st.text_input(
            "Registration / number plate",
            value=st.session_state.get("listing_rego", ""),
            max_chars=12,
            key="listing_rego",
            placeholder="Optional",
        )

        st.link_button("Open official PPSR check", PPSR_OFFICIAL_URL, use_container_width=True)
        st.caption("Official PPSR online search is listed by PPSR as $2. Third-party sites often charge more for the same basic idea.")

    ppsr_done = st.radio(
        "Have you completed the official PPSR check?",
        ["No - not checked yet", "Yes - checked", "Unclear / seller has not provided proof"],
        index=0,
        horizontal=True,
        key="ppsr_done",
    )

    if ppsr_done == "Yes - checked":
        ph1, ph2, ph3, ph4 = st.columns(4)

        with ph1:
            finance_status = st.selectbox(
                "Finance/security interest?",
                ["No", "Yes", "Unsure"],
                index=2,
                key="ppsr_finance",
            )

        with ph2:
            written_off_status = st.selectbox(
                "Written off?",
                ["No", "Yes", "Unsure"],
                index=2,
                key="ppsr_written_off",
            )

        with ph3:
            stolen_status = st.selectbox(
                "Stolen?",
                ["No", "Yes", "Unsure"],
                index=2,
                key="ppsr_stolen",
            )

        with ph4:
            takata_status = st.selectbox(
                "Takata recall?",
                ["No", "Yes", "Unsure"],
                index=2,
                key="ppsr_takata",
            )
    else:
        finance_status = "Not checked"
        written_off_status = "Not checked"
        stolen_status = "Not checked"
        takata_status = "Not checked"


    if st.button("Analyse listing", type="primary"):
        if model == "Select model":
            st.error("Please select a model before analysing the listing. I refuse to guess a model from just the make, because apparently we are avoiding chaos today.")
            analysis = None
        else:
            analysis = analyse_listing(
                df,
                make,
                model,
                year_range,
                asking_price,
                kilometres,
                seller_type,
                service_history,
                accident_concern,
                state,
            )

            if analysis is not None:
                analysis = apply_vehicle_history_checks(
                    analysis,
                    vin,
                    rego,
                    ppsr_done,
                    finance_status,
                    written_off_status,
                    stolen_status,
                    takata_status,
                )

        if analysis is None and model != "Select model":
            st.error("No matching vehicle found in the dataset.")
        elif analysis is not None:
            st.markdown("---")
            st.subheader(f"{make} {model} ({year_range}) listing assessment")

            a1, a2, a3, a4 = st.columns(4)

            with a1:
                st.metric("Risk level", analysis["risk_level"])

            with a2:
                st.metric("Price status", analysis["price_status"])

            with a3:
                st.metric(
                    "Fair range",
                    f"{money(analysis['low_fair'])} - {money(analysis['high_fair'])}",
                )

            with a4:
                st.metric("Suggested first offer", money(analysis["suggested_offer"]))

            st.write(
                f"Prototype adjusted fair price estimate: **{money(analysis['adjusted_fair_price'])}**"
            )
            st.write(
                f"Estimated expected kilometres for this age band: **{analysis['expected_km']:,.0f} km**"
            )
            st.write(
                f"Kilometres vs expected: **{analysis['km_difference']:,.0f} km**"
            )
            st.write(
                f"Asking price difference vs adjusted fair estimate: **{money(analysis['over_under'])}**"
            )

            vehicle_history = analysis.get("vehicle_history", {})
            if vehicle_history:
                st.subheader("Vehicle history / PPSR summary")
                h1, h2, h3, h4 = st.columns(4)

                with h1:
                    st.metric("PPSR checked", vehicle_history.get("ppsr_done", "Not recorded"))

                with h2:
                    st.metric("Finance owing", vehicle_history.get("finance_status", "Not recorded"))

                with h3:
                    st.metric("Written off", vehicle_history.get("written_off_status", "Not recorded"))

                with h4:
                    st.metric("Stolen", vehicle_history.get("stolen_status", "Not recorded"))

                st.caption(
                    "The app does not fetch the official PPSR certificate. It records your result and adjusts the risk logic. "
                    "Always verify the certificate yourself before payment, because repossession is not a fun surprise feature."
                )

            st.subheader("Final verdict")

            if analysis["risk_level"] == "Low":
                st.success(f"{analysis['final_verdict']}: {analysis['verdict_reason']}")
            elif analysis["risk_level"] == "Medium":
                st.warning(f"{analysis['final_verdict']}: {analysis['verdict_reason']}")
            else:
                st.error(f"{analysis['final_verdict']}: {analysis['verdict_reason']}")

            gcol, ccol = st.columns(2)

            with gcol:
                st.success("Green flags")

                if analysis["green_flags"]:
                    for item in analysis["green_flags"]:
                        st.write(f"- {item}")
                else:
                    st.write("- No major green flags identified.")

            with ccol:
                st.error("Critical checks")

                if analysis["critical_checks"]:
                    for item in analysis["critical_checks"]:
                        st.write(f"- {item}")
                else:
                    st.write("- No critical checks triggered.")

            wcol, xcol = st.columns(2)

            with wcol:
                st.warning("Cost warnings")

                if analysis["cost_warnings"]:
                    for item in analysis["cost_warnings"]:
                        st.write(f"- {item}")
                else:
                    st.write("- No major cost warnings triggered.")

            with xcol:
                st.info("Context warnings")

                if analysis["context_warnings"]:
                    for item in analysis["context_warnings"]:
                        st.write(f"- {item}")
                else:
                    st.write("- No extra context warnings triggered.")

            st.subheader("Find similar listings online")
            render_listing_search_buttons(
                make,
                model,
                year_range,
                max_price=asking_price,
                state=state,
                city=city,
                key_suffix="listing_analyser",
            )

            render_expert_help_section(
                analysis,
                make,
                model,
                year_range,
                asking_price,
                kilometres,
                state,
                city,
                listing_url,
            )

            st.caption(
                "These buttons open external marketplace searches. The app is not scraping or importing listings yet, "
                "because apparently building software without angering every marketplace on day one is considered mature."
            )

            st.subheader("Questions to ask the seller")

            for q in analysis["questions"]:
                st.write(f"- {q}")

            st.subheader("Negotiation script")

            negotiation_script = (
                f"Hi, I’m interested in the {make} {model}. Based on similar prototype market estimates, "
                f"the fair range appears to be around {money(analysis['low_fair'])} to {money(analysis['high_fair'])}, "
                f"depending on condition and service history. Would you consider {money(analysis['suggested_offer'])} "
                f"subject to inspection and a clear PPSR check?"
            )

            st.info(negotiation_script)

            report_text = build_listing_report(
                analysis,
                make,
                model,
                year_range,
                asking_price,
                kilometres,
                seller_type,
                service_history,
                accident_concern,
                state,
            )

            file_name = (
                f"autoadvisor_report_{safe_filename(make)}_{safe_filename(model)}_{safe_filename(year_range)}.md"
            )

            st.download_button(
                label="Download buyer report",
                data=report_text,
                file_name=file_name,
                mime="text/markdown",
            )

            st.caption(
                "This is a prototype estimate, not financial advice or a certified valuation. "
                "For real use, verify with official PPSR certificate, inspection, service records, and market data. "
                "Tragically, common sense remains mandatory."
            )


with tab3:
    st.subheader("Dataset preview")
    st.write(f"Dataset rows: **{len(df):,}**")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Prototype limits")
    st.write(
        """
        This MVP uses seed/prototype estimates. Before public release, replace the data with verified sources:

        - Pricing: real listings, dealer feeds, RedBook, AutoGrab, or trained valuation model
        - Safety: verified ANCAP data
        - Vehicle specs: manufacturer/RedBook style data
        - Market valuation: real listing/sold-price data where available
        - Listing analyser: actual listing extraction and verified VIN/PPSR API workflow later
        """
    )
