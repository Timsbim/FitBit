import json
from time import strftime
from pathlib import Path
from shutil import copy2
import pandas as pd
import xarray as xr


""" (A) Folder structure:
 base_path / data / raw / source
"""

base_path = Path("/Users/me/...")
data_path = base_path / "data"
raw_data_path = data_path / "raw"
source_data_path = raw_data_path / "source"


""" (D) Preliminary arranging of data from zip-archive. The data are mainly 
located in the folders: 
 - Physical Activity 
 - Sleep 
 - Stress 
Copy JSON/CSV-files from the sub-folders of raw_data_path / source to 
raw_data_path. 
"""


def suffix_check():
    """Check what kind of files are in the sub-folders of raw_data_path"""
    suffixes = set()
    for file_path in source_data_path.glob("**/*.*"):
        suffixes.add(file_path.suffix)
    print(suffixes)


def collect_data():
    """Collect all JSON/CSV-files from from sub-folders of raw_data_path
    directly into raw_data_path
    """
    for file_path in source_data_path.glob("**/*.json"):
        copy2(file_path, raw_data_path / file_path.name)
    for file_path in source_data_path.glob("**/*.csv"):
        copy2(file_path, raw_data_path / file_path.name)


""" (C) Helper functions """


def log(message):
    print(f"{strftime('%H:%M:%S')}$ {message}")


def file_to_json(infile_path):
    """Load one JSON-file"""
    with open(infile_path, "r", encoding="utf_8") as json_file:
        return json.load(json_file)


def files_to_json(file_list):
    """Read all the files in the list into one JSON-structure"""
    return sum((file_to_json(infile_path) for infile_path in file_list), [])


def files_to_df(file_list, columns, datetime=True):
    """Read all the files in the list into a DataFrame"""
    df = pd.json_normalize(files_to_json(file_list)).rename(columns=columns)
    if datetime:
        df.datetime = pd.to_datetime(df.datetime)
        df = df.set_index("datetime", drop=True).sort_index(ascending=True)
    return df


def files_to_dfslist(file_list, columns, datetime=True):
    """Read every file in the list into a DataFrame and collect them in a
    list
    """
    dfs = []
    for infile_path in file_list:
        df = pd.json_normalize(file_to_json(infile_path)).rename(
            columns=columns
        )
        if datetime:
            df.datetime = pd.to_datetime(df.datetime)
            df = df.set_index("datetime", drop=True).sort_index(ascending=True)
        dfs.append(df)

    return dfs


""" (D) Select the interesting data and package them in a more compact manner: 
"""

""" (D.1) Heart rate data """

""" (D.1.1) Individual measurements (one file per day with several data 
points) 
"""


def prep_heart_rate_details_xr():
    log("Processing heart_rate-YYYY-MM-DD.json files ...")
    columns = {
        "dateTime": "datetime",
        "value.bpm": "bpm",
        "value.confidence": "confidence",
    }
    file_list = sorted(raw_data_path.glob("heart_rate-*.json"))
    (
        xr.concat(
            (
                df.to_xarray()
                for df in files_to_dfslist(file_list, columns=columns)
            ),
            dim="datetime",
        )
        .sortby("datetime", ascending=True)
        .to_netcdf(data_path / "heart_rate.nc", mode="w")
    )
    log("File heart_rate.nc ready")


def prep_heart_rate_details(fmt="hdf"):
    if fmt not in ["hdf", "csv"]:
        print("format should be 'hdf' (default) or 'csv'!")
    log("Processing heart_rate-YYYY-MM-DD.json files ...")

    columns = {
        "dateTime": "datetime",
        "value.bpm": "bpm",
        "value.confidence": "confidence",
    }

    file_list = sorted(raw_data_path.glob("heart_rate-*.json"))
    df = pd.concat(
        files_to_dfslist(file_list, columns=columns), axis="index"
    ).sort_index(ascending=True)

    if fmt == "hdf":
        df.to_hdf(
            str((data_path / "heart_rate.h5").resolve()), key="df", mode="w"
        )
        log("File heart_rate.h5 ready")
    else:
        df.to_csv(data_path / "heart_rate.csv")
        log("File heart_rate.csv ready")

    df.resample("H").mean().to_csv(data_path / "heart_rate_hourly.csv")
    log("File heart_rate_hourly.csv ready")

    df.resample("D").mean().to_csv(data_path / "heart_rate_daily.csv")
    log("File heart_rate_daily.csv ready")


""" (D.1.2) Zoning data (one file per day with one data point) """


def prep_heart_rate_zoning_data():
    log("Processing time_in_heart_rate_zones_YYYY-MM-DD.json files ...")

    def columns_mapper(col_name):
        if col_name == "dateTime":
            return "datetime"
        return col_name.replace("value.valuesInZones.", "").casefold()

    file_list = sorted(raw_data_path.glob("time_in_heart_rate_zones*.json"))
    (
        files_to_df(file_list, columns=columns_mapper).to_csv(
            data_path / "heart_rate_zones.csv"
        )
    )
    log("File heart_rate_zones.csv ready")


""" (D.1.3) Resting heart rate data (one file per year with daily data 
points) 
"""


def prep_heart_rate_resting_data():
    log("Processing resting_heart_rate-YYYY-04-17.json files ...")

    def columns_mapper(col_name):
        return col_name.replace("value.", "").replace("dateTime", "datetime")

    file_list = sorted(raw_data_path.glob("resting_heart_rate-*.json"))
    df = files_to_df(file_list, columns=columns_mapper, datetime=False)

    df = df[df.date.notna()]
    for col in {"datetime", "date"}:
        df[col] = pd.to_datetime(df[col])
    (
        df.set_index("datetime", drop=True)
        .sort_index(ascending=True)
        .to_csv(data_path / "heart_rate_resting.csv")
    )
    log("File heart_rate_resting.csv ready")


""" (D.2) Sleep data """


def prep_sleep_data():
    log("Processing sleep_YYYY-MM-DD.json files ...")
    file_list = sorted(raw_data_path.glob("sleep-*.json"))
    json_data = files_to_json(file_list)

    df = pd.json_normalize(
        json_data, record_path=["levels", "data"], errors="ignore"
    ).rename(columns={"dateTime": "datetime"})
    df.datetime = pd.to_datetime(df.datetime)
    (
        df.set_index("datetime", drop=True)
        .sort_index(ascending=True)
        .to_csv(data_path / "sleep_details.csv")
    )
    log("File sleep_details.csv ready")

    for data in json_data:
        data["levels"] = data["levels"]["summary"]

    def columns_mapper(col_name):
        col_name = col_name.replace("levels_", "").replace(
            "thirtyDayAvgMinutes", "30_day_avg_minutes"
        )
        mapping = {
            "logId": "sleep_log_entry_id",
            "dateOfSleep": "date_of_sleep",
            "startTime": "start_time",
            "endTime": "end_time",
            "minutesToFallAsleep": "minutes_to_fall_asleep",
            "minutesAsleep": "minutes_asleep",
            "minutesAwake": "minutes_awake",
            "minutesAfterWakeup": "minutes_after_wakeup",
            "timeInBed": "time_in_bed",
            "mainSleep": "main_sleep",
        }
        return mapping.get(col_name, col_name)

    df = (
        pd.json_normalize(json_data, sep="_")
        .drop(columns=["type", "infoCode"])
        .rename(columns=columns_mapper)
    )
    for col_name in {"date_of_sleep", "start_time", "end_time"}:
        df[col_name] = pd.to_datetime(df[col_name])
    (
        df.set_index("date_of_sleep", drop=True)
        .sort_index(ascending=True)
        .to_csv(data_path / "sleep_overviews.csv")
    )
    log("File sleep_overviews.csv ready")

    infile_path = raw_data_path / "sleep_score.csv"
    df = pd.read_csv(infile_path).rename(columns={"timestamp": "datetime"})
    df.datetime = pd.to_datetime(df.datetime)
    (
        df.set_index("datetime", drop=True)
        .sort_index(ascending=True)
        .to_csv(data_path / "sleep_scores.csv")
    )
    log("File sleep_scores.csv ready")


""" (D.3) Activity """

""" (D.3.1) Number of steps (several data points per day) """


def prep_steps_data():
    log("Processing steps-YYYY-MM-DD.json files ...")
    file_list = sorted(raw_data_path.glob("steps-*.json"))
    columns = {"dateTime": "datetime", "value": "number"}
    df = files_to_df(file_list, columns=columns)
    df[~df.number.eq("0")].to_csv(data_path / "activity_steps.csv")
    log("File activity_steps.csv ready")


def prep_distance_data():
    log("Processing distance-YYYY-MM-DD.json files ...")
    file_list = sorted(raw_data_path.glob("distance-*.json"))
    columns = {"dateTime": "datetime", "value": "distance"}
    df = files_to_df(file_list, columns=columns)
    df[~df.distance.eq("0")].to_csv(data_path / "activity_distance.csv")
    log("File activity_distance.csv ready")


def prep_activity_level_data():
    log("Processing *activity-level*_minutes-YYYY-MM-DD.json files ...")
    dfs = []
    for level in (
        "sedentary",
        "lightly_active",
        "moderately_active",
        "very_active",
    ):
        columns = {
            "dateTime": "datetime",
            "value": level.replace("_active", ""),
        }
        file_list = sorted(raw_data_path.glob(f"{level}_minutes-*.json"))
        df = files_to_df(file_list, columns=columns)
        dfs.append(df)
    df = pd.concat(dfs, axis="columns")
    (
        df[~df.sedentary.astype(int).eq(1440)].to_csv(
            data_path / "activity_levels.csv"
        )
    )
    log("File activity_levels.csv ready")


def prep_exercise_data():
    log("Processing exercise-NNN.json files ...")

    file_list = sorted(raw_data_path.glob("exercise-*.json"))
    json_data = files_to_json(file_list)

    """ 4. file (300) has one faulty data point """
    activity_level = [
        {"minutes": None, "name": "sedentary"},
        {"minutes": None, "name": "lightly"},
        {"minutes": None, "name": "fairly"},
        {"minutes": None, "name": "very"},
    ]
    heart_rate_zones = [
        {
            "name": "Außerhalb der Zone",
            "min": None,
            "max": None,
            "minutes": None,
        },
        {"name": "Fettverbrennung", "min": None, "max": None, "minutes": None},
        {"name": "Kardio", "min": None, "max": None, "minutes": None},
        {"name": "Höchstleistung", "min": None, "max": None, "minutes": None},
    ]
    for item in json_data:
        if "activityLevel" not in item:
            item["activityLevel"] = activity_level
        if "heartRateZones" not in item:
            item["heartRateZones"] = heart_rate_zones

    columns = {
        "logId": "log_id",
        "activityName": "activity_name",
        "activityTypeId": "activity_type_id",
        "averageHeartRate": "average_heart_rate",
        "activeDuration": "active_duration",
        "lastModified": "last_modified",
        "startTime": "start_time",
    }
    (
        pd.json_normalize(
            json_data,
            record_path="activityLevel",
            meta=[
                "logId",
                "activityName",
                "activityTypeId",
                "averageHeartRate",
                "calories",
                "duration",
                "activeDuration",
                "steps",
                "lastModified",
                "startTime",
            ],
            errors="ignore",
        )
        .rename(columns=columns)
        .set_index("log_id", drop=True)
        .to_csv(data_path / "activity_exercise_1.csv")
    )
    log("File activity_exercise_1.csv ready")
    (
        pd.json_normalize(
            json_data,
            record_path="heartRateZones",
            meta=[
                "logId",
                "activityName",
                "activityTypeId",
                "averageHeartRate",
                "calories",
                "duration",
                "activeDuration",
                "steps",
                "lastModified",
                "startTime",
            ],
            errors="ignore",
        )
        .rename(columns=columns)
        .set_index("log_id", drop=True)
        .to_csv(data_path / "activity_exercise_2.csv")
    )
    log("File activity_exercise_2.csv ready")


# prep_exercise_data()
# prep_heart_rate_details()
