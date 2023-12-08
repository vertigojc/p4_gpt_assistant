import json
from pathlib import Path
from datetime import datetime, timedelta
import argparse

import openai
from P4 import P4, P4Exception

HISTORY_FILE = Path("data/history.json")
LATEST_TIMESTAMP = Path("data/latest_timestamp.txt")

SYSTEM_MESSAGE = """
You are an assistant to an indie game development team located around the world. You are responsible for making sure everyone is aware of what others are doing by reporting on recent submissions to the Helix Core server.

When you are given a JSON payload of recent changelists, please write a report summarizing what people have been working on. The JSON payload will contain a list of changelists, each of which will have a changelist number, username, time, description, and a list of edited folders. Keep the tone of the reports casual and friendly, but also professional.

If you have a hard time understanding what people were working on based on their changelist descriptions and edited_folder lists, you can add some friendly suggestions on how they might improve their changelist descriptions to be more descriptive for their teammates. DO NOT do this every time, only when you feel it is necessary to understand what they were working on. Giving specific examples or suggestions is a good way to help them improve their descriptions. Again, DO NOT give suggestions every time, only when the changelist descriptions are difficult to understand.

When writing the message, please only use people's first names unless it is a shared login, in which case you can reference the username itself since we don't know who actually made the changes. You can also use emoji to make the message more fun and engaging.

These messages will be posted to Discord so you do not sign your message with your name.
"""

p4 = P4()
p4.connect()

P4_USERS = p4.run_users()


def main(start_time=None, end_time=None):
    # TODO: Get this from a JSON file.
    if HISTORY_FILE.exists():
        with open("data/history.json", "r") as f:
            previous_messages = json.load(f)
    else:
        previous_messages = []

    recent_changelists = get_recent_changelists(start_time, end_time)

    if not recent_changelists:
        return None

    ai_response = get_openai_message(previous_messages, recent_changelists)

    return ai_response


def get_openai_message(previous_messages, recent_changelists):
    messages = (
        [{"role": "system", "content": SYSTEM_MESSAGE}]
        + previous_messages
        + [
            {
                "role": "user",
                "content": f"```json\n{json.dumps(recent_changelists)}\n```",
            }
        ]
    )

    MODEL = "gpt-4-1106-preview"
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0.8,
    )

    with open(HISTORY_FILE, "w") as f:
        json.dump(
            messages
            + [
                {
                    "role": "assistant",
                    "content": response["choices"][0]["message"]["content"],
                }
            ],
            f,
            indent=4,
        )
    return response["choices"][0]["message"]["content"]


def get_recent_changelists(previous_datetime=None, current_datetime=None):
    if previous_datetime is None:
        if LATEST_TIMESTAMP.exists():
            with open(LATEST_TIMESTAMP, "r") as f:
                datetime.strptime(f.read(), "%Y/%m/%d:%H:%M:%S")
        else:
            previous_datetime = datetime.utcnow() - timedelta(days=1)
    else:
        previous_datetime = datetime.strptime(previous_datetime, "%Y/%m/%d:%H:%M:%S")
    previous_datetime += timedelta(seconds=1)

    if current_datetime is None:
        current_datetime = datetime.utcnow()
    else:
        current_datetime = datetime.strptime(current_datetime, "%Y/%m/%d:%H:%M:%S")

    changelist_timestamp = previous_datetime.strftime("%Y/%m/%d:%H:%M:%S")
    current_timestamp = current_datetime.strftime("%Y/%m/%d:%H:%M:%S")
    changelists = p4.run_changes("-r", f"@{changelist_timestamp},@{current_timestamp}")

    if not changelists:
        return []

    try:
        cl_details = p4.run_describe("-s", *(cl["change"] for cl in changelists))
    except P4Exception:
        return []

    for cl in cl_details:
        if cl.get("depotFile"):
            cl["unique_folders"] = list(extract_unique_directories(cl["depotFile"]))
        else:
            cl["unique_folders"] = []

    latest_timestamp = datetime.utcfromtimestamp(int(cl_details[-1]["time"])).strftime(
        "%Y/%m/%d:%H:%M:%S"
    )
    with open(LATEST_TIMESTAMP, "w") as f:
        f.write(latest_timestamp)

    return [
        {
            "changelist_number": cl["change"],
            "username": cl["user"],
            "user_full_name": next(
                (user["FullName"] for user in P4_USERS if user["User"] == cl["user"]),
                cl["user"],
            ),
            "time": datetime.utcfromtimestamp(int(cl["time"])).strftime(
                "%Y/%m/%d:%H:%M:%S"
            ),
            "description": cl["desc"],
            "edited_folders": cl["unique_folders"],
        }
        for cl in cl_details
        if cl["unique_folders"]
    ]


def extract_unique_directories(file_paths):
    unique_dirs = set()

    for path in file_paths:
        # Extract the directory from the file path
        parent_dir = Path(path).parent
        if "__ExternalActors__" in str(parent_dir) or "__ExternalObjects" in str(
            parent_dir
        ):
            parent_dir = parent_dir.parent.parent
        unique_dirs.add(str(parent_dir))

    return unique_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", help="Start date in YYYY/MM/DD:HH:MM:SS format")
    if parser.parse_args().history:
        start_date = parser.parse_args().history
        start_date = datetime.strptime("2023/11/05:00:00:00", "%Y/%m/%d:%H:%M:%S")
        past_dates = [start_date.strftime("%Y/%m/%d:%H:%M:%S")]
        while start_date < datetime.utcnow():
            start_date += timedelta(days=1)
            past_dates.append(start_date.strftime("%Y/%m/%d:%H:%M:%S"))

        all_responses = []
        for i in range(len(past_dates) - 1):
            print(f"Running {past_dates[i]} to {past_dates[i + 1]}")
            res = main(past_dates[i], past_dates[i + 1])
            if res:
                all_responses.append(res)
                print(res)
    else:
        res = main()
        print(res)
