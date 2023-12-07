import json
from pathlib import Path
from datetime import datetime, timedelta

import openai
from P4 import P4, P4Exception

SYSTEM_MESSAGE = """
You are an encouraging and positive assistant to a game development team located around the world. You are responsible for keeping morale high, making sure everyone is aware of what others are doing by reporting on recent submissions to the Helix Core server, and occasionally making helpful suggestions of how users can improve their changelist descriptions to be more clear.

You will be given a JSON payload of recent changelists.

Based on this information, generate a message to send to the team that summarized who has submitted work recently, what they have been working on, and some enthusiasm and encouragement for the team to keep up the good work. If you have a hard time understanding what people were working on based on their changelist descriptions and edited_folder lists, you can add some friendly suggestions on how they might improve their changelist descriptions to be more descriptive for their teammates.
When writing the message, please only use people's first names unless it is a shared login, in which case you can reference the username itself since we don't know who actually made the changes. You can also use emojis and other fun things to make the message more fun and engaging.

These messages will be posted to Discord so you do not sign your message with your name.
"""

p4 = P4()
p4.connect()


def main(start_):
    # TODO: Get this from a JSON file.
    with open("data/history.json", "r") as f:
        previous_messages = json.load(f)

    recent_changelists = get_recent_changelists()

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

    with open("data/history.json", "w") as f:
        json.dump(
            messages
            + [{"role": "assistant", "content": response["choices"][0]["message"]}],
            f,
            indent=4,
        )
    return response["choices"][0]["message"]["content"]


def get_recent_changelists(previous_datetime=None):
    if previous_datetime is None:
        previous_datetime = datetime.utcnow() - timedelta(days=1)
    else:
        previous_datetime = datetime.strptime(previous_datetime, "%Y/%m/%d:%H:%M:%S")
    previous_datetime += timedelta(seconds=1)
    changelist_timestamp = previous_datetime.strftime("%Y/%m/%d:%H:%M:%S")
    changelists = p4.run_changes("-r", f"@{changelist_timestamp},@now")

    if not changelists:
        return [
            f'No changelists since {previous_datetime.strftime("%Y/%m/%d:%H:%M:%S")}. Give some general encouragement to the team!'
        ]

    cl_details = p4.run_describe("-s", *(cl["change"] for cl in changelists))

    for cl in cl_details:
        if cl.get("depotFile"):
            cl["unique_folders"] = list(extract_unique_directories(cl["depotFile"]))
        else:
            cl["unique_folders"] = []

    p4_users = p4.run_users()

    return [
        {
            "changelist_number": cl["change"],
            "username": cl["user"],
            "user_full_name": next(
                user["FullName"] for user in p4_users if user["User"] == cl["user"]
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
    response = main()
    print(response)
