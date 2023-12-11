import json
import os
from datetime import timezone
from pathlib import Path
from datetime import datetime, timedelta
import argparse
from typing import Dict, List, Optional, Union

import requests
from openai import OpenAI
import tiktoken
from P4 import P4, P4Exception

ROOT_DIR = Path(__file__).parent.parent.resolve()

MAX_TOKENS = 100000

HISTORY_FILE = ROOT_DIR / "data/history.json"
LATEST_TIMESTAMP = ROOT_DIR / "data/latest_timestamp.txt"
CONFIG_FILE = ROOT_DIR / "config.json"

SYSTEM_MESSAGE = """
You are an assistant to an indie game development team located around the world. You are responsible for making sure everyone is aware of what others are doing by reporting on recent submissions to the Helix Core server.

When you are given a JSON payload of recent changelists, please write a report summarizing what people have been working on. The JSON payload will contain a list of changelists, each of which will have a changelist number, username, time (in UTC), description, and a list of edited folders. Keep the tone of the reports casual and friendly, but also professional. Do not give specific details about changelist numbers or times unless they are relevant to the report. If relevant, you could mention how close together some changelists are, but do not give specific times or numbers unless asked.

If you receive a question or instruction instead of a JSON payload, please respond to the best of your ability. For example, if you are asked to give more details on a certain changelist or user, you should respond with a report on that changelist. Or if you are asked to give a summary of what a certain user has been working on, you should respond with a report on that user.

If you have a hard time understanding what people were working on based on their changelist descriptions and edited_folder lists, you can add some friendly suggestions on how they might improve their changelist descriptions to be more descriptive for their teammates. DO NOT do this every time, only when you feel it is necessary to understand what they were working on. Giving specific examples or suggestions is a good way to help them improve their descriptions. Again, DO NOT give suggestions every time, only when the changelist descriptions are difficult to understand.

When writing the message, please only use people's first names unless it is a shared login, in which case you can reference the username itself since we don't know who actually made the changes. You can also use emoji to make the message more fun and engaging.

These messages will be posted to Discord so you do not sign your message with your name.
"""

p4 = P4()
p4.connect()

P4_USERS = p4.run_users()
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK")


def create_report(
    start_time: Optional[str] = None, end_time: Optional[str] = None
) -> Union[str, None]:
    history = get_history()
    recent_changelists = get_recent_changelists(start_time, end_time)
    if not recent_changelists:
        return None
    messages = assemble_messages(
        history, f"```json\n{json.dumps(recent_changelists)}\n```"
    )
    ai_response = get_openai_message(messages)
    save_history(messages, ai_response)
    send_discord_message(ai_response)

    return ai_response


def get_history() -> List[Dict[str, str]]:
    if HISTORY_FILE.exists():
        with open("data/history.json", "r") as f:
            previous_messages = json.load(f)
    else:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.touch()
        previous_messages = []

    return previous_messages


def save_history(messages: List[Dict[str, str]], new_response: str) -> None:
    history = [message for message in messages if message["role"] != "system"]
    history.append({"role": "assistant", "content": new_response})
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)


def assemble_messages(
    history: List[Dict[str, str]], new_message: str
) -> List[Dict[str, str]]:
    return truncate_history(
        [{"role": "system", "content": SYSTEM_MESSAGE}]
        + history
        + [{"role": "user", "content": new_message}]
    )


def truncate_history(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    while num_tokens_from_messages(messages) > MAX_TOKENS:
        print("Too many tokens, removing oldest message")
        messages.pop(1)
    return messages


def num_tokens_from_messages(
    messages: List[Dict[str, str]], model: str = "gpt-4"
) -> int:
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens_per_message = 3
    tokens_per_name = 1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def ask_query(query):
    if HISTORY_FILE.exists():
        with open("data/history.json", "r") as f:
            previous_messages = json.load(f)
    else:
        previous_messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    messages = previous_messages + [{"role": "user", "content": query}]
    messages = truncate_history(messages)

    MODEL = "gpt-4-1106-preview"

    client = OpenAI()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
    )
    print(f"---- Used {response.usage.total_tokens} tokens ----")

    with open(HISTORY_FILE, "w") as f:
        json.dump(
            messages
            + [
                {
                    "role": "assistant",
                    "content": response.choices[0].message.content,
                }
            ],
            f,
            indent=4,
        )
    send_discord_message(response.choices[0].message.content)
    return response.choices[0].message.content


def send_discord_message(message: str) -> None:
    # Prepare the payload
    data = {"content": message}

    # Send the POST request to the Discord webhook URL
    response = requests.post(DISCORD_WEBHOOK, json=data)

    # Check for successful delivery
    if response.status_code != 204:
        print(f"Failed to send message, status code: {response.status_code}")


def get_openai_message(messages: List[Dict[str, str]]) -> str:
    MODEL = "gpt-4-1106-preview"
    client = OpenAI()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
    )
    print(f"---- Used {response.usage.total_tokens} tokens ----")

    return response.choices[0].message.content


def get_recent_changelists(previous_datetime=None, current_datetime=None):
    previous_datetime = get_previous_datetime(previous_datetime)
    current_datetime = get_current_datetime(current_datetime)

    changelist_timestamp = previous_datetime.strftime("%Y/%m/%d:%H:%M:%S")
    current_timestamp = current_datetime.strftime("%Y/%m/%d:%H:%M:%S")
    changelists = p4.run_changes("-r", f"@{changelist_timestamp},@{current_timestamp}")

    if not changelists:
        return []

    try:
        cl_details = p4.run_describe("-s", *(cl["change"] for cl in changelists))
    except P4Exception:
        return []

    latest_timestamp = datetime.utcfromtimestamp(int(cl_details[-1]["time"])).strftime(
        "%Y/%m/%d:%H:%M:%S"
    )
    with open(LATEST_TIMESTAMP, "w") as f:
        f.write(latest_timestamp)

    for cl in cl_details:
        if cl.get("depotFile"):
            cl["unique_folders"] = list(extract_unique_directories(cl["depotFile"]))
        else:
            cl["unique_folders"] = []

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


def get_current_datetime(current_datetime: Optional[str] = None) -> datetime:
    if current_datetime is None:
        current_datetime = datetime.now(timezone.utc)
    else:
        current_datetime = datetime.strptime(current_datetime, "%Y/%m/%d:%H:%M:%S")
    return current_datetime


def get_previous_datetime(previous_datetime: Optional[str] = None) -> datetime:
    if previous_datetime is None:
        if LATEST_TIMESTAMP.exists():
            with open(LATEST_TIMESTAMP, "r") as f:
                previous_datetime = datetime.strptime(
                    f.read().strip(), "%Y/%m/%d:%H:%M:%S"
                )
        else:
            previous_datetime = datetime.now(timezone.utc) - timedelta(days=1)
    else:
        previous_datetime = datetime.strptime(previous_datetime, "%Y/%m/%d:%H:%M:%S")
    previous_datetime += timedelta(seconds=1)
    return previous_datetime


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

    # If our list is long, we will go up a level to keep it more succinct.
    unique_dirs_list = list(unique_dirs)
    while len(unique_dirs_list) > 12:
        unique_dirs = set()
        for directory in unique_dirs_list:
            unique_dirs.add(str(Path(directory).parent))
        unique_dirs_list = list(unique_dirs)

    return unique_dirs_list


from datetime import timezone

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", help="Start date in YYYY/MM/DD:HH:MM:SS format")
    parser.add_argument("--query", "-q", help="A plaintext query to ask the LLM.")

    args = parser.parse_args()

    if args.history:
        start_date = datetime.strptime(args.history, "%Y/%m/%d:%H:%M:%S")
        past_dates = [start_date.strftime("%Y/%m/%d:%H:%M:%S")]
        while start_date < datetime.now(timezone.utc):
            start_date += timedelta(days=1)
            past_dates.append(start_date.strftime("%Y/%m/%d:%H:%M:%S"))
        for i in range(len(past_dates) - 1):
            print(f"Running {past_dates[i]} to {past_dates[i + 1]}")
            if res := create_report(past_dates[i], past_dates[i + 1]):
                print(res)

    elif args.query:
        res = ask_query(args.query)
        print(res)
    else:
        res = create_report()
        print(res)
