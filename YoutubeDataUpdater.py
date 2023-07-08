import os
import sys

import datetime as dt
from dotenv import load_dotenv

import googleapiclient.discovery
import googleapiclient.errors

# Load the environment variables (API_KEY variable must be set)
load_dotenv()

scopes = ["https://www.googleapis.com/auth/youtube.force-ssl"]

# Get a timestamp to use for the file name
timestamp = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# The path to the file which videos need to be updated
file_to_update = "Dataset 1/data_2023_05_25_17_05_39.csv"


# Append a line to the current file
def append_data_to_file(line):
    path = "data_" + timestamp + ".csv"
    if not os.path.exists(path):
        f = open(path, "x")
        f.close()
    f = open(path, "a")
    f.write(line)
    f.write("\n")
    f.close()


# Initialise YouTube object and start searching for videos
def get_video_ids_from_file(path):
    f = open(path, "r")
    ids = []
    for line in f:
        ids.append(line.split(",")[0].replace("\"", ""))
    f.close()
    return ids


# Paginate the video IDs so they can be searched for in batches of 50
def paginate_ids(ids):
    result = []
    for i in range((len(ids) // 50) + 1):
        result.append([])
    for i in range(0, len(ids)):
        result[i // 50].append(ids[i])
    
    return result


# Get the video data and save to file
def save_video_data(youtube, ids):
    print(ids)
    request = youtube.videos().list(
        part="snippet,contentDetails,statistics",
        id=','.join(ids)
    )
    response = request.execute()

    for item in response["items"]:
        data = ""
        data += "\"" + item["id"] + "\","
        data += "\"" + str(item["snippet"]["title"].encode('utf-8')) + "\","
        data += item["snippet"]["publishedAt"] + ","
        data += item["statistics"]["viewCount"] + ","
        if "statistics" in item and "likeCount" in item["statistics"]:
            data += item["statistics"]["likeCount"] + ","
        data += item["snippet"]["categoryId"]
        append_data_to_file(data)


# Initialise the YouTube object and update the video data
def get_video_update():
    # Get the videos to update
    ids = get_video_ids_from_file(file_to_update)[1:]
    ids = paginate_ids(ids)

    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    api_key = os.getenv("API_KEY")

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=api_key)

    for id_list in ids:
        save_video_data(youtube, id_list)


if __name__ == "__main__":
    get_video_update()