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
def search_videos():
    # Disable OAuthlib's HTTPS verification when running locally.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    api_key = os.getenv("API_KEY")

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=api_key)

    search_terms = [
        "how to",
        "news",
        "funny",
        "comedy",
        "gaming",
        "movies",
        "music",
        "football"
    ]

    # Add an initial line to the file with headings (optional)
    append_data_to_file("id,title,publishedAt,viewCount,likeCount,categoryId")
    for term in search_terms:
        get_videos(youtube, term)


# Search for videos using the search term and record their ids
def get_videos(youtube, search_term):
    request = youtube.search().list(
        part="snippet",
        maxResults=50,
        q=search_term,
        publishedAfter="2023-05-01T00:00:00Z"
    )
    response = request.execute()

    ids = []
    for item in response["items"]:
        if item["id"]["kind"] != "youtube#video":
            continue
        ids.append(item["id"]["videoId"])
    
    save_video_data(youtube, ids)


# Retrieve information about each video and record it to the current file
def save_video_data(youtube, ids):
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


# Main function
def main():
    search_videos()
    print("Done")


if __name__ == "__main__":
    main()