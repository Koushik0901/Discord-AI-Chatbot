# import all the libraries
import os
import discord
from tensorflow import keras
import json
import numpy as np
import random
import pickle

import requests
from bs4 import BeautifulSoup

# open the intents file
with open("intents.json") as file:
    data = json.load(file)

# load the trained model
model = keras.models.load_model("checkpoints/chat_model.h5")

# load tokenizer object
with open("checkpoints/tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# load label encoder object
with open("checkpoints/label_encoder.pickle", "rb") as enc:
    lbl_encoder = pickle.load(enc)


def get_songs(text):
    # Send request to the website and get the response
    response = requests.get("https://www.billboard.com/charts/hot-100")

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the elements that contain the song titles and artist names
    songs = soup.find_all(
        "h3",
        class_="c-title a-no-trucate a-font-primary-bold-s u-letter-spacing-0021 lrv-u-font-size-18@tablet lrv-u-font-size-16 u-line-height-125 u-line-height-normal@mobile-max a-truncate-ellipsis u-max-width-330 u-max-width-230@tablet-only",
    )
    artists = soup.find_all(
        "span",
        class_="c-label a-no-trucate a-font-primary-s lrv-u-font-size-14@mobile-max u-line-height-normal@mobile-max u-letter-spacing-0021 lrv-u-display-block a-truncate-ellipsis-2line u-max-width-330 u-max-width-230@tablet-only",
    )
    # Extract the text from the elements
    song_titles = [song.get_text().strip() for song in songs]
    artist_names = [artist.get_text().strip() for artist in artists]

    idx = random.randint(0, 9)

    song_name = song_titles[idx]
    artist_name = artist_names[idx]
    return text.replace("{song_name}", song_name).replace("{artist_name}", artist_name)


def get_movies_and_tv(text, tag):
    # Send request to the website and get the response
    if tag == "movies":
        response = requests.get(
            "https://www.rottentomatoes.com/browse/movies_in_theaters/critics:certified_fresh~sort:popular"
        )
    else:
        response = requests.get(
            "https://www.rottentomatoes.com/browse/tv_series_browse/critics:fresh~sort:popular"
        )
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the elements that contain the song titles and artist names
    names = soup.find_all("span", class_="p--small")[4:]
    ratings = soup.find_all("score-pairs")[4:]

    idx = random.randint(0, 9)

    name = names[idx].get_text().strip()
    score = ratings[idx].attrs["criticsscore"].strip()
    return text.replace("{name}", name).replace("{score}", score)


intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"{client.user} has connected to Discord!")


@client.event
async def on_message(message):
    # we do not want the bot to reply to itself
    if message.author == client.user:
        return

    else:
        inps = message.content.lower()  # getting the input
        if inps.startswith(
            "!"
        ):  # if the input starts with '!' then run the predictions on it
            result = model.predict(
                keras.preprocessing.sequence.pad_sequences(
                    tokenizer.texts_to_sequences([inps[1:]]),
                    truncating="post",
                    maxlen=25,
                    padding="post",
                )
            )
            tag = lbl_encoder.inverse_transform(result)  # getting the tags

            if np.argmax(result) > 0.60:  # setting a threshold
                for i in data["intents"]:
                    if i["tag"] == tag:
                        selected_response = random.choice(
                            i["responses"]
                        )  # select a random response
                        if tag == "songs":
                            bot_response = get_songs(selected_response)
                        elif tag in ["movies", "series"]:
                            bot_response = get_movies_and_tv(selected_response, tag=tag)
                        else:
                            bot_response = selected_response

                await message.channel.send(
                    bot_response.format(message)
                )  # sending the message

            else:
                await message.channel.send(
                    "I didn't get that. Can you explain or try again.".format(message)
                )


if __name__ == "__main__":
    client.run(
        os.environ["DISCORD_TOKEN"],
    )  # bot token
