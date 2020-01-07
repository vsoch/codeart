"""

Copyright (C) 2020 Vanessa Sochat.

This Source Code Form is subject to the terms of the
Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""

from PIL import Image
import pandas
import os


def get_avg_rgb(image):
    """
    get_avg_rgb:
        returns a tuple with (True/False,R,G,B)
        True/False indicates if the image is not empty
        the R G B are red, green, blue values, respectively
    """

    img = Image.open(image)
    width, height = img.size

    # make a list of all pixels in the image
    pixels = img.load()
    data = []
    for x in range(width):
        for y in range(height):
            cpixel = pixels[x, y]
            data.append(cpixel)

    r = 0
    g = 0
    b = 0
    counter = 0

    # loop through all pixels
    # if alpha value is greater than 200/255, add it to the average
    for x in range(len(data)):
        if data[x][3] > 200:
            if not (data[x][0] == 0 and data[x][1] == 0 and data[x][2] == 0):
                # Don't count white, black is 0 so we don't care
                if not (data[x][0] == 255 and data[x][1] == 255 and data[x][2] == 255):
                    r += data[x][0]
                    g += data[x][1]
                    b += data[x][2]
                    counter += 1

    # compute average RGB values
    if counter != 0:
        rAvg = r / counter
        gAvg = g / counter
        bAvg = b / counter
        return (True, rAvg, gAvg, bAvg)
    else:
        return (False, 0, 0, 0)


def generate_color_lookup(png_images, remove_path=False):
    """generate_color_lookup, meaning we iterate through images, get the
       average color for each, and then can use it as a pixel.
    """
    color_lookup = pandas.DataFrame(columns=["R", "G", "B"])
    for png_image in png_images:
        valid, R, G, B = get_avg_rgb(png_image)
        if valid:
            if remove_path == True:
                png_image = os.path.basename(png_image)
            color_lookup.loc[png_image] = [R, G, B]
    return color_lookup
