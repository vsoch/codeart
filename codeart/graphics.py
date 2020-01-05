"""

Copyright (C) 2019-2020 Vanessa Sochat.

This Source Code Form is subject to the terms of the
Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

Modified from https://github.com/Visual-mov/Colorful-Julia (MIT License)

"""

from PIL import Image, ImageFont, ImageDraw
from codeart.utils import get_static

import json
import math
import os
import pandas
import shutil
import sys
import tempfile


def generate_interactive_colormap(
    vectors, counts, color_width=20, width=6000, row_height=20, outdir=None
):
    """Based on a set of vectors, generate an interactive colormap.
       with d3. This function should take output from get_vectors and
       get_color_percentages. The xdim and ydim are generated based on
       sorting by columns 0, 1, 2. If an output folder is not provided,
       create one in tmp.
    """
    # Sort by all three columns
    vectors = vectors.sort_values(by=[0, 1, 2])
    groups = [x.replace("-counts", "") for x in counts.columns if "-counts" in x] + [
        "all"
    ]

    # Rename to be RGB values
    vectors.columns = ["R", "B", "G"]

    # Assume each color needs a width of 5 pixels, how many rows do we need?
    rows = math.ceil(vectors.shape[0] * color_width / width)
    height = row_height * rows
    colors_per_row = math.floor(vectors.shape[0] / rows)

    print("Generating data for d3...")
    coords = pandas.DataFrame(index=vectors.index, columns=["x_center", "y_center"])

    color_index = 0
    for row in range(rows + 1):

        # If we're at the last row, likely not same length
        if row == rows:
            names = coords.index[color_index:].tolist()
        else:
            names = coords.index[color_index : color_index + colors_per_row].tolist()
        coords.loc[names, "x_center"] = [row_height * row + 1] * len(names)
        coords.loc[names, "y_center"] = list(
            range(0, color_width * len(names), color_width)
        )
        color_index += colors_per_row

    # Combine matrices to save to file
    vectors["x_center"] = coords.loc[:, "x_center"]
    vectors["y_center"] = coords.loc[:, "y_center"]
    vectors["name"] = vectors.index

    for column in counts.columns:
        vectors[column] = counts.loc[:, column]

    print("Saving data to file...")
    if not outdir:
        outdir = tempfile.mkdtemp()
    with open(os.path.join(outdir, "data.json"), "w") as filey:
        savedata = {
            "records": vectors.to_dict(orient="records"),
            "groups": groups,
            "width": width,
            "color_width": color_width,
            "row_height": row_height,
            "colors_per_row": colors_per_row,
        }
        filey.writelines(json.dumps(savedata, indent=4))

    # Copy the static file there
    template_file = get_static("interactive-grid.html")
    shutil.copyfile(template_file, os.path.join(outdir, "index.html"))
    print("Interactive grid report generated in %s" % outdir)
    return outdir


def save_vectors_gradient_grid(
    outfile,
    vectors,
    alphas=None,
    width=3000,
    row_height=20,
    color_width=80,
    font_size=10,
):
    """Given a vectors data frame (or just a group/extension), save a vectors
       gradient grid to an output image. We draw the word (text) on
       each section. This function produces a mapping of a codebase to 
       colors. Optionally provide a vectors of alphas to determine 
       transparency of each color.
    """
    # Sort by all three columns
    vectors = vectors.sort_values(by=[0, 1, 2])

    # Assume each color needs a width of 5 pixels, how many rows do we need?
    rows = math.ceil(vectors.shape[0] * color_width / width)
    height = row_height * rows

    # Create a new image
    image = Image.new("RGBA", (width, height), (255, 0, 0, 0))
    pixels = image.load()
    colors_per_row = math.floor(vectors.shape[0] / rows)

    # We will separately draw text (Open Sans Font is default)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(get_font(), font_size)

    # Print each color to its row
    x = 0
    color_index = 0
    for row in range(rows):
        y = 0
        xcoords = range(x, x + row_height)
        names = vectors.index[color_index : color_index + colors_per_row]
        for color in names:
            if color:
                rgb = vectors.loc[color].tolist()

                alpha = 255

                # if an alpha vector is provided, use it
                if alphas:
                    alpha = alphas[row]

                for xcoord in xcoords:
                    for ycoord in range(y, y + color_width):
                        pixels[ycoord, xcoord] = (*rgb, alpha)

                draw.text(
                    (y + 2, xcoords[0] + 2), color, (255, 255, 255, alpha), font=font
                )
            y += color_width
        x += row_height
        color_index += colors_per_row

    image.save(outfile)
    return image
