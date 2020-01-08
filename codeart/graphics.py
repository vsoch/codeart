"""

Copyright (C) 2019-2020 Vanessa Sochat.

This Source Code Form is subject to the terms of the
Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

Modified from https://github.com/Visual-mov/Colorful-Julia (MIT License)

"""

from PIL import Image, ImageFont, ImageDraw
from codeart.utils import get_static, get_font, nearest_square_root

from .namer import RobotNamer
import json
import math
import numpy
import operator
import os
import pandas
import random
import shutil
import sys
import tempfile


def generate_codeart(
    template, color_lookup, top=20, sample=15, bgcolor="white", outfile="codeart.html"
):
    """generate codeart will take a template image and generate a web interface
       for the same image (plotted with the images from the color lookup) 
       using d3. It's expected to generate with paths relative to the images.
    """
    base = Image.open(template)
    width, height = base.size
    pixels = base.load()

    count = 0
    new_image = pandas.DataFrame(columns=["x", "y", "corr", "png"])

    for x in range(width):
        for y in range(height):

            # And take only every [sample]th pixel
            if x % sample == 0 and y % sample == 0:
                cpixel = pixels[x, y]
                tmp = color_lookup.copy()
                tmp = (tmp - cpixel).abs().sum(axis=1)

                # We don't take the exact match, but rather some distance from the top
                png = random.choice(tmp.loc[tmp.index[0:top]].index.tolist())
                new_image.loc[count] = [x, y, 0, png]
                count += 1

    new_image["x"] = [int(x) for x in (new_image["x"] / sample) * 10]
    new_image["y"] = [int(x) for x in (new_image["y"] / sample) * 10]

    with open(get_static("codeart.html"), "r") as filey:
        template = filey.read()

    # Save output to file
    records = new_image.to_dict(orient="records")
    data = {"image": records, "bgcolor": bgcolor}
    template = template.replace("{{DATA}}", json.dumps(data))

    with open(outfile, "w") as filey:
        filey.writelines(template)

    return new_image


def generate_codeart_text(
    text,
    color_lookup,
    top=20,
    bgcolor="white",
    outfile="codert-text.html",
    font_size=50,
    width=600,
    height=600,
    coords=None,
):
    image = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(get_font(), font_size)

    if not coords:
        coords = (50, 20)

    # Draw the text onto the canvas
    draw.text(coords, text, font=font)

    # use coordinates to draw text to data
    pixels = image.load()

    count = 0
    new_image = pandas.DataFrame(columns=["x", "y", "corr", "png"])

    for x in range(width):
        for y in range(height):
            cpixel = pixels[x, y]
            if sum(cpixel) != 0:
                # We don't take the exact match, but rather some distance from the top
                png = random.choice(color_lookup.index.tolist())
                new_image.loc[count] = [x, y, 0, png]
                count += 1

    with open(get_static("codeart.html"), "r") as filey:
        template = filey.read()

    # Save output to file
    records = new_image.to_dict(orient="records")
    data = {"image": records, "bgcolor": bgcolor}
    template = template.replace("{{DATA}}", json.dumps(data))

    with open(outfile, "w") as filey:
        filey.writelines(template)

    return new_image


def generate_colored_image(
    image,
    vectors,
    counts=None,
    rcol="R",
    gcol="G",
    bcol="B",
    maxwidth=600,
    maxheight=600,
):
    """This function will take an input image and color vectors,
       and generates a version of the image mapped to the color space of
       the code. If colors are provided, generate the same image for each
       grouping. This is in spirit of the brainart library that I created
       in graduate school https://github.com/vsoch/brainart.
    """
    # Read in the image
    if not os.path.exists(image):
        sys.exit("%s does not exist." % image)

    # Color lookup
    color_lookup = vectors[[rcol, gcol, bcol]]

    # Read in the original image
    base = Image.open(image)
    width, height = base.size

    # Resize to smaller
    scale = min(250 / width, 250 / height)
    base.thumbnail((width * scale, height * scale), Image.ANTIALIAS)
    width, height = base.size
    pixels = base.load()

    print("Generating new pixels!")
    for x in range(width):
        for y in range(height):

            rgb_pixel = pixels[x, y]

            # Create a temporary copy to calculate the closest
            tmp = color_lookup.copy()
            tmp = (tmp - rgb_pixel).abs().sum(axis=1)

            # Find smallest distance
            mins = tmp[tmp == tmp.min()]
            term = random.choice(mins.index)

            pixels[x, y] = tuple(color_lookup.loc[term, [rcol, gcol, bcol]].tolist())

    if outfile is None:
        outfile = "%s.png" % RobotNamer().generate()
    base.save(outfile, "PNG")


def generate_interactive_colormap(
    vectors, counts, color_width=20, row_height=20, outdir=None
):
    """Based on a set of vectors, generate an interactive colormap.
       with d3. This function should take output from get_vectors and
       get_color_percentages.
    """
    # Create a lookup for RGB values, because pandas sucks
    lookup = dict()
    for row in vectors.iterrows():
        lookup["%s-%s-%s" % (row[1][0], row[1][1], row[1][2])] = row[0]

    # Supporting functions
    def row_sort(grid, key):
        new_grid = []
        for i in range(grid.shape[0]):
            new_grid.append(sorted(grid[i], key=key))
        return numpy.array(new_grid)

    def column_sort(grid, key):
        return numpy.swapaxes(row_sort(numpy.swapaxes(grid, 0, 1), key), 0, 1)

    def row_color_sort(grid):
        return row_sort(grid, lambda x: sum(x))

    def hue(rgb):
        rgb_max = max(rgb)
        rgb_min = min(rgb)
        red, green, blue = rgb
        if rgb[0] == rgb_max:
            return (green - blue) / (rgb_max - rgb_min)
        elif rgb[1] == rgb_max:
            return 2.0 + (blue - red) / (rgb_max - rgb_min)
        else:
            return 4.0 + (red - green) / (rgb_max - rgb_min)

    def col_color_sort(grid):
        return column_sort(grid, lambda x: hue(x))

    # Reshape into a square grid with extra spaces as white
    dim = nearest_square_root(vectors.shape[0]) + 1

    extra_space = (dim * dim) - vectors.shape[0]
    data = numpy.vstack(
        (vectors.to_numpy(), numpy.zeros((extra_space, 3), dtype=numpy.int8))
    )

    # Reshape to grid
    data = numpy.reshape(data, (dim, dim, 3))

    # sort by columns and rows
    for _ in range(4):
        data = col_color_sort(data)
        data = row_color_sort(data)

    groups = [x.replace("-counts", "") for x in counts.columns if "-counts" in x] + [
        "all"
    ]
    vectors.columns = ["R", "B", "G"]

    print("Generating data for d3...")
    records = []
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            r, g, b = data[x, y, :]
            rgb_index = "%s-%s-%s" % (r, g, b)
            if rgb_index in lookup:
                name = lookup[rgb_index]
                record = {
                    "R": int(r),
                    "G": int(g),
                    "B": int(b),
                    "name": name,
                    "x_center": int(x),
                    "y_center": int(y),
                }
                for column in counts.columns:
                    if column.endswith("percent"):
                        record[column] = float(counts.loc[name, column])
                    else:
                        record[column] = int(counts.loc[name, column])

                records.append(record)

    print("Saving data to file...")
    if not outdir:
        outdir = tempfile.mkdtemp()
    with open(os.path.join(outdir, "data.json"), "w") as filey:
        savedata = {
            "records": records,
            "groups": groups,
            "rows": int(data.shape[1]),
            "color_width": color_width,
            "row_height": row_height,
            "colors_per_row": int(data.shape[0]),
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
