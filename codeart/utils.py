"""

Copyright (C) 2019-2020 Vanessa Sochat.

This Source Code Form is subject to the terms of the
Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

Modified from https://github.com/Visual-mov/Colorful-Julia (MIT License)

"""

import os
import sys

here = os.path.dirname(os.path.abspath(__file__))


def download_nltk():
    """download nltk to home
    """
    home = os.environ["HOME"]
    download_dir = os.path.join(home, "nltk_data")
    print("Downloading nltk to %s" % (download_dir))
    if not os.path.exists(download_dir):
        import nltk

        nltk.download("all")
    return os.path.join(home, "nltk_data")


def get_font(filename="OpenSans-Regular.ttf"):
    """Return the default font for writing on the images. A user could
       add additional fonts to this folder, if desired.
    """
    font_file = os.path.join(here, "fonts", filename)
    if not os.path.exists(font_file):
        sys.exit("Font %s does not exist." % font_file)
    return font_file


def check_restricted(value, min_range, max_range):
    """Ensure that we have a float between a min and max range.
    """
    try:
        value = float(value)
    except ValueError:
        sys.exit("Parameter %s must be a float." % value)

    if value < min_range or value > max_range:
        sys.exit("ca and cb must be in range (-1, 1)")
