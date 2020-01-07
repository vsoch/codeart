"""

Copyright (C) 2019-2020 Vanessa Sochat.

This Source Code Form is subject to the terms of the
Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

Modified from https://github.com/Visual-mov/Colorful-Julia (MIT License)

"""

from datetime import datetime
import json
import os
import re
import sys

here = os.path.dirname(os.path.abspath(__file__))

## Groups
# Helper Functions to be passed to add_folder, add_repo to determine
# grouping for a file


def group_by_year_created(filename):
    """Given a filename, return the year it was created.
       This function is called after testing the file for read access. 
    """
    stat = os.stat(filename)
    return datetime.fromtimestamp(stat.st_ctime).year


## Find
# More helper functions to return metadata from folders


def recursive_find(folder, return_files=True):
    """Iterate through files and directories recursively
    """
    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if return_files:
                yield os.sep.join([dirpath, filename])
            else:
                yield dirpath


def recursive_find_repos(folder, return_folders=False):
    """Look through folders searching for .git files, if we find one,
       read to extract origin names.
    """
    # Regular expression for repo
    regexp = "((git|ssh|http(s)?)|(git@[\w\.]+))(:(//)?)([\w\.@\:/\-~]+)(\.git)(/)?"
    repos = []

    if return_folders:
        repos = dict()

    for filename in recursive_find(folder):
        if os.sep.join([".git", "config"]) in filename:

            # Must be readable / accessible
            if not os.access(filename, os.R_OK):
                continue

            with open(filename, "r") as filey:
                lines = filey.readlines()

            # Find the most common (origin) and get the url on the next line
            while lines:
                line = lines.pop()
                if "url" in line:
                    match = re.search(regexp, line)
                    if match:

                        # Grab the repository
                        repo = match.group().replace(".git", "")
                        if "github" in repo:

                            # Get rid of any ssl config
                            repo = repo.split(":")[-1]
                            reponame = "https://github.com/%s" % "/".join(
                                repo.split("/")[-2:]
                            )
                            if return_folders:
                                repos[reponame] = filename.replace(
                                    os.path.sep.join([".git", "config"]), ""
                                ).rstrip("/")
                            else:
                                repos.append(reponame)

    return repos


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


def nearest_square_root(number):
    """find the nearest square to a number
    """
    answer = 0
    while (answer + 1) ** 2 < number:
        answer += 1
    return int(answer)


def get_font(filename="OpenSans-Regular.ttf"):
    """Return the default font for writing on the images. A user could
       add additional fonts to this folder, if desired.
    """
    font_file = os.path.join(here, "fonts", filename)
    if not os.path.exists(font_file):
        sys.exit("Font %s does not exist." % font_file)
    return font_file


def get_static(filename):
    """return a file in static, if it exists
    """
    filename = os.path.join(here, "static", filename)
    if not os.path.exists(filename):
        sys.exit("Static file %s does not exist." % filename)
    return filename


def check_restricted(value, min_range, max_range):
    """Ensure that we have a float between a min and max range.
    """
    try:
        value = float(value)
    except ValueError:
        sys.exit("Parameter %s must be a float." % value)

    if value < min_range or value > max_range:
        sys.exit("ca and cb must be in range (-1, 1)")


# Files


def write_json(json_obj, filename, mode="w", print_pretty=True):
    """write_json will (optionally,pretty print) a json object to file

       Parameters
       ==========
       json_obj: the dict to print to json
       filename: the output file to write to
       pretty_print: if True, will use nicer formatting
    """
    with open(filename, mode) as filey:
        if print_pretty:
            filey.writelines(print_json(json_obj))
        else:
            filey.writelines(json.dumps(json_obj))
    return filename


def print_json(json_obj):
    """ just dump the json in a "pretty print" format
    """
    return json.dumps(json_obj, indent=4, separators=(",", ": "))


def read_file(filename, mode="r", readlines=True):
    """write_file will open a file, "filename" and write content, "content"
       and properly close the file
    """
    with open(filename, mode) as filey:
        if readlines is True:
            content = filey.readlines()
        else:
            content = filey.read()
    return content


def read_json(filename, mode="r"):
    """read_json reads in a json file and returns
       the data structure as dict.
    """
    with open(filename, mode) as filey:
        data = json.load(filey)
    return data
