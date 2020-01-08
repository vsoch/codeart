"""

Copyright (C) 2019-2020 Vanessa Sochat.

This Source Code Form is subject to the terms of the
Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""

from codeart.nlp import text2sentences, sentence2words
from codeart.utils import get_static, nearest_square_root, get_font
from itertools import chain
from PIL import Image, ImageFont, ImageDraw
from gensim.models import Word2Vec
from glob import glob

import json
import math
import os
import pandas
import shutil
import re
import sys
import tempfile


class CodeBase(object):
    """A CodeBase can hold multiple extractors, meaning that each corresponds
       to a language (determined by script extension) and each can be used to
       build it's own word2vec model. Optionally, a list of extensions can
       be provided to limit to. If not defined, we look at all extensions.
    """

    def __init__(self, groups=None):
        self.codefiles = dict()
        self.models = dict()
        self.extensions = groups

    def __str__(self):
        return "[codeart-base]"

    def __repr__(self):
        return self.__str__()

    def get_groups(self):
        """Return names of groups of codefiles
        """
        return list(self.codefiles.keys())

    def add_folder(self, folder, func=None, group=None):
        """a courtesy function for the user to add a folder, gives to add_repo.
           optionally add a custom function to filter the file and assign
           to a codefiles group, OR to add a custom group name (group) 
           to assign to.
        """
        if not os.path.exists(folder):
            sys.exit("%s does not exist." % folder)

        skips = [".git", ".eggs", "build", "dist"]
        skip_re = "(%s)" % "|".join(["/%s/" % x for x in skips])
        end_re = "(%s)" % "|".join(["%s$" % x for x in skips])
        for dirpath, dirnames, filenames in os.walk(folder):
            if re.search(skip_re, dirpath) or re.search(end_re, dirpath):
                continue
            for filename in filenames:
                self.add_file(os.sep.join([dirpath, filename]), func=func, group=group)

        return folder

    # Load and Save and Get

    def save_model(self, outdir, group):
        """Save a model particular model, named by the group
        """
        if not os.path.exists(outdir):
            sys.exit("%s does not exist" % outdir)

        if group in self.models:
            model = self.models[group]
            outfile = os.path.join(outdir, "model%s.word2vec" % group)
            print("Saving %s" % outfile)
            model.save(outfile)

    def save_filelist(self, outdir, group):
        """Save one file list.
        """
        if not os.path.exists(outdir):
            sys.exit("%s does not exist." % outdir)

        outfile = os.path.join(outdir, "files%s.txt" % group)

        # Saving all files versus just for a group
        if group == "all":
            files = self.get_allfiles()
        else:
            files = self.codefiles[group].files
        with open(outfile, "w") as filey:
            for filename in files:
                filey.writelines(filename)

    def save_all(self, outdir, groups=None):
        """Given one or more groups (extensions or other), save file lists and 
           models to file. If no list of extensions is provided, we use 
           models as base.
        """
        if not groups:
            groups = list(self.models.keys())

        for group in groups:
            self.save_model(outdir, group)
            self.save_filelist(outdir, group)

    def load_models(self, indir, extension=".word2vec"):
        """Load models, named by the extension (for direct path, use load_model)
        """
        for filename in os.listdir(indir):
            if filename.endswith(extension):
                model = Word2Vec.load(filename)
                group = filename.split("model", 1)[-1].replace(extension, "")
                self.models[group] = model

    def load_model(self, filename, group):
        """Load a specific model and group/extension (or grouping pattern)
        """
        self.models[group] = Word2Vec.load(filename)

    def get_allfiles(self, groups=None):
        """Return an iterator with all files, or all files for some listing of
           groups
        """
        if groups is None:
            groups = list(self.codefiles.keys())

        files = []
        for group in groups:
            files = files + self.codefiles[group].files
        return files

    def add_repo(self, repo, branch="master", func=None, group=None):
        """Download a repository to a temporary directory to add as a codebase or
           Take a branch name  (defaults to master). We don't use external 
           libraries (e.g., gitpython)  but just execute the command to the system.
           Optionally provided a helper function to custom group the files.
        """
        tmpdir = next(tempfile._get_candidate_names())
        tmpdir = os.path.join(tempfile.gettempdir(), tmpdir)
        os.system("git clone -b %s %s %s" % (branch, repo, tmpdir))
        return self.add_folder(tmpdir, func=func, group=group)

    def threshold_files(self, thresh=100):
        """Iterate over codefiles, and return a dict subset of 
           only those greater than threshold
        """
        passing = dict()
        for group, codefiles in self.codefiles.items():
            if len(codefiles.files) >= thresh:
                passing[group] = codefiles
        return passing

    def train(self, groups=None, size=3, workers=4, min_count=40, thresh=100):
        """train word2vec models for given code files. If None defined, train
           all models. We train with size 3 (vector length) to eventually
           produce RGB values.
        """
        if groups is None:
            groups = list(self.codefiles.keys())

        for group in groups:
            if group == "all":
                self.train_all()
                continue

            files = self.codefiles[group]
            if len(files.files) < thresh:
                continue

            print("Training model for group %s" % group)
            model = Word2Vec(files, size=size, workers=workers, min_count=min_count)
            self.models[group] = model

    def train_all(self, groups=None, size=3, workers=4, min_count=40):
        """train a single word2vec model for all groups/extensions. Useful to plot
           code base in same color space.
        """
        if groups is None:
            groups = list(self.codefiles.keys())

        # Train model with the first extension
        files = [self.codefiles[g] for g in groups]
        files = chain(*files)

        print("Training model with groups %s" % "|".join(groups))
        model = Word2Vec(files, size=size, workers=workers, min_count=min_count)
        self.models["all"] = model

    ## Vectors

    def get_color_percentages(self, groups, vectors):
        """Given a data frame of vectors and a group, parse a code base
           and determine the relative prevalence of a term. For example,
           if we find 200 instances of a term in the vectors lookup,
           we might then parse individual groups and find that it appears
           10 times in Python. The overall score for Python and that term
           would be 10/200 or 1/20 or 0.05, and this would correspond to an
           opacity value returned.
        """
        totals = {word: 0 for word in vectors.index}
        counts = {group: {word: 0 for word in vectors.index} for group in groups}

        # First derive total counts for each term
        print("Deriving counts... please wait!")
        for group in groups:
            print("Processing group %s." % group)
            files = self.codefiles[group]
            for words in files:
                for word in words:
                    if word in totals:
                        totals[word] += 1
                        counts[group][word] += 1

        # Put all into data frame
        columns = ["%s-counts" % g for g in groups] + ["%s-percent" % g for g in groups]
        df = pandas.DataFrame(columns=["totals"] + columns)
        df["totals"] = pandas.DataFrame.from_dict(totals, orient="index")[0]

        print("Generating data frame...")
        for group in groups:
            print("Updating %s" % group)
            df.loc[:, "%s-counts" % group] = 0

            for word in totals:
                df.loc[word, "%s-counts" % group] = counts[group][word]
                if totals[word] != 0:
                    df.loc[word, "%s-percent" % group] = (
                        counts[group][word] / totals[word]
                    )
                else:
                    df.loc[word, "%s-percent" % group] = 0

        return df

    def get_vectors(self, group, rescale_rgb=True):
        """Extract vectors for a model. By default, rescale to be between
           0 and 255 to fit a color space. An extension is required. 
        """
        if group not in self.models:
            self.train(groups=[group])

        model = self.models[group]
        vectors = pandas.DataFrame(columns=range(model.vector_size))
        for word in model.wv.vocab:
            vectors.loc[word] = self.models[group].wv.__getitem__(word)

        if rescale_rgb:
            for col in vectors.columns:
                minval = vectors[col].min()
                maxval = vectors[col].max()
                series = ((vectors[col] - minval) / (maxval - minval)) * 255
                vectors[col] = series.astype(int)

        return vectors

    def make_gallery(
        self, groups=None, bgcolor="white", combine=True, outdir=None, images=None
    ):
        """generate a small web directory with a colorful grid that
           shows one or more languages of interest. We do this for each 
           model that is present. If combine is True, we generate the "ALL"
           derived model and then plot the different languages into the 
           same color space. If images is not None, should be a listing of
           images in a relative "images" folder.
        """
        template_file = get_static("template.html")
        d3 = get_static("js/d3.v3.js")

        if groups is None:
            groups = list(self.models.keys())

        # Create a temporary directory to write files to
        if outdir is None:
            outdir = tempfile.mkdtemp(prefix="code-art-")

        image_dir = os.path.join(outdir, "images")
        js_dir = os.path.join(outdir, "js")

        for dirname in [image_dir, js_dir]:
            if not os.path.exists(dirname):
                os.mkdir(dirname)

        # Copy static js file there
        shutil.copyfile(d3, os.path.join(js_dir, os.path.basename(d3)))

        # If we want to generate across languages, create model first
        vectors = None
        if combine is True:
            if "all" not in self.models:
                self.train_all(groups)
            vectors = self.get_vectors("all")

        for group in groups:

            # Read in template file
            with open(template_file, "r") as filey:
                template = filey.read()

            print("Generating web output for '%s'" % group)

            # Generate images for each group or extension
            if images is None:
                images = self.make_art(group=group, outdir=image_dir, vectors=vectors)

            # Generate data for images
            imagedata = []
            inputjson = {"bgcolor": bgcolor}

            width = height = nearest_square_root(len(images))

            for x in range(0, width):
                for y in range(0, height):
                    if images:
                        image = images.pop()
                        imagedata.append(
                            {
                                "y": y,
                                "x": x,
                                "ext": group,
                                "png": os.path.join("images", os.path.basename(image)),
                            }
                        )
            inputjson["image"] = imagedata

            # Write in data and title
            template = template.replace("{{DATA}}", json.dumps(inputjson, indent=4))
            template = template.replace("{{TITLE}}", "CodeArt for %s" % group)

            # Generate page to render
            output_html = os.path.join(outdir, "codeart%s.html" % group)
            with open(output_html, "w") as filey:
                filey.writelines(template)

        print("Finished web files are in %s" % outdir)
        return outdir

    def make_tree(self, group, vectors, outdir=None, outfile="tree.html"):
        """Given colors vectors, generate images that are used for a filesystem tree
        """
        if outdir is None:
            outdir = tempfile.mkdtemp(prefix="code-art-tree-")
        image_dir = os.path.join(outdir, "images")

        # Create output image directory
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

        if group == "all":
            files = self.get_allfiles()
        elif group in self.codefiles:
            files = self.codefiles[group].files
        else:
            sys.exit("%s is not a known group." % group)

        # Generate images for each group or extension
        print("Generating images for tree...")
        images = self.make_art(group=group, outdir=image_dir, vectors=vectors)

        # Find the highest level shared folder (will remove from tree)
        print("Generating tree...")
        level = min([x.count("/") for x in files])
        root = "/".join(files[0].split("/")[:level])

        # Create a list of nodes, count holds an id for nodes
        nodes = {}
        lookup = {}
        count = 1
        max_depth = 0

        for filename in files:
            relative_path = filename.replace(root, "")
            path_components = relative_path.split(os.sep)
            for p in range(len(path_components)):
                path_component = path_components[p]
                fullpath = os.sep.join(path_components[0 : p + 1])

                # Have we created the node yet?
                if fullpath not in lookup:
                    lookup[fullpath] = count
                    png = "%s.png" % os.path.join(
                        "images", filename.replace(os.sep, "-").strip("-")
                    )
                    node = {
                        "id": count,
                        "name": path_component,
                        "path": fullpath,
                        "png": png,
                        "level": p,
                        "children": [],
                    }
                    count += 1
                    # Did we find a deeper level?
                    if p > max_depth:
                        max_depth = p
                    # Does the node have a parent?
                    if p == 0:  # base node, no parent
                        parent_id = 0
                    else:  # look up the parent id
                        parent_path = os.sep.join(path_components[0:p])
                        parent_id = lookup[parent_path]
                    node["parent"] = parent_id
                    nodes[node["id"]] = node

        # Now make the graph, we simply append children to their parents
        seen = []
        graph = []
        iters = list(range(max_depth + 1))  # 0,1,2,3...
        iters.reverse()  # ...3,2,1,0
        iters.pop()  # remove 0
        for level in iters:
            children = {x: y for x, y in nodes.items() if y["level"] == level}
            seen = seen + [y["id"] for x, y in children.items()]
            nodes = {x: y for x, y in nodes.items() if y["id"] not in seen}
            for node_id, child_node in children.items():
                if node_id == 0:  # base node
                    graph[node_id] = child_node
                else:
                    parent_id = child_node["parent"]
                    nodes[parent_id]["children"].append(child_node)

        # Now add the parents to graph, with name as main lookup
        for parent, parent_info in nodes.items():
            graph.append(parent_info)
        graph = {"name": "base", "children": graph}
        result = {"graph": graph, "lookup": lookup, "depth": max_depth + 1}

        # Get the template
        template = get_static("tree.html")
        data_file = os.path.join(outdir, "data.json")
        with open(data_file, "w") as filey:
            filey.writelines(json.dumps(result, indent=4))

        shutil.copyfile(template, os.path.join(outdir, outfile))
        return result

    def make_art(self, group, outdir, vectors=None, files=None):
        """based on an extension, create a set of vectors in the RGB space,
           and then write images for them to some output directory
        """
        images = []

        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if files is None:
            if group == "all":
                files = self.get_allfiles()
            elif group in self.codefiles:
                files = self.codefiles[group].files
            else:
                sys.exit("%s is not found in codefiles." % group)

        if hasattr(files, "files"):
            files = files.files

        if vectors is None:
            vectors = self.get_vectors(group)

        # Parse through files
        for filename in files:
            with open(filename, "rb") as fp:
                content = fp.read().decode("utf8", "ignore")

            # get rid of windows newline, replace tab with 4 spaces, lines
            lines = content.replace("\r", "").replace("\t", "    ").split("\n")

            # The max width is the final width
            width = max([len(line) for line in lines])
            height = len(lines)

            if width == 0 or height == 0:
                continue

            # Create image of this size
            image = Image.new("RGBA", (width, height), (255, 0, 0, 0))
            pixels = image.load()

            for l, line in enumerate(lines):
                padded = line + (width - len(line)) * " "

                # Find the coordinates for each word
                for word in sentence2words(padded):

                    if word not in vectors.index:
                        continue

                    rgb = vectors.loc[word].tolist()
                    for match in re.finditer(word, padded):
                        for y in range(match.start(), match.end()):
                            pixels[y, l] = (*rgb, 255)

            # Save output file
            outfile = "%s.png" % os.path.join(
                outdir, filename.replace(os.path.sep, "-").strip("-")
            )
            image.save(outfile)
            images.append(outfile)

        return images

    def add_file(self, filename, max_bytes=100000, func=None, group=None):
        """Based on a filename, add the filename to the appropriate code files
           instance depending on the extension. Ignore files in list of skip.
           Optionally add a function to determine assignment to a group, OR
           the group name directly
        """
        # Must be readable / accessible
        if not os.access(filename, os.R_OK):
            return

        size = os.path.getsize(filename)
        name, ext = os.path.splitext(os.path.basename(filename))

        # Generally skip image and extension files
        skip_ext = [
            ".exe",
            ".gif",
            ".gz",
            ".img",
            ".jpg",
            ".jpeg",
            ".png",
            ".pdf",
            ".pkl",
            ".pyc",
            ".sif",
            ".simg",
            ".zip",
        ]
        ext = ext.lower()

        # Skip based on extension, temporary, or file is too big
        if ext in skip_ext or size >= max_bytes or ext.endswith("~"):
            return

        # If extensions are provided, ensure parsed is in list
        if self.extensions is not None:
            if ext not in self.extensions:
                return

        # If a group is provided, use that
        if group is not None:
            ext = group

        # Otherwise if a function is provided, run it to derive the group
        elif func is not None:
            ext = func(filename)

        # A return of None / False indicates that we don't add
        if ext:
            if ext not in self.codefiles:
                self.codefiles[ext] = CodeFiles()
            self.codefiles[ext].files.append(filename)


class CodeFiles(object):
    """A code art files object holds a list of files and a model built from them.
       The CodeBase instance is responsible for keeping track of extensions (or
       some logical grouping) of hte files for each model, and also training,
       etc.
    """

    def __init__(self):
        """each extractor holds a list of files with a single extension
        """
        self.files = []

    def __iter__(self):
        """When used as an iterator, we iterate over files to yield words
        """
        # Iterating over a list of file paths
        for input_file in self.files:
            try:
                for text in open(input_file, "r").readlines():
                    for line in text2sentences(text):
                        words = sentence2words(line)
                        if len(words) < 3:
                            continue
                        yield words
            except FileNotFoundError:
                print("Input file was not found. Was it moved?" % input_file)
            except UnicodeDecodeError:
                pass

    def __str__(self):
        return "[codeart-files:%s]" % len(self.files)

    def __repr__(self):
        return self.__str__()
