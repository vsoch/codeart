"""

Copyright (C) 2019-2020 Vanessa Sochat.

This Source Code Form is subject to the terms of the
Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

Modified from https://github.com/Visual-mov/Colorful-Julia (MIT License)

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

    def __init__(self, extensions=None):
        self.codefiles = dict()
        self.models = dict()
        self.extensions = extensions

    def __str__(self):
        return "[codeart-base]"

    def __repr__(self):
        return self.__str__()

    def add_folder(self, folder):
        """a courtesy function for the user to add a folder, gives to add_repo.
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
                self.add_file(os.sep.join([dirpath, filename]))

        return folder

    # Load and Save

    def save_model(self, outdir, extension):
        """Save a model particular model, named by the extension
        """
        if not os.path.exists(outdir):
            sys.exit("%s does not exist" % outdir)

        if extension in self.models:
            model = self.models[extension]
            outfile = os.path.join(outdir, "model%s.word2vec" % extension)
            print("Saving %s" % outfile)
            model.save(outfile)

    def save_filelist(self, outdir, extension):
        """Save one file list.
        """
        if not os.path.exists(outdir):
            sys.exit("%s does not exist." % outdir)

        outfile = os.path.join(outdir, "files%s.txt" % ext)
        with open(outfile, "w") as filey:
            filey.writelines(files.files)

    def save_all(self, outdir, extensions=None):
        """Given one or more extensions, save file lists and models to file.
           If no list of extensions is provided, we use models as base.
        """
        if not extensions:
            extensions = list(self.models.keys())

        for ext in extensions:
            self.save_model(outdir, ext)
            self.save_filelist(outdir, ext)

    def load_models(self, indir, extension=".word2vec"):
        """Load models, named by the extension (for direct path, use load_model)
        """
        for filename in os.listdir(indir):
            if filename.endswith(extension):
                model = Word2Vec.load(filename)
                ext, rest = filename.split("model", 1)
                ext = filename.split("model", 1)[-1].replace(extension, "")
                self.models[ext] = model

    def load_model(self, filename, ext):
        """Load a specific model and extension (or grouping pattern)
        """
        self.models[ext] = Word2Vec.load(filename)

    def add_repo(self, repo, branch="master"):
        """Download a repository to a temporary directory to add as a codebase or
           Take a branch name  (defaults to master). We don't use external 
           libraries (e.g., gitpython)  but just execute the command to the system.
        """
        tmpdir = next(tempfile._get_candidate_names())
        tmpdir = os.path.join(tempfile.gettempdir(), tmpdir)
        os.system("git clone -b %s %s %s" % (branch, repo, tmpdir))
        return self.add_folder(tmpdir)

    def threshold_files(self, thresh=100):
        """Iterate over codefiles, and return a dict subset of 
           only those greater than threshold
        """
        passing = dict()
        for ext, codefiles in self.codefiles.items():
            if len(codefiles.files) >= thresh:
                passing[ext] = codefiles
        return passing

    def train(self, extensions=None, size=3, workers=4, min_count=40, thresh=100):
        """train word2vec models for given code files. If None defined, train
           all models. We train with size 3 (vector length) to eventually
           produce RGB values.
        """
        if extensions is None:
            extensions = list(self.codefiles.keys())

        for extension in extensions:
            files = self.codefiles[extension]
            if len(files.files) < thresh:
                continue

            print("Training model for extension %s" % extension)
            model = Word2Vec(files, size=size, workers=workers, min_count=min_count)
            self.models[extension] = model

    def train_all(self, extensions=None, size=3, workers=4, min_count=40):
        """train a single word2vec model for all extensions. Useful to plot
           code base in same color space.
        """
        if extensions is None:
            extensions = list(self.codefiles.keys())

        # Train model with the first extension
        files = [self.codefiles[ext] for ext in extensions]
        files = chain(*files)

        print("Training model with extensions %s" % "|".join(extensions))
        model = Word2Vec(files, size=size, workers=workers, min_count=min_count)
        self.models["all"] = model

    ## Vectors

    def get_vectors(self, extension, rescale_rgb=True):
        """Extract vectors for a model. By default, rescale to be between
           0 and 255 to fit a color space. An extension is required. 
        """
        if extension not in self.models:
            self.train(extensions=[extension])

        model = self.models[extension]
        vectors = pandas.DataFrame(columns=range(model.vector_size))
        for word in model.wv.vocab:
            vectors.loc[word] = self.models[extension].wv.__getitem__(word)

        if rescale_rgb:
            for col in vectors.columns:
                minval = vectors[col].min()
                maxval = vectors[col].max()
                series = ((vectors[col] - minval) / (maxval - minval)) * 255
                vectors[col] = series.astype(int)

        return vectors

    def save_vectors_gradient_grid(
        self,
        extension,
        outfile,
        vectors=None,
        width=3000,
        row_height=20,
        color_width=80,
        font_size=10,
    ):
        """Given a vectors data frame (or just an extension), save a vectors
           gradient grid to an output image. We draw the word (text) on
           each section. This function produces a mapping of a codebase to 
           colors.
        """
        if vectors is None:
            vectors = self.get_vectors(extension)

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
                    for xcoord in xcoords:
                        for ycoord in range(y, y + color_width):
                            pixels[ycoord, xcoord] = (*rgb, 255)

                    draw.text(
                        (y + 2, xcoords[0] + 2), color, (255, 255, 255), font=font
                    )
                y += color_width
            x += row_height
            color_index += colors_per_row

        image.save(outfile)

    def make_gallery(self, extensions=None, bgcolor="white", combine=True):
        """generate a small web directory with a colorful grid that
           shows one or more languages of interest. We do this for each 
           model that is present. If combine is True, we generate the "ALL"
           derived model and then plot the different languages into the 
           same color space.
        """
        template_file = get_static("template.html")
        d3 = get_static("js/d3.v3.js")

        if extensions is None:
            extensions = list(self.models.keys())

        # Create a temporary directory to write files to
        tmpdir = tempfile.mkdtemp(prefix="code-art-")
        image_dir = os.path.join(tmpdir, "images")
        js_dir = os.path.join(tmpdir, "js")

        for dirname in [image_dir, js_dir]:
            os.mkdir(dirname)

        # Copy static js file there
        shutil.copyfile(d3, os.path.join(js_dir, os.path.basename(d3)))

        # If we want to generate across languages, create model first
        vectors = None
        if combine is True:
            if "all" not in self.models:
                self.train_all(extensions)
            vectors = self.get_vectors("all")

        for ext in extensions:

            if ext == "all":
                continue

            # Read in template file
            with open(template_file, "r") as filey:
                template = filey.read()

            print("Generating web output for '%s'" % ext)

            # Generate images for each extension
            images = self.make_art(extension=ext, outdir=image_dir, vectors=vectors)

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
                                "ext": ext,
                                "png": os.path.join("images", os.path.basename(image)),
                            }
                        )
            inputjson["image"] = imagedata

            # Write in data and title
            template = template.replace("{{DATA}}", json.dumps(inputjson, indent=4))
            template = template.replace("{{TITLE}}", "CodeArt for %s" % ext)

            # Generate page to render
            output_html = os.path.join(tmpdir, "codeart%s.html" % ext)
            with open(output_html, "w") as filey:
                filey.writelines(template)

        print("Finished web files are in %s" % tmpdir)
        return tmpdir

    def make_art(self, extension, outdir, vectors=None, files=None):
        """based on an extension, create a set of vectors in the RGB space,
           and then write images for them to some output directory
        """
        images = []
        if extension in self.codefiles:

            # We can take a custom list of files, or the iterator
            files = files or self.codefiles.get(extension, [])

            if hasattr(files, "files"):
                files = files.files

            if vectors is None:
                vectors = self.get_vectors(extension)

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

    def add_file(self, filename, max_bytes=100000):
        """Based on a filename, add the filename to the appropriate code files
           instance depending on the extension. Ignore files in list of skip.
        """
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
            except UnicodeDecodeError:
                pass

    def __str__(self):
        return "[codeart-files:%s]" % len(self.files)

    def __repr__(self):
        return self.__str__()
