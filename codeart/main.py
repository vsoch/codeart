"""

Copyright (C) 2019-2020 Vanessa Sochat.

This Source Code Form is subject to the terms of the
Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

Modified from https://github.com/Visual-mov/Colorful-Julia (MIT License)

"""

from codeart.nlp import text2sentences, sentence2words, find_phrases
from PIL import Image
from gensim.models import Word2Vec
from glob import glob
import numpy
import os
import pandas
import sys

from math import sqrt
import re
import os
import shutil
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
        tmpdir = os.path.join(tempfile.tempdir, tmpdir)
        os.system("git clone -b %s %s %s" % (branch, repo, tmpdir))
        return self.add_folder(tmpdir)

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

    def get_vectors(self, extension, rescale_rgb=True):
        """Extract vectors for a model. By default, rescale to be between
           0 and 255 to fit a color space. An extension is required. 
        """
        if extension in self.models:
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

        print("%s is not a valid model, did you train() yet?" % extension)

    def make_art(self, extension, outdir, vectors=None, files=None):
        """based on an extension, create a set of vectors in the RGB space,
           and then write images for them to some output directory
        """
        if extension in self.models:

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
                            for y in range(match.start(), match.end() + 1):
                                pixels[y, l] = (*rgb, 255)

                # Save output file
                outfile = "%s.png" % os.path.join(
                    outdir, filename.replace(os.path.sep, "-")
                )
                image.save(outfile)

        print("%s is not a valid model, did you train() yet?" % extension)

    def add_file(self, filename):
        """Based on a filename, add the filename to the appropriate code files
           instance depending on the extension. Ignore files in list of skip.
        """
        name, ext = os.path.splitext(os.path.basename(filename))

        # Generally skip image and extension files
        skip_ext = [".gz", ".exe", ".sif", "simg", "img", "png", "gif", "jpg", "jpeg"]
        ext = ext.lower()

        if ext in skip_ext:
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
            for text in open(input_file, "r").readlines():
                for line in text2sentences(text):
                    words = sentence2words(line)
                    if len(words) < 3:
                        continue
                    yield words

    def __str__(self):
        return "[codeart-files:%s]" % len(self.files)

    def __repr__(self):
        return self.__str__()
