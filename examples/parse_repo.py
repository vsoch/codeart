#!/usr/bin/env python

# This example will show how to build a model for a repository. Specifically,
# we choose a Python repository (a large one!) and then build a model for
# all file extensions with > 100 files. We will show generating raw images
# (data for your use) along with a web gallery for each extension (using
# the same model)

from codeart.main import CodeBase

code = CodeBase()
code.add_repo("https://github.com/spack/spack")

# Look at extractors, one added per extension
# code.codefiles


###
## Example 1: Generate Raw Data
###

# You likely want to chose those above a certain threshold.
# Here we look at extensions with >= 100 files. If you get an error during
# training, it's because the size of the vocabularity isn't big enough due
# to too few files, so choose your extension threshold generously.

code.threshold_files(thresh=100)

# {'': [codeart-files:115],
# '.py': [codeart-files:4227],
# '.patch': [codeart-files:531]}

# Let's train a word2vec model, size 3 for RGB space, for each of those extensions
code.train(extensions=[".py", "", ".patch"])

# We can also train a single model for those extensions
code.train_all(extensions=[".py", "", ".patch"])

# We now have a model for each extension (and all)
code.models
code.models["all"]

# First let's just generate vectors (RGB values across words in each model) for each language
# This is a pandas data frame you can easily save to csv, pickle, etc.
vectors = code.get_vectors(".py")

###
## Example 2: Generate a Plot
###

# Here is an simple visualization example using Matplotlib.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(
    vectors[0].tolist(),
    vectors[1].tolist(),
    vectors[2].tolist(),
    c=vectors.to_numpy() / 255,
)

for row in vectors.iterrows():
    ax.text(row[1][0], row[1][1], row[1][2], row[0])

plt.show()
plt.savefig("spack-python.png")

###
## Example 3: Generate Code Images
###

# Generate images for all files
if not os.path.exists("images"):
    os.mkdir("images")

# Create folder of code images (if you want to work with them directly)
code.make_art(extension=".py", outdir="images", vectors=vectors)


###
## Example 4: Generate A Gallery
###

# The gallery example here will plot each language, using the same model
gallery = code.make_gallery(extensions=["", ".py", ".patch"])
