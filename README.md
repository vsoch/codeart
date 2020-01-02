# Code Art

[![PyPI version](https://badge.fury.io/py/codeart.svg)](https://pypi.org/project/codeart/)
[![GitHub actions status](https://github.com/vsoch/codeart/workflows/ci/badge.svg?branch=master)](https://github.com/vsoch/codeart/actions?query=branch%3Amaster+workflow%3Aci)

I wanted a way to culminate the last decade of programming that I've done, and realized
it would be much more fun to create a tool that others could use as well. Specifically, I want to be able to:

 1. Have a class that parses over repos on my local machine, and organizes files based on extension..
 2. Builds a word2vec model, one per extension, the idea being that each language has it's own model. The word2vec model should have three dimensions so that we can map it to an RGB colorspace. This will mean that the embeddings for each character, along with being unique for the language, will also have a unique color. We should also be able to build one model across extensions, and map each extension to it.
 3. At the end, I should be able to visualize any particular script (or some other graphic) using these embeddings. I'd like to add in some variable / dimension to represent age / date too.

## Progress

What I have so far is an ability to do the above, and then to produce a tiny image for
each code file, where the words are colored by their word2vec (3 dimensional, RGB) embedding. Here
is a small example of a Python script from spack:

![spack-repos-builtin-packages-hsakmt-package.py.png](https://vsoch.github.io/codeart/examples/spack/images/tmp-mpcq8hau-var-spack-repos-builtin-packages-hsakmt-package.py.png)

And here is a randomly placed grid with all python files from the spack repository:

![examples/spack.png](examples/spack.png)

This is obviously disorganized - I'm working on a way to cluster the individual scripts, and then
plot them, or possibly plot them to show change over time. Note that although these are python
scripts, the embeddings are generated relative to files without an extension (e.g., LICENSE files) 
and .patch files. I suspect as the codebase gets larger (more languages) the color space will
get more interesting.

## Usage

### Install

You can install from pypi

```bash
pip install codeart
```

or install from the repository directly:

```bash
$ git clone https://github.com/vsoch/codeart
$ python setup.py install
```

The following examples are also provided in the [examples](examples) folder.

### Generate a Gallery

You can generate a web gallery for a root folder (including all files beneath it) or
a repository:

```python
from codeart.main import CodeBase                                       
code = CodeBase()                                                       

# How to add a folder
code.add_folder('/home/vanessa/code')  

# How to add a repository
code.add_repo("https://github.com/spack/spack")

# See languages with >100 files
code.threshold_files(100)

# Generate a web report (one page per language above this threshold)
gallery = code.make_gallery(extensions=['', '.py', '.patch']) 
Training model with extensions |.py|.patch
Generating web output for ''
Generating web output for '.py'
Generating web output for '.patch'
Finished web files are in /tmp/code-art-xp73v5ji
```

And then the example files for each of:

 - [python](https://vsoch.github.io/codeart/examples/spack/codeart.py.html)
 - [patch](https://vsoch.github.io/codeart/examples/spack/codeart.patch.html)
 - [empty space](https://vsoch.github.io/codeart/examples/spack/codeart.html)


You can also browse raw images [here](https://github.com/vsoch/codeart/tree/master/docs/examples/spack/images).

### Generate RGB Vectors

If you want to generate RGB vectors for a code base, you can use these
for your own machine learning projects. Here is how to do that
for a repository.

```python
from codeart.main import CodeBase

code = CodeBase()
code.add_repo("https://github.com/spack/spack")
```

The codebase will have codefiles added, a lookup in code.codefiles for each
extension found. You'll want to take a look at these and choose some subset above
a given threshold.

```python
# Look at extractors, one added per extension
code.codefiles

# Find those with >100 files
code.threshold_files(thresh=100)

# {'': [codeart-files:115],
# '.py': [codeart-files:4227],
# '.patch': [codeart-files:531]}
```

And then train a word2vec model, one for each language / extension, with size of 3
so that we can map to the RGB colorspace.

```python
code.train(extensions=['.py', '', '.patch'])
```

You can also train a single model for those extensions

```python
code.train_all(extensions=['.py', '', '.patch'])
```

We now have a model for each extension (and all)

```python
code.models
code.models['all']
```

Here is how to get a panda frame for a particular extension (or all)

```python
vectors = code.get_vectors(".py")
vectors = code.get_vectors("all")
```

### Example 2: Generate a Plot

You might just want to get vectors and plot the RGB space. You can do:

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(vectors[0].tolist(), vectors[1].tolist(), vectors[2].tolist(), c=vectors.to_numpy()/255)

# Optionally add text (a bit crowded)
for row in vectors.iterrows():
    ax.text(row[1][0], row[1][1], row[1][2], row[0])

plt.show()
plt.savefig("spack-python.png")
```

### Example 3: Generate Code Images

If you want to generate raw images for some extension, you can do that too.
These can be used in some image processing project, to generate other images,
or some other kind of clustering analysis.

```python
# Generate images for all files
if not os.path.exists('images'):
    os.mkdir('images')

# Create folder of code images (if you want to work with them directly)
code.make_art(extension=".py", outdir='images', vectors=vectors)
```

### Example 4: Generate a Grid

I haven't developed this in detail, but you can also generate a grid of colors
for a given image. The idea here would be to have a grid that
can be used as a legend. Here we again load in the spack code base,
and generate a lookup for the ".py" (python) extension.

```python
from codeart.main import CodeBase
code = CodeBase()                      
code.add_repo("https://github.com/spack/spack")
code.save_vectors_gradient_grid('.py', 'spack-image-gradient.png') 
```

An example gradient image is the following:

![examples/spack-image-gradient.png](examples/spack-image-gradient.png)

You can of course adjust the dimensions, the row height, and the column width
and font size depending on your needs or matrix size.

Do you have a question? Or want to suggest a feature to make it better?
Please [open an issue!](https://www.github.com/vsoch/codeart)
