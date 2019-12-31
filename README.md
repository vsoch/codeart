# Code Art

[![PyPI version](https://badge.fury.io/py/codeart.svg)](https://pypi.org/project/codeart/)
[![GitHub actions status](https://github.com/vsoch/codeart/workflows/ci/badge.svg?branch=master)](https://github.com/vsoch/codeart/actions?query=branch%3Amaster+workflow%3Aci)

I wanted a way to culminate the last decade of programming that I've done, and realized
it would be much more fun to create a tool that others could use as well. Specifically, I want to be able to:

 1. Have a class that parses over repos on my local machine, and organizes files based on extension. I also don't want to take files that are older than 10 years.
 2. Builds a word2vec model, one per extension, the idea being that each language has it's own model. The word2vec model should have three dimensions so that we can map it to an RGB colorspace. This will mean that the embeddings for each character, along with being unique for the language, will also have a unique color. 
 3. At the end, I should be able to visualize any particular script (or some other graphic) using these embeddings. I'd like to add in some variable / dimension to represent age / date too.

**under development**


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

**being written**

Do you have a question? Or want to suggest a feature to make it better?
Please [open an issue!](https://www.github.com/vsoch/codeart)
