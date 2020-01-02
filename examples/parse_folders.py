#!/usr/bin/env python

from codeart.main import CodeBase

# Sitting where you want root of analysis to be
root = os.getcwd()
code = CodeBase()

# Parse each subfolder of a root directory
for folder in os.listdir(root):
    code.add_folder(folder)

# Look at extractors, one added per extension
code.codefiles

# Look at extensions that have > 100 files
code.threshold_files(100)

# Select a subset of extensions to build one model
extensions=[
 '.py',
 '.txt',
 '.sh',
 '',
 '.md',
 '.r',
 '.tab',
 '.out',
 '.js',
 '.html',
 '.css',
 '.yml',
 '.svg',
 '.rst',
 '.in',
 '.err'
]

# train single model for some subset of extensions (could also choose thresh=100)
code.train_all(extensions)

# extract vectors for an extension (pandas dataframe, words in rows)
# normalized to RGB color space
vectors = code.get_vectors('all')

# Save vectors gradients to tsv
vectors.to_csv("vectors-gradients.tsv", sep="\t")

# Make a gallery (use custom vectors)
gallery = code.make_gallery(extensions=extensions, vectors=vectors)

# Or generate a vector gradient
code.save_vectors_gradient_grid('all', 'colors-gradient.png') 
