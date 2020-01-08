"""

Copyright (C) 2019-2020 Vanessa Sochat.

This Source Code Form is subject to the terms of the
Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""

__version__ = "0.0.12"
AUTHOR = "Vanessa Sochat"
AUTHOR_EMAIL = "vsochat@stanford.edu"
NAME = "codeart"
PACKAGE_URL = "http://www.github.com/vsoch/codeart"
KEYWORDS = "code, pointilism, art, graphics"
DESCRIPTION = "Data visualizations with your own code."
LICENSE = "LICENSE"

################################################################################
# Requirements


INSTALL_REQUIRES = (
    ("Pillow", {"min_version": "6.0.0"}),
    ("textblob", {"min_version": "0.15.3"}),
    ("nltk", {"min_version": "3.4"}),
    ("gensim", {"min_version": "3.8.1"}),
    ("numpy", {"min_version": "1.16.2"}),
    ("pandas", {"min_version": "0.24.2"}),
)

TESTS_REQUIRES = (("pytest", {"min_version": "4.6.2"}),)
ANIMATE_REQUIRES = (("imageio", {"min_version": "2.5.0"}),)

INSTALL_REQUIRES_ALL = INSTALL_REQUIRES + ANIMATE_REQUIRES
