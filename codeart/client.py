#!/usr/bin/env python

"""

Copyright (C) 2019-2020 Vanessa Sochat.

This Source Code Form is subject to the terms of the
Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""

from codeart.main import CodeBase
import codeart
import argparse
from glob import glob
import sys
import os
import tempfile


def get_parser():
    parser = argparse.ArgumentParser(description="Code Art Generator")
    parser.add_argument(
        "--version",
        dest="version",
        help="print the version and exit.",
        default=False,
        action="store_true",
    )

    description = "actions for Code Art generator"
    subparsers = parser.add_subparsers(
        help="codeart actions", title="actions", description=description, dest="command"
    )

    textart = subparsers.add_parser(
        "textart", help="extract images from GitHub or local file system."
    )

    textart.add_argument(
        "--github",
        dest="github",
        help="GitHub username to download repositories for.",
        type=str,
        default=None,
    )

    textart.add_argument(
        "--root",
        dest="root",
        help="root directory to parse for files.",
        type=str,
        default=None,
    )

    textart.add_argument(
        "--outdir",
        dest="outdir",
        help="output directory to extract images (defaults to temporary directory)",
        type=str,
        default=None,
    )

    textart.add_argument(
        "--text",
        dest="text",
        help="Text to write for image (defaults to folder name).",
        type=str,
        default=None,
    )

    return parser


def main():
    """main is the entrypoint to the code art client.
    """

    parser = get_parser()

    # Will exit with subcommand help if doesn't parse
    args, extra = parser.parse_known_args()

    # Show the version and exit
    if args.version:
        print(codeart.__version__)
        sys.exit(0)

    # Initialize the JuliaSet
    if args.command == "textart":

        from codeart.graphics import generate_codeart_text
        from codeart.colors import generate_color_lookup

        # If not output directory, create temporary
        outdir = args.outdir
        if not args.outdir:
            outdir = tempfile.mkdtemp()
        code = CodeBase()

        # GitHub repository
        if args.github is not None:
            code.add_repo(args.github)
            text = os.path.basename(args.github)
        elif args.root is not None:
            code.add_folder(args.root)
            text = os.path.basename(args.root)

        text = args.text or text

        # Train a model for all extensions
        image_dir = os.path.join(outdir, "images")
        images = code.make_art(group="all", outdir=image_dir)
        images = glob("%s/*" % os.path.join(outdir, "images"))
        color_lookup = generate_color_lookup(images)

        # Generate an image with text (dinosaur!)
        generate_codeart_text(
            text, color_lookup, outfile=os.path.join(outdir, "index.html")
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
