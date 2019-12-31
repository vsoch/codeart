#!/usr/bin/env python

"""

Copyright (C) 2019-2020 Vanessa Sochat.

This Source Code Form is subject to the terms of the
Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""

import codeart
import argparse
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

    extract = subparsers.add_parser(
        "extract", help="extract images from GitHub or local file system."
    )

    extract.add_argument(
        "--github",
        dest="github",
        help="GitHub username to download repositories for.",
        type=str,
        default=None,
    )

    extract.add_argument(
        "--root", dest="root", help="root directory to parse for files.", type=str
    )

    extract.add_argument(
        "--outdir",
        dest="outdir",
        help="output directory to extract images (defaults to temporary directory)",
        type=str,
        default=None,
    )

    extract.add_argument(
        "--year",
        dest="year",
        help="Oldest year to include files from (defaults to 2010).",
        type=int,
        default=2010,
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
    if args.command == "extract":

        # If not output directory, create temporary
        outdir = args.outdir
        if not args.outdir:
            outdir = tempfile.mkdtemp()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
