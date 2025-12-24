
"""
Auto-generated download script for Thingi10K dataset.
Assuming the following python packages and external commands are available.

* argparse: for parse command line args.
* requests: for http communication.
* wget: for downloading files.

Usage:

    python Thingi10K_download.py

or

    python Thingi10K_download.py -o output_dir

"""

import argparse
import os.path
import sys
import os
from subprocess import check_call

try:
    import requests
except ImportError:
    error_msg = "Missing module: requests.  To fix it, please run '$ pip install requests'";
    sys.exit(error_msg);

try:
    check_call("which wget".split());
except:
    error_msg = "Missing command: wget.  To fix it, please run '$ port install wget'";
    sys.exit(error_msg);

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__);
    parser.add_argument("--output", "-o", help="Output directory", default="./");
    return parser.parse_args();

def download_file(file_id, output_dir):
    if not os.path.isdir(output_dir):
        raise IOError("Directory {} does not exists".format(output_dir));
    url = "https://www.thingiverse.com/download:{}".format(file_id);
    r = requests.head(url);
    link = r.headers.get("Location", None);
    if link is None:
        print("File {} is no longer available on Thingiverse.".format(file_id));
        return;

    __, ext = os.path.splitext(link);
    output_file = "{}{}".format(file_id, ext.lower());
    output_file = os.path.join(output_dir, output_file);
    try : 
        output_file = output_file[:output_file.index('?')]
    except Exception :
        pass
    print("Downloading {}".format(output_file));
    command = "wget -q -O {} --tries=10 {}".format(output_file, link);
    check_call(command.split());

def main():
    args = parse_args();
    with open('file_ids.txt') as fp :
        file_ids = [int(_.strip()) for _ in fp.readlines()]
    for file_id in file_ids:
        download_file(file_id, args.output);

if __name__ == "__main__":
    main();
