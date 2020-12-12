#!/usr/bin/env python2.7
#
# thesaurus-lookup
#
# Author: Anupam Sengupta (anupam@Anupams-MacBook-Pro.local.)
#
# Copyright (C) 2014
#
# Released under the BSD license.
#
# This script provides a list of thesaurus entries from the webservice
# at <http://thesaurus.altervista.org>
#
# You will need to get a personal API key from:
# <http://thesaurus.altervista.org/mykey>
#

import sys
import urllib
import pickle
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen
import json
import argparse
from collections import Counter
from tqdm import tqdm
from util import load_pickle, save_pickle, load_file

def get_params(word, source):
    """Return the param hash with the word"""
    if source == "altervista":
        params['word'] = word
    elif source == "datamuse":
        params["ml"] = word
    return params

def get_encoded_url(base, params):
    """Get the URL encoded URLs"""
    return base + '?' + urlencode(params)

def get_response(url):
    """Get the JSON response from the URL"""
    resp = urlopen(url)
    return json.loads(resp.read().decode('UTF-8'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--smaller', action="store_true")
    args = parser.parse_args()

    source = args.source
    dataset = args.dataset
    smaller = args.smaller

    if source == "altervista":
        baseurl = 'http://thesaurus.altervista.org/thesaurus/v1'
        params = {
            'key'      : 'HkHuVUtxLtmmQUxmTZW2',
            'language' : 'en_US',
            'output'   : 'json',
        }
    elif source == "datamuse":
        baseurl = 'https://api.datamuse.com/words'
        params = {}

    # load not found
    not_found = load_pickle("./data/KB/VAD/{0}{1}_not_found_v2.pkl".format(dataset.lower(), "-smaller" if smaller else ""))

    # syns
    syns_dict = {}
    for word, cnt in tqdm(not_found):
        try:
            syns_dict[word] = get_response(get_encoded_url(baseurl, get_params(word, source)))
        except HTTPError as e:
            pass
    
    # save syns
    print("Saving syns...")
    save_pickle(syns_dict, "./data/KB/VAD/{0}{1}_not_found_v2_{2}_raw.pkl".format(dataset.lower(), "-smaller" if smaller else "", source))
