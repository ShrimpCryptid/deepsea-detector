import json
from pathlib import Path
import urllib.parse
import requests

""" Formats identifications from FathomNet.
"""

default_phyla = "Unidentified Biology"

class CategoryDictionary:
    """ Stores concept phyla for faster lookup.
    """

    # Note: phylum is not a perfect organization for fish because phylum Chordata also includes tunicates/urochordates. 
    # Must perform multiple levels of checks in order to determine organization.
    # These are just invertebrate phyla that are easy to categorize :)
    recognized_phyla = {
        "Annelida",
        "Arthropoda",
        "Cnidaria",
        "Echinodermata",
        "Mollusca",
        "Porifera",
    }
    category_to_concepts: dict[str, set[str]]

    def __init__(self, blocklist: list[str]=[]):
        return

    def contains(self, concept: str) -> bool:
        return
    
    def lookup(self, concept: str) -> str or None:
        return

    def load() -> CategoryDictionary:
        return


def try_get_worms_record(name: str) -> json:
    """ Tries to get the Aphia Record for a given identification name from the
    World Register of Marine Species (WoRMS).
    Searches first by exact scientific name, and then with fuzzy search.

    Returns the Aphia record as a JSON object if object was successfully matched, otherwise returns None.
    """

    # Escape non-URL characters in name
    parsed_name = urllib.parse.quote(name)

    # First try to access using the scientific name.
    url = "https://www.marinespecies.org/rest/AphiaRecordsByName/" + parsed_name
    response = requests.get(url)
    if response.status_code == 200: # Success! 
        # Response may return multiple Aphia records, so we return the first one.
        return response.json()[0]

    elif response.status_code == 204:
        # No records found, so we try again with fuzzy search.
        url = "https://www.marinespecies.org/rest/AphiaRecordsByMatchNames?scientificnames[]={}&marine_only=true"
        url = url.format(parsed_name)
        response = requests.get(url)
        if (response.status_code == 200): # Success!
            return response.json()[0][0]   

    if response.status_code == 400:  # Check if either response was faulty.
        # Likely an issue with query, so we raise an error.
        raise RuntimeError("Response status code was 400 for url '{}'. Parameters may be invalid.".format(url))
    
    return None


def get_concept_phyla(concept: str, dictionary: CategoryDictionary) -> str or None:
    """ Gets the phylum of an organism for a given concept.

    Optionally, 
    """
    return
