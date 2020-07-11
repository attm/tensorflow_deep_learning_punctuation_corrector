import pandas as pd 
import numpy as np 
import re
import sys
from utils.check_decorators import type_check


# ============ STR PROCESSORS ============ #
@type_check
def pad_symbol(string : str, symbol_to_pad : str) -> str:
    """
    Padds specified symbols by adding space before and after symbol.

    Parameters: 
        string (str) : String to be processed.
        symbol_to_pad (str) : symbol that needs to be padded
    Returns:
        Padded string.
    """
    string = string.replace(symbol_to_pad, " {0} ".format(symbol_to_pad))
    return string

@type_check
def remove_multiple_spaces(string : str) -> str:
    """
    Removs double or more spaces in a row.

    Parameters:
        string (str) : String to be processed.
    Returns:
        String without multiple spaces
    """
    # Replace all double and more spaces with single space
    string = re.sub('\s+', " ", string)
    return string

@type_check
def replace_uppercase(string : str, replacing_with : str) -> str:
    """
    Replaces all uppercase characters with given replacing_with str.

    Parameters:
        string (str) : String to be processed.
        replacing_with (str) : str will replace all uppercase characters
    Returns:
        String with replaced uppercase characters.
    """
    string = re.sub('[A-Z]', replacing_with, string)
    return string

@type_check
def add_before_uppercase(string : str, add_str : str) -> str:
    """
    Adds add_str before all uppercase characters.

    Parameters:
        string (str) : String to be processed.
        add_str (str) : str will add before all uppercase characters.
    Returns:
        String with added add_str strings.
    """
    string = re.sub(r"([A-Z])", r"{0}\1".format(add_str), string)
    return string

def token_to_uppercase(string : str, uppercase_token : str) -> str:
    """
    Finds all tokenst and makes next characters uppercase.

    Parameters:
        string (str) : String to be processed.
        uppercase_token (str) : after all those tokens next characters will be changed to uppercase, token will be removed.
    Returns:
        String with added add_str strings.
    """
    # Matching patter "token + letter", getting last char of string (letter), uppercase it
    token_plus_letter = re.match(r"({0})([a-z])".format(uppercase_token), string)
    if token_plus_letter is not None or "":
        token_plus_letter = token_plus_letter.group(0)[-1].upper()
        string = re.sub(r"({0})([a-z])".format(uppercase_token), token_plus_letter, string)
    return string