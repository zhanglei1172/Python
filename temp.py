# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
def flatten(nested):
    try:
        for sublist in nested:

            for ele in flatten(sublist):
                yield ele

    except:
        yield nested
