# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:43:12 2020

@author: lenovo
"""

import csv
import urllib

csvfile = open("pepBDB_results.csv", "r")
reader = csv.reader(csvfile)
querys = {}
for item in reader:
    if reader.line_num == 1:
        continue
    pdbid = item[0][:6]
    url = "http://huanglab.phys.hust.edu.cn/pepbdb/db/" + pdbid + "/inter.dat"
    urllib.request.urlretrieve(url, pdbid + ".dat")
print("down")
