# -*- coding: utf-8 -*-
import csv

Data_location="../Data/"
InputFile = Data_location + "fake.csv"

with open(InputFile,'rt',encoding="utf8") as f:
        data = csv.reader(f)
        for row in data:
            #print(row[3])
            print(row)
