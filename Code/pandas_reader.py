# -*- coding: utf-8 -*-
import pandas

Data_location="../Data/"
FileName = "fake.csv"



InputFile = Data_location + FileName

LoadedFile = pandas.read_csv(InputFile)
for t in LoadedFile['text'] :
	print(t)
	#for w in t :
	#	print(w.lower())

#with open(InputFile,'rt',encoding="utf8") as f:
#        data = csv.reader(f)
#        for row in data:
#            #print(row[3])
#            print(row)
