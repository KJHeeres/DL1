import shutil
import os
import csv

# vars
root = "D:/AI/3/Deep Learning/Dataset/"

# the stache column
#stacheCol = 23
staches = []

# things staches should be (male)
stachePositives = [21]
# things staches can't be (not no beard, stubble)
stacheNegatives = []

# things no staches should be (no beard, male)
noStachePositives = []
# things staches can't be (have mustache, stubble)
noStacheNegatives = [21]
noStaches = []

nrOfStaches = 0
nrOfNoStaches = 0

# find all the staches
with open(root + "list_attr_celeba.csv") as csv_file:
	reader = csv.reader(csv_file, delimiter=",")
	currRow = 0
	# for each row
	for row in reader:
		stache = True
		noStache = True

		# check if they qualify as stache
		
		for col in stachePositives:
			if row[col] == "-1":
				stache = False
		for col in stacheNegatives:
			if row[col] == "1":
				stache = False

		# check if they qualify as nostache
		for col in noStachePositives:
			if row[col] == "-1":
				noStache = False
		for col in noStacheNegatives:
			if row[col] == "1":
				noStache = False
		
		# add to the correct list
		if stache and nrOfStaches < 10000 * 1: 
			staches.append(currRow)
			nrOfStaches += 1
		elif noStache and nrOfNoStaches < 10000 * 1: 
			noStaches.append(currRow)
			nrOfNoStaches += 1

		currRow += 1


# copy the staches
currFile = 1
for file in os.listdir(root + "img_align_celeba/img_align_celeba"):
	if nrOfStaches >= 0 and currFile in staches:
		shutil.copyfile(root + "img_align_celeba/img_align_celeba/" + file, root + "mf/m/" + file)
	elif nrOfNoStaches >= 0 and currFile in noStaches:
		shutil.copyfile(root + "img_align_celeba/img_align_celeba/" + file, root + "mf/f/" + file)
		nrOfNoStaches -= 1
	if currFile % 1000 == 0:
		
		print(currFile)
	currFile += 1