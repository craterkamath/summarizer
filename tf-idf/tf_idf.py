# -*- coding: utf-8 -*-
from __future__ import division, unicode_literals

#import stemmer_test
import re
from textblob import TextBlob as tb
import json
import math
import os.path
import sys
import pandas as pd
reload(sys)
sys.setdefaultencoding('utf-8')
import pickle
import mtranslate as m
from rouge import Rouge
import csv

rouge = Rouge()

class TfIdf:

	def __init__(self, corpusPath, outDir ):
		self.cps = corpusPath
		self.corpus = ""#json.load(open(corpusPath[0], 'r'))
		self.outFileDir = outDir
		self.rougedir = {}

		if not os.path.exists(self.outFileDir):
			os.makedirs(self.outFileDir)
		self.reg = re.compile('\. |\.\xa0')
		self.wordDfDict = {}
		self.trainBloblist = []
		self.testBloblist = []
		self.trainBloblistLength = 0
		self.testBloblistLength = 0
		#gss
		with open('../gss.pickle','r') as f:
			self.gss = pickle.load(f)
		with open('../allDicts.pickle','r') as f:
			l = pickle.load(f)
			self.categoryWordDict = []
			self.categoryWordDict.append(l[0])
			self.categoryWordDict.append(l[1])
			self.categoryWordDict.append(l[2])
			self.categoryDictLength = l[3]
			self.cindex = l[4]


	def setup(self):
		for cp in self.cps:
			df = pd.read_csv(cp)
			self.corpus = json.loads(df.to_json(orient = "records"))
			# import pdb; pdb.set_trace()
			self.buildCorpus()
		self.calculateDf()

	def tf(self,blob):
		out = {}
		for word in blob.words:
			if word not in out:
				out[word] = 0
			out[word] += 1
		for key,value in out.iteritems():
			out[key] = value/len(blob.words)
		return out

	def computeIdf(self, df):
		return math.log(self.trainBloblistLength + 1 / (1 + df))

	def buildCorpus(self):
		for i in range(0,len(self.corpus)):
			# content = '.'.join(self.corpus[i]['t_content'].replace("\n", " "))
			content = self.corpus[i]['t_content'].replace("\n", " ")
			content.replace('..','.')
			self.trainBloblist.append(tb(content))
		self.trainBloblistLength = len(self.trainBloblist)

	def buildTestData(self, devPath = None):
		self.testBloblist = {}
		for idx in range(0, len(self.dev)):
			# content = '.'.join(self.dev[idx]['t_content'].replace("\n", " "))
			content = self.dev[idx]['t_content'].replace("\n", " ")
			content.replace('..','.')
			self.rougedir[str(idx) + devPath.split(".")[0]] = {"ground_truth": self.dev[idx]['summary'], "predicted": "", "rouge": 0.0}
			self.testBloblist[idx] = (tb(content))
		self.testBloblistLength = len(self.testBloblist)

	def calculateDf(self):
		for i, blob in enumerate(self.trainBloblist):
			#print i
			for word in set(blob.words):
				if word not in self.wordDfDict:
					self.wordDfDict[word] = 0
				self.wordDfDict[word] += 1

	def extractSummary(self, devPath, outFileName):
		df = pd.read_csv(devPath)
		self.dev = json.loads(df.to_json(orient = "records"))
		self.buildTestData(devPath)
		out = {}
		c = {0:0,1:0,2:0}
		for i, blob in self.testBloblist.iteritems():
			cn = self.getCategoryNumber(blob)
			c[cn] += 1
			sentenceList = self.reg.split(unicode(blob))
			sentenceRankDict = {}
			tfw = self.tf(blob)
			for j in range(0,len(sentenceList)):
				sentence = tb(sentenceList[j])
				sentenceRank = 0
				for word in sentence.words:
					if word in self.wordDfDict:
						tf = tfw[word]
						df = self.wordDfDict[word]
						tfIdf = tf * self.computeIdf(df+1)
						gss = 0
						if word in self.gss:
							gss = tf*self.gss[word][cn]
						sentenceRank += (tfIdf + gss)

				if sentenceRank != 0:
					sentenceRankDict[sentence] = [sentenceRank, j]

			topSentences = sorted(sentenceRankDict.items(), key=lambda x: x[1][0], reverse=True)
			#deciding
			topSentencesToFile = ""
			#select 20% of article, with min = 4 , max = 6 sentences
			numberOfSentence = int(math.floor(0.2*len(sentenceList)))
			if  numberOfSentence > 6:
				numberOfSentence = 6
			elif numberOfSentence < 4:
				numberOfSentence = 4

			topSentences = sorted(topSentences[:numberOfSentence], key=lambda x: x[1][1])
			for sentence, sentenceNumber in topSentences:
				topSentencesToFile += format(sentence)+". \n"
			out[i] = {"text" : topSentencesToFile}
			articleNumber = i
			sentencesToFile = ""
			for sentence in sentenceList:
				sentencesToFile += format(sentence)+". \n"
			t = outFileName.split(".")[0]
			self.writeToFile(str(articleNumber)+t, sentencesToFile, topSentencesToFile)
		print c
		outfileName = "system_"+outFileName
		with open(outfileName, 'w') as outfile:
			json.dump(out, outfile)

	def getCategoryNumber(self, blob):
		#naive bayes to determine category
		out = [1.0, 1.0, 1.0]
		#cinema     #state        #sports
		for i in range(0, len(self.cindex)):
			#out[i] *= self.categoryDictLength[i]/ (sum(self.categoryDictLength)- self.categoryDictLength[i]) # prior
			for word in blob.words:
				if word in self.categoryWordDict[i]:
					out[i] = out[i]*math.log( self.categoryWordDict[i][word]/self.categoryWordDict[i]["total_words_category"])
		return out.index(max(out))


	def writeToFile(self, articleNumber, sentencesToFile, topSentencesToFile):
		outfileName = os.path.join(self.outFileDir, articleNumber + ".txt")
		outFile = open(outfileName, 'w')
		outFile.write(sentencesToFile)
		outFile.write('\n')
		outFile.write("--------------------- Summary -----------------------------")
		outFile.write('\n')
		try:
		    topSentencesToFile = m.translate(topSentencesToFile.encode('utf-8'), "en", "kn").encode('utf-8')
		except Exception as e:
		    print(e, "Out File Name:", outfileName)
		    MX_LIMIT = 2000
		    sentences_len = len(topSentencesToFile)
		    sentences_split = topSentencesToFile.split(".")
		    resp_sent = ""
		    while sentences_split:
		        request_sent = ""
		        while(sentences_split and len(request_sent) + len(sentences_split[0]) < MX_LIMIT):
		            request_sent += sentences_split[0]
		            sentences_split.pop(0)
		        resp_sent = resp_sent + ". " + m.translate(request_sent.encode('utf-8'), "en", "kn").encode('utf-8')
		    topSentencesToFile = resp_sent
		# import pdb; pdb.set_trace()
		self.rougedir[articleNumber]["predicted"] = topSentencesToFile
		self.rougedir[articleNumber]["rouge"] = rouge.get_scores(topSentencesToFile,
																	 self.rougedir[articleNumber]["ground_truth"])
		outFile.write(topSentencesToFile)
		outFile.close()

	def to_csv(self, fileName = "output.csv"):
		field_names = ["id", "ground_truth", "predicted", "rouge-1", "rouge-2", "rouge-l"]
		with open(fileName, 'w') as f:
			writer = csv.writer(f)
			writer.writerow(field_names)
			rouge_1 = []
			rouge_2 = []
			rouge_l = []
			for key, value in self.rougedir.iteritems():
				writer.writerow([key,
									value["ground_truth"],
									value["predicted"],
									value["rouge"][0]["rouge-1"]["f"],
									value["rouge"][0]["rouge-2"]["f"],
									value["rouge"][0]["rouge-l"]["f"],
									])
				rouge_1.append(value["rouge"][0]["rouge-1"]["f"])
				rouge_2.append(value["rouge"][0]["rouge-2"]["f"])
				rouge_l.append(value["rouge"][0]["rouge-l"]["f"])
			writer.writerow(["Mean", "Mean", "Mean",
							sum(rouge_1)/float(len(rouge_1)),
							sum(rouge_2)/float(len(rouge_2)),
							sum(rouge_l)/float(len(rouge_l))])

#corpusPath = ["../crawler/udayavani_cinema_news.json", "../crawler/udayavani_sports_news.json", "../crawler/udayavani_state_news.json"]
#corpusPath = ["cinema_test.json", "state_test.json","sports_test.json"]
corpusPath = ["sample_data.csv"]
t = TfIdf(corpusPath, 'results_csv' )
t.setup()
# t.extractSummary("cinema_test.json", "cinema.json")
# t.extractSummary("state_test.json", "state.json")
#t.extractSummary("sports_test.json", "sports.json")
t.extractSummary("sample_data.csv", "sample_data.json")
t.to_csv()