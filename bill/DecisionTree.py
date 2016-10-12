# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 18:07:19 2016

@author: Bill
"""

class TreeNodes():
    
    def __init__(self, columnIndex=-1, value=None, results=None, trueBranch=None, falseBranch=None):
        self.col = columnIndex
        self.val = value
        self.res = results #none except for leafnodes
        self.tb = trueBranch
        self.fb = falseBranch
        
    #divide data on a specific column
    def splitData(data, col, value):
        splitter = lambda row: row[col] >= value
        set1 = [row for row in data if splitter(row)]
        set2 = [row for row in data if not splitter(row)]
        return (set1, set2)
        
    def uniqueCount(data):
        results = {}
        for row in data:
            r = row[len(row)-1]
            if r not in results: 
                results[r] = 0
            results[r]+=1
        return results
    
    def entropy(data):
        from math import log
        log2 = lambda x: log(x)/log(2)
        ent = 0.0
        results = TreeNodes.uniqueCount(data)
        for key in results.keys():
            p = float(results[key])/len(data)
            ent = ent - p*log2(p)
        return ent
        
    def buildTree(data, scoref=entropy):
        if len(data)==0: 
            return TreeNodes()
        currentScore = scoref(data)
        
        #track best criteria
        bestGain = 0.0
        bestCriteria = None
        bestSets = None
        
        columnCount = len(data[0])-1
        for col in range(0, columnCount):
            #generate list of different possible values of each column, stored as keys in colVals
            colVals = {}
            for row in data:
                colVals[row[col]]=1
            for val in colVals.keys():
                set1, set2 = TreeNodes.splitData(data, col, val)
                
                #information gain
                p = float(len(set1))/len(data)
                gain = currentScore - p*scoref(set1) - (1-p)*scoref(set2)
                if gain > bestGain and len(set1) > 0 and len(set2) > 0:
                    bestGain = gain
                    bestCriteria = col, val
                    bestSets = (set1, set2)
        #create sub-branches
        if bestGain > 0:
            print(bestGain)
            trueBranch = TreeNodes.buildTree(bestSets[0])
            falseBranch = TreeNodes.buildTree(bestSets[1])
            return TreeNodes(columnIndex=bestCriteria[0], value=bestCriteria[1], 
                             trueBranch=trueBranch, falseBranch = falseBranch)
        else:
            return TreeNodes(results = TreeNodes.uniqueCount(data))
                    
#####################################
#utility methods for drawing the tree
#####################################
from PIL import Image, ImageDraw

def getWidth(tree):
    if tree.tb == None and tree.fb == None: 
        return 1
    return getWidth(tree.tb) + getWidth(tree.fb)
    
def getDepth(tree):
    if tree.tb == None and tree.fb == None:
        return 0
    return max(getDepth(tree.tb), getDepth(tree.fb))+1    
    
def drawTree(tree, jpeg='tree.jpg'):
    w = getWidth(tree)*180
    h = getDepth(tree)*100 + 120

    img = Image.new('RGB', (w,h), (255,255,255))
    draw = ImageDraw.Draw(img)
    
    drawNode(draw, tree, w/2, 20)
    img.save(jpeg, 'JPEG')
    
def drawNode(draw, tree, x, y):
    if tree.res == None:
        w1 = getWidth(tree.fb)*100
        w2 = getWidth(tree.tb)*100

        left = x - (w1+w2)/2
        right = x + (w1+w2)/2

        draw.text((x-20, y-10), str(tree.col) + ':' + str(tree.val),(0,0,0))
        
        draw.line((x, y, left+w1/2, y+100), fill=(255,0,0))
        draw.line((x, y, right+w2/2, y+100), fill=(255,0,0))
        
        drawNode(draw, tree.fb, left+w1/2, y+100)
        drawNode(draw, tree.tb, right+w2/2, y+100)
    else:
        txt = ' \n'.join(['%s:%d'%v for v in tree.res.items()])
        draw.text((x-20,y), txt, (0,0,0))

        
        
#############################################
#methods for loading and vectorizing the data
##############################################                  
import random, csv

features = []
dataset = []
testData = []  
#load the list of features from 'features.csv'
def loadFeatures():
    with open('features.csv') as infile:
        for i, row in enumerate(csv.reader(infile)):
            if i == 0:
                continue
            features.append(row[0])
    
    random.shuffle(features)
    return features[:200]

#vectorizes the abstracts    
def vectorize(tokens):
    vector = []
    for word in features:
        count = 0
        for item in tokens:
            if word == item:
                count = count + 1
        vector.append(count)
        #if word in tokens:
        #    vector.append(1)
        #else:
        #    vector.append(0)          
    return vector
    
features = loadFeatures()    
    
with open('processedData.csv') as infile:
    for i, row in enumerate(csv.reader(infile)):
        if i == 0:
            continue
        if i > 2000:
            break
        if len(row) < 4:
            continue
        vector = vectorize(row[2:])
        vector.append(row[1])
        dataset.append(vector)

tree = TreeNodes.buildTree(dataset)
drawTree(tree, jpeg='treeview.jpg')
