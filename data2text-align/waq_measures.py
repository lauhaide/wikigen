__author__ = 'lperez'

import sys, argparse, os
from dataset.DBPProperty import DBPProperty


def loadPropertyList(dataFile):
    """
    Auxiliary method, read original list of properties to be able to merge redundant ones.
    This would not be needed if data was cleaned beforehand.
    """
    propNameList = []
    f = open(dataFile, "r")
    lNb = 0
    for line in f.readlines():
        if lNb == 2:
            #read props
            props = line.split()
            for prop in props:
                propNameList.append(prop.split("(")[0])
            break
        lNb +=1
    return propNameList


def loadAbstractSentences(dirName):
    """
    Recovers the nb and lengths for sentences in a given abstract.
    :param dirName:
    :return:
    """

    abstractLengths = {}
    for file in os.listdir(dirName):
      sentSizes = []
      if file.endswith(".crp"):
        curid = file.split(".crp")[0].split(".")[1]
        f = open(dirName+file, "r")
        lNb =0
        nextlnb=1
        sentCnt=0
        for line in f.readlines():
            if lNb == nextlnb:
                print(line)
                print(line.strip().split())
                nextlnb +=3
                sentCnt +=1
                sentSizes.append((sentCnt, len(line.strip().split())))
            lNb +=1
        abstractLengths[curid] = sentSizes

    return abstractLengths

def distributeAlignmentsPerSentence(systemsAlns, abstractLengths):
    """

    :param systemsAlns: elements in this dict are of the form [('0', '4', 'P'), ('1', '8', 'S'),....,('53', '14', 'S')]
                        this list should be ordered by first element
    :param abstractLengths: elements in this dict are of the form [(1, 17), (2, 19), (3, 18)]
    :return:
    """
    newSystemAlns = {}
    for curidSentID in systemsAlns.keys():
        curid = curidSentID.split(".")[0]
        sentLengths = abstractLengths[curid]
        iSent = 0
        sentIdAlns = []
        sentId = sentLengths[iSent][0]
        sentBoundary = sentLengths[iSent][1]
        for wid, rid, atype in systemsAlns[curidSentID]:
            if int(wid) < sentBoundary:
                sentIdAlns.append((wid, rid, atype))
            else:
                newSystemAlns[curid+"."+str(sentId-1)] = sentIdAlns #sent ids read from yawat start at 0
                sentIdAlns = [(wid, rid, atype)]
                iSent +=1
                if iSent <= len(sentLengths):
                    sentId = sentLengths[iSent][0]
                    sentBoundary = sentBoundary  + sentLengths[iSent][1]
        newSystemAlns[curid+"."+str(sentId-1)] = sentIdAlns
    return newSystemAlns


def mergeProperties(properties, propertyList):
    """
    Merge redundant/duplicated e.g. "birth_name"/"birthname" or "fullname"/"name"
    this would not be needed if data was cleaned beforehand
    """
    pnames = []
    for p in properties:
        pnames.append(propertyList[int(p)])
    if set(DBPProperty.IDENTPROPS1).issuperset(set(pnames)):
        return True
    if set(DBPProperty.IDENTPROPS2).issuperset(set(pnames)):
        return True
    return False

def alnYawatRead(dirName, max, asSystem=False):
    """
    reads Yawat alignments file.
    :return:
    """
    nbAbstracts = 0
    yawatAlignments = {}
    sentCount = 0
    for file in os.listdir(dirName):
      if file.endswith(".aln"):
        dataFile = file.replace(".aln", ".crp")
        propertyList = loadPropertyList(dirName+dataFile)
        nbAbstracts +=1
        curid = file.split(".aln")[0].split(".")[1]
        f = open(dirName+file, "r")
        for i, line in enumerate(f.readlines()):
            if not line.strip(): #or not line.strip().split()[1:]:
                continue
            key = curid + "." + str(line.strip().split()[0])
            yawatAlignments[key] = []
            sentCount +=1
            alignments = line.strip().split()[1:]

            for aln in alignments:
                parts = aln.split(":")
                atype = parts[2]
                words = parts[0].split(",")
                triples = parts[1].split(",")
                if max:
                    merge = mergeProperties(triples, propertyList)
                    for w in words:
                        if merge:
                            yawatAlignments[key].append((w,triples,atype))
                        else:
                            for t in triples:
                                yawatAlignments[key].append((w,[t],atype))
                else:
                    for w in words:
                        for t in triples:
                            if asSystem:
                                yawatAlignments[key].append((w,t,atype))
                            else:
                                yawatAlignments[key].append((w,[t],atype))


    print("nb of abtracts read from yawat: {} with total of {} sentences.".format(nbAbstracts, sentCount))
    return yawatAlignments


def alnDtaRead(fileName):
    """
    This is the format on input file

    curid=17799472  sentence ID=0
    word=7 relation=0 || skater || 11041#14797#10258#18157 || S
    word=9 relation=0 || competed || 11041#14797#10258#18157 || P

    :param fileName:
    :return:
    """
    nbAbstracts = set()
    dtaAlignments = {}
    f = open(fileName, "r")
    currentKey = None
    sentCount = 0
    for i, line in enumerate(f.readlines()):
        if line.startswith("curid="):
            parts = line.strip().split()
            currentKey = parts[0].split("curid=")[1] + "." + parts[1].split("sentence_ID=")[1]
            dtaAlignments[currentKey] = []
            nbAbstracts.add(parts[0].split("curid=")[1])
            sentCount +=1
        else:
            parts = line.split(" || ")
            if "2#2" == parts[2].strip(): #skip alingments to NONE property, parts[1] is the word from the sentence
                #print("skip NONE relation")
                continue
            aln = parts[0].strip().split()
            w = aln[0].split("word=")[1]
            t = aln[1].split("relation=")[1]
            a = parts[3].strip()
            dtaAlignments[currentKey].append((w, t, a))

    print("nb of abtracts read from dta: {} with total of {} sentences.".format(len(nbAbstracts), sentCount))
    return dtaAlignments


def computeAlnMeasures(dtaAlignments, yawatAlignments, first='2', givenSet=[]):
    """
    https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
    """
    allIntersects = 0
    alnsYawat = 0
    alnsDta = 0
    recs = 0.0
    precs = 0.0
    precsOnSure = 0.0
    recsOnSure = 0.0
    nbSentences = 0
    nbSentencesForRecall = 0
    for alnKey in yawatAlignments.keys(): ##Yawat files have a complete list of all IDs, i.e. all abstracts all sentences

        if givenSet and (int(alnKey.split(".")[0]) not in givenSet):
            print("Yawat curid not in results {}".format(alnKey.split(".")[0]))
            continue
        if first=='1' and alnKey.split(".")[1]!='0':
            continue #only do stats on first sentence
        if first=='0' and alnKey.split(".")[1]=='0':
            continue #only do stats on ALL MINUS first sentence


        if not alnKey in dtaAlignments.keys():
            dtaAlignments[alnKey] = []

        nbSentences += 1

        dtaAln = dtaAlignments[alnKey]
        dtaAlnAll = [(w,t) for w,t,a in  dtaAln]
        dtaAlnSure = [(w,t) for w,t,a in  dtaAln if a=="S" ]

        yawatAln = yawatAlignments[alnKey]
        yawatAlnAll = [(w,t) for w,t,a in  yawatAln ]
        yawatAlnSure = [(w,t) for w,t,a in  yawatAln if a=="sure" ]
        if yawatAln:
            nbSentencesForRecall +=1

        #intersect = [1 for w,t in dtaAlnAll if (w,t) in yawatAlnAll ]
        intersect = []
        for w,t in dtaAlnAll:
            for yw, yt in yawatAlnAll:
                if w==yw and t in yt:
                    intersect.append(1)

        #intersectSure = [1 for w,t in dtaAlnAll if (w,t) in yawatAlnSure ]
        intersectSureRec = []
        for w,t in dtaAlnAll:
            for yw, yt in yawatAlnSure:
                if w==yw and t in yt:
                    intersectSureRec.append(1)

        intersectSurePrec = []
        for w,t in dtaAlnSure:
            for yw, yt in yawatAlnAll:
                if w==yw and t in yt:
                    intersectSurePrec.append(1)

        allIntersects += len(intersect)
        alnsYawat += len(yawatAlnAll)
        alnsDta += len(dtaAlnAll)

        prec = float(len(intersect)) / float(len(dtaAlnAll)) if len(dtaAlnAll) > 0 else 0.0
        rec = float(len(intersect)) / float(len(yawatAlnAll)) if len(yawatAlnAll) > 0 else 0.0
        if (len(dtaAlnSure)==0): #switch to normal way of computing
            precOnSure = prec
        else:
            precOnSure = float(len(intersectSurePrec)) / float(len(dtaAlnSure)) if len(dtaAlnSure) > 0 else 0.0 #float(len(dtaAlnAll)) if len(dtaAlnAll) > 0 else 0.0
        if (len(yawatAlnSure)==0): #switch to normal way of computing
            recOnSure = rec
        else:
            recOnSure = float(len(intersectSureRec)) / float(len(yawatAlnSure)) if len(yawatAlnSure) > 0 else 0.0 #float(len(yawatAlnAll)) if len(yawatAlnAll) > 0 else 0.0

        if prec > 0 or rec > 0:
            fmeasure = 2.0 * ((prec * rec) / (prec + rec))
        else:
            fmeasure = 0.0
        print("[{}] prec: {} rec: {}, rec_S: {}, f1: {}".format(alnKey, prec, rec, recOnSure, fmeasure ))

        precs += prec
        recs += rec
        precsOnSure += precOnSure
        recsOnSure += recOnSure

    macroPrec = precs / float(nbSentences)
    macroRec = recs / float(nbSentencesForRecall)
    macroPrecOnSure = precsOnSure / float(nbSentences)
    macroRecOnSure = recsOnSure / float(nbSentencesForRecall)

    if (macroPrec + macroRec) > 0:
        macrof1 = 2.0 * ((macroPrec * macroRec) / (macroPrec + macroRec))
        macrof1OnSure = 2.0 * ((macroPrecOnSure * macroRecOnSure) / (macroPrecOnSure + macroRecOnSure))
    else:
        macrof1 = 0.0
    if alnsDta>0 and alnsYawat>0 :
        microPrec = float(allIntersects)/float(alnsDta)
        microRec = float(allIntersects)/float(alnsYawat)
        microf1 = 2.0 * ((microPrec * microRec) / (microPrec + microRec))
    else:
        microPrec = 0.0
        microRec = 0.0
        microf1 = 0.0
    print("\n{} dbp - sentence pairs.".format(nbSentences))
    print("\n{} dbp - sentence pairs with alignments.".format(nbSentences))
    print("Macro-averaged prec/rec/f1 : {} / {} / {}".format(round(macroPrec,2), round(macroRec,2), round(macrof1,2)))
    print("Macro-averaged Sure-Possible prec/rec/f1 : {} / {} / {}".format(round(macroPrecOnSure,2), round(macroRecOnSure,2), round(macrof1OnSure,2)))
    print("Micro-averaged prec/rec/f1 : {} / {} / {}".format(round(microPrec,2), round(microRec,2), round(microf1,2)))


def main(args):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--aFile', help="Different alignment extraction. Values are [max | all | seg]", default="none")
    parser.add_argument('--resultsDir', help="Directory containing the results for the given model.", default="none")
    parser.add_argument('--iaAgreement', help="Annotator agreement.", action='store_true', default=False)
    parser.add_argument('--first', help="=1 do on 1st sentence, =0 do on all MINUS 1st sentence. If argument is not given will do on all.", default=2)
    parser.add_argument('--abs', help="System alignments are on abstract as a whole sequence.", action='store_true', default=False)
    arguments = parser.parse_args(args)

    fileName = ""
    if arguments.aFile == "max":
        fileName = "wa_max.txt"
    elif arguments.aFile == "all":
        fileName = "wa_all.txt"
    elif arguments.aFile == "seg":
        fileName = "sa_b1.0.txt"
    if fileName.startswith("wa_max") or fileName.startswith("sa_b"):
        max= True
    else:
        max= False

    #Reference annotation
    referenceAnnotationDir = "../../yawat/valid-abstracts-01/annot1/"
    yawatAlignments_annot1 = alnYawatRead(referenceAnnotationDir, max)

    if arguments.iaAgreement:
        yawatAlignments_annot2 = alnYawatRead("../../yawat/valid-abstracts-01/annot2/", False, True)
        computeAlnMeasures(yawatAlignments_annot2, yawatAlignments_annot1, arguments.first)
    else:
        dtaAlignments = alnDtaRead(arguments.resultsDir + fileName)
        if arguments.abs:
            #split alignments per sentence
            abstractsSentences = loadAbstractSentences(referenceAnnotationDir)
            dtaAlignments = distributeAlignmentsPerSentence(dtaAlignments, abstractsSentences)

        print(dtaAlignments)
        computeAlnMeasures(dtaAlignments, yawatAlignments_annot1, arguments.first)

    print("done.")

if __name__ == "__main__":
    main(sys.argv[1:])