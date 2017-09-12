
import os
CUR_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

NER_MODEL_PATH = CUR_DIRECTORY + 'english.all.3class.distsim.crf.ser.gz'
NER_JAR_PATH = CUR_DIRECTORY + 'stanford-ner.jar'
POS_MODEL_PATH = CUR_DIRECTORY + 'english-left3words-distsim.tagger'
POS_JAR_PATH = CUR_DIRECTORY + 'stanford-postagger.jar'