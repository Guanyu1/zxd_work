# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 14:29:34 2020

@author: Administrator
comment：测试情感分析模型
"""
from pyhanlp import SafeJClass
import zipfile
import os
from pyhanlp.static import download, remove_file, HANLP_DATA_PATH


NaiveBayesClassifier = SafeJClass('com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier')
IOUtil = SafeJClass('com.hankcs.hanlp.corpus.io.IOUtil')

def load_model(path):
    model_path = path + '.ser'
    if os.path.isfile(model_path):
        return NaiveBayesClassifier(IOUtil.readObjectFrom(model_path))
    
    
def predict(classifier, text):
    print("《%16s》\t属于分类\t【%s】" % (text, classifier.classify(text)))
    # 如需获取离散型随机变量的分布，请使用predict接口
    # print("《%16s》\t属于分类\t【%s】" % (text, classifier.predict(text)))

if __name__ == '__main__':
    
    # ChnSentiCorp_path = ensure_data('酒店评论情感分析', \
    #      					'http://hanlp.linrunsoft.com/release/corpus/ChnSentiCorp.zip')
    # 此处感谢网友给出的下载链接
    # 本文示例中，如果需要使用本地资料，请通过上面的DATA_FILES_PATH变量控制
    ChnSentiCorp_path="d:/programdata/anaconda3/envs/zxd_work/lib/site-packages/pyhanlp/static\data/dictionary\酒店评论情感分析"
    classifier = load_model(ChnSentiCorp_path)
    print(dir(classifier))
    predict(classifier, '不怎么好')