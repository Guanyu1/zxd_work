# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:16:15 2020

@author: Administrator
"""

from pyhanlp import SafeJClass
import zipfile
import os
from pyhanlp.static import download, remove_file, HANLP_DATA_PATH

# 设置路径，否则会从配置文件中寻找
# HANLP_DATA_PATH = "/home/fonttian/Data/CNLP"

HANLP_DATA_PATH = "d:/programdata/anaconda3/envs/zxd_work/lib/site-packages/pyhanlp/static"
"""
获取测试数据路径，位于$root/data/textClassification/sogou-mini，
根目录由配置文件指定,或者等于我们前面手动设置的HANLP_DATA_PATH。
"""
# DATA_FILES_PATH = "textClassification/sogou-mini"
# DATA_FILES_PATH = "data/dictionary"
DATA_FILES_PATH = "data/dictionary"


def test_data_path():       #创建文件夹
    data_path = os.path.join(HANLP_DATA_PATH, DATA_FILES_PATH)
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    return data_path     


def ensure_data(data_name, data_url):  #保证数据，存在本地文件则返回，否则进行下载解压缩
    root_path = test_data_path()
    dest_path = os.path.join(root_path, data_name)
    print("111dest_path:" + dest_path)
    if os.path.exists(dest_path):
        return dest_path
    if data_url.endswith('.zip'):
        dest_path += '.zip'
    download(data_url, dest_path)
    if data_url.endswith('.zip'):
        with zipfile.ZipFile(dest_path, "r") as archive:
            archive.extractall(root_path)
        remove_file(dest_path)
        dest_path = dest_path[:-len('.zip')]
        print("dest_path:" + dest_path)
    return dest_path


NaiveBayesClassifier = SafeJClass('com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier')
IOUtil = SafeJClass('com.hankcs.hanlp.corpus.io.IOUtil')
sogou_corpus_path = ensure_data('搜狗文本分类语料库迷你版',
                                'http://hanlp.linrunsoft.com/release/corpus/sogou-text-classification-corpus-mini.zip')

# ChnSentiCorp_path = ensure_data('酒店评论情感分析', \
#          					'http://hanlp.linrunsoft.com/release/corpus/ChnSentiCorp.zip')

def train_or_load_classifier(path):
    model_path = path + '.ser'
    if os.path.isfile(model_path):
        return NaiveBayesClassifier(IOUtil.readObjectFrom(model_path))
    classifier = NaiveBayesClassifier()
    classifier.train(path)
    model = classifier.getModel()
    IOUtil.saveObjectTo(model, model_path)
    return NaiveBayesClassifier(model)


def predict(classifier, text):
    print("《%16s》\t属于分类\t【%s】" % (text, classifier.classify(text)))
    # 如需获取离散型随机变量的分布，请使用predict接口
    # print("《%16s》\t属于分类\t【%s】" % (text, classifier.predict(text)))


if __name__ == '__main__':

    classifier = train_or_load_classifier(sogou_corpus_path)
    predict(classifier, "C罗压梅西内马尔蝉联金球奖 2017=C罗年")
    predict(classifier, "英国造航母耗时8年仍未服役 被中国速度远远甩在身后")
    predict(classifier, "研究生考录模式亟待进一步专业化")
    predict(classifier, "如果真想用食物解压,建议可以食用燕麦")
    predict(classifier, "通用及其部分竞争对手目前正在考虑解决库存问题")
    
    
    print("\n 我们这里再用训练好的模型连测试一下新的随便从网上找来的几个新闻标题 \n")
    predict(classifier, "今年考研压力进一步增大，或许考研正在变成第二次高考")
    predict(classifier, "张继科被刘国梁连珠炮喊醒:醒醒！奥运会开始了。")
    predict(classifier, "福特终于开窍了！新车1.5T怼出184马力，不足11万，思域自愧不如")

###############

# """
# 获取测试数据路径，位于$root/data/textClassification/sogou-mini，
# 根目录由配置文件指定,或者等于我们前面手动设置的HANLP_DATA_PATH。
# ChnSentiCorp评论酒店情感分析
# """
# # DATA_FILES_PATH = "sentimentAnalysis/ChnSentiCorp"


# if __name__ == '__main__':
    
#     ChnSentiCorp_path = ensure_data('酒店评论情感分析', \
#          					'http://hanlp.linrunsoft.com/release/corpus/ChnSentiCorp.zip')
#     # 此处感谢网友给出的下载链接
#     # 本文示例中，如果需要使用本地资料，请通过上面的DATA_FILES_PATH变量控制
#     classifier = train_or_load_classifier(ChnSentiCorp_path)
#     predict(classifier, '距离川沙公路较近,但是公交指示不对,如果是"蔡陆线"的话,会非常麻烦.建议用别的路线.房间较为简单.')
#     predict(classifier, "商务大床房，房间很大，床有2M宽，整体感觉经济实惠不错!")
#     predict(classifier, "标准间太差 房间还不如3星的 而且设施非常陈旧.建议酒店把老的标准间从新改善.")
#     predict(classifier, "服务态度极其差，前台接待好象没有受过培训，连基本的礼貌都不懂，竟然同时接待几个客人")
    
    
#     print("\n 我们这里再用训练好的模型连测试一下我自己编的‘新的’的文本 \n")
#     predict(classifier, "服务态度很好，认真的接待了我们，态度可以的！")
#     predict(classifier, "有点不太卫生，感觉不怎么样。")