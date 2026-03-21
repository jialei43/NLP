import jieba.posseg as pseg

# 示例文本
text = "我爱北京天安门"

# 步骤 1: 分词并词性标注
# # 结果返回一个装有pair元组的列表, 每个pair元组中分别是词汇及其对应的词性, 具体词性含义请参照[附录: jieba词性对照表]()
words = pseg.lcut(text)
print('words->', words)
# 提取命名实体（人名、地名、组织机构名）
named_entities = []
for word, flag in words:
    if flag in ['r', 'v', 'ns']:  # r: 代词, v:动词, ns: 地名
        named_entities.append((word, flag))
print('named_entities->', named_entities)