import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel


def demo1():
    tokenizer = AutoTokenizer.from_pretrained("model/chinese_sentiment")

    model = AutoModelForSequenceClassification.from_pretrained("model/chinese_sentiment")

    message = "人生该如何起头"

    tokenized_id = tokenizer.encode(message, padding=True, truncation=True, return_tensors="pt", max_length=20)
    model_out= model(tokenized_id)
    print(tokenized_id)
    print(model_out)

# 特征提取任务-不带任务输出头的任务
def dm02_test_feature_extraction():
    # 1 加载tokenizer
    my_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='model/bert-base-chinese')

    # 2 加载模型
    my_model = AutoModel.from_pretrained(pretrained_model_name_or_path='model/bert-base-chinese')

    # 3 文本转张量
    message = ['你是谁, 人生该如何起头']
    # encode_plus() 的主要功能是将原始文本转换为模型所需的输入格式，包括：
    # 分词（Tokenization）
    # 添加特殊标记（如 [CLS] 和 [SEP]）
    # 转换为 ID 序列（input_ids）
    # 生成注意力掩码（attention_mask）
    # 填充（padding）或截断（truncation）到指定长度
    msgs_tensor = my_tokenizer(text=message, max_length=30, truncation=True, padding='max_length', return_tensors='pt')
    print('msgs_tensor--->', msgs_tensor)

    # 4 给模型送数据提取特征
    my_model.eval()
    output = my_model(**msgs_tensor)
    print('不带模型头输出output--->', output)
    # last_hidden_state表示最后一个隐藏层的数据
    print('output.last_hidden_state.shape--->', output.last_hidden_state.shape)  # torch.Size([1, 30, 768])
    # pooler_output表示池化，也就是对最后一个隐藏层再进行线性变换以后平均池化的结果，分类时候使用。
    print('output.pooler_output.shape--->', output.pooler_output.shape)  # torch.Size([1, 768])

from transformers import AutoModelForMaskedLM

# 完型填空任务
def dm03_test_fill_mask():

    # 1 加载tokenizer
    modelname = "model/chinese-bert-wwm"
    # modelname = "model/bert-base-chinese"
    my_tokenizer = AutoTokenizer.from_pretrained(modelname)

    # 2 加载模型
    my_model = AutoModelForMaskedLM.from_pretrained(modelname)

    # 3 文本转张量
    input = my_tokenizer('我想明天去[MASK]家吃饭.', return_tensors='pt')
    print('input--->', input)

    # 4 给模型送数据提取特征
    my_model.eval()
    output = my_model(**input)
    print('output--->', output)
    print('output.logits--->', output.logits.shape) # [1,12,21128]

    # 5 取概率最高
    mask_pred_idx = torch.argmax(output.logits[0][6]).item()
    print('打印概率最高的字:', my_tokenizer.convert_ids_to_tokens([mask_pred_idx]))

from transformers import AutoModelForQuestionAnswering

# 阅读理解任务(抽取式问答)
def dm04_test_question_answering():

    # 1 加载tokenizer
    my_tokenizer = AutoTokenizer.from_pretrained('model/chinese_pretrain_mrc_roberta_wwm_ext_large')

    # 2 加载模型
    my_model = AutoModelForQuestionAnswering.from_pretrained('model/chinese_pretrain_mrc_roberta_wwm_ext_large')

    # 3 文本转张量
    # 文字中的标点符号如果是中文的话，会影响到预测结果 也可以去掉标点符号
    context = '我叫张三 我是一个程序员 我的喜好是打篮球'
    questions = ['我是谁？', '我是做什么的？', '我的爱好是什么？']

    # 4 给模型送数据 模型做抽取式问答
    my_model.eval()
    for question in questions:
        input = my_tokenizer(question, context, return_tensors='pt')
        print('input--->', input)
        output = my_model(**input)
        print('output--->', output)
        start, end = torch.argmax(output.start_logits), torch.argmax(output.end_logits) + 1
        answer =  my_tokenizer.convert_ids_to_tokens(input['input_ids'][0][start:end])
        print('question:', question, 'answer:', answer)

from transformers import AutoModelForSeq2SeqLM

# 文本摘要任务
def dm05_test_summarization():
    text = "BERT is a transformers model pretrained on a large corpus of English data " \
           "in a self-supervised fashion. This means it was pretrained on the raw texts " \
           "only, with no humans labelling them in any way (which is why it can use lots " \
           "of publicly available data) with an automatic process to generate inputs and " \
           "labels from those texts. More precisely, it was pretrained with two objectives:Masked " \
           "language modeling (MLM): taking a sentence, the model randomly masks 15% of the " \
           "words in the input then run the entire masked sentence through the model and has " \
           "to predict the masked words. This is different from traditional recurrent neural " \
           "networks (RNNs) that usually see the words one after the other, or from autoregressive " \
           "models like GPT which internally mask the future tokens. It allows the model to learn " \
           "a bidirectional representation of the sentence.Next sentence prediction (NSP): the models" \
           " concatenates two masked sentences as inputs during pretraining. Sometimes they correspond to " \
           "sentences that were next to each other in the original text, sometimes not. The model then " \
           "has to predict if the two sentences were following each other or not."

    # 1 加载tokenizer
    my_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="model/distilbart-cnn-12-6")

    # 2 加载模型
    my_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path='model/distilbart-cnn-12-6')

    # 3 文本转张量
    # my_tokenizer()：比encode更高级，支持批量输入，传入列表
    input = my_tokenizer([text], return_tensors='pt')
    # print('input--->', input)

    # 4 送给模型做摘要
    my_model.eval()
    output = my_model.generate(input.input_ids)
    print('output--->', output)

    # 5 处理摘要结果
    # skip_special_tokens:是否去除token前面的特殊字符
    # clean_up_tokenization_spaces:是否清理产生的空格
    summary_text = [my_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for g in output]
    print('summary_text--->', summary_text)

    # convert_ids_to_tokens 函数只能将 ids 还原为 token
    # print(my_tokenizer.convert_ids_to_tokens(output[0]))

from transformers import AutoModelForTokenClassification
from transformers import AutoConfig

# NER任务
def dm06_test_ner():
    # 1 加载tokenizer 加载模型 加载配置文件
    # https://huggingface.co/uer/roberta-base-finetuned-cluener2020-chinese
    my_tokenizer = AutoTokenizer.from_pretrained('model/roberta-base-finetuned-cluener2020-chinese')
    my_model = AutoModelForTokenClassification.from_pretrained('model/roberta-base-finetuned-cluener2020-chinese')
    config = AutoConfig.from_pretrained('model/roberta-base-finetuned-cluener2020-chinese')

    # 2 数据张量化
    inputs = my_tokenizer('我爱北京天安门，天安门上太阳升', return_tensors='pt')
    print('inputs--->', inputs.input_ids.shape, inputs.input_ids) # torch.Size([1, 17])

    # 3 送入模型 预测ner概率 每个字预测的标签概率
    my_model.eval()
    logits = my_model(inputs.input_ids).logits
    print('logits--->', logits.shape)           # torch.Size([1, 17, 32])

    # 4 对预测数据 进行显示
    input_tokens = my_tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    print('input_tokens--->', input_tokens)
    outputs = []

    for token, value in zip(input_tokens, logits[0]):
        # all_special_tokens: 获取当前分词器中定义的所有特殊标记
        if token in my_tokenizer.all_special_tokens:
            continue

        # 获得每个字预测概率最大的标签索引
        idx = torch.argmax(value).item()

        # 打印索引对应标签
        outputs.append((token, config.id2label[idx]))

    print('outputs--->', outputs)

if __name__ == '__main__':
    # demo1()
    # dm02_test_feature_extraction()
    # dm03_test_fill_mask()
    # dm04_test_question_answering()
    # dm05_test_summarization()
    dm06_test_ner()