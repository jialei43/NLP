from googletrans import Translator,LANGCODES

# print(LANGCODES)
def translate():
    text = '隔音太差，走廊有人说话就像在自己房间说话一样。'
    translator = Translator()
    middle = translator.translate(text, src='zh-cn', dest='en')
    print(middle.text)

    target = translator.translate(middle.text, src='en', dest='zh-cn')
    print(target.text)

if __name__ == '__main__':
    translate()