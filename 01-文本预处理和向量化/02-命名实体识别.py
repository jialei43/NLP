from hanlp_restful import HanLPClient

hanlp = HanLPClient(
    'https://www.hanlp.com/api',
    auth=None,
    language='zh',
    verify=False
)

print(hanlp.parse(text='鲁迅, 浙江绍兴人, 五四新文化运动的重要参与者, 周树人.', tasks=['ner/msra']))