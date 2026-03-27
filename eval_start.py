# 用rule function 评估多个模型的结果。
import json

"""
dataPath为数据的路径，jsonl的格式，每个json主要包括promptTitle、prompt、response字段
urls为各个模型请求的url，键值对，key为标识，value 为模型部署的url，若url为None，则数据中response为模型结果。
"""
if __name__ == "__main__":

    urls = {}
    dataPath = ""

    with open(dataPath, "r") as rf:
        for line in rf:
            item = json.loads(line.strip())



