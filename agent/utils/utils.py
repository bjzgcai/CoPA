import json
import json5
from json_repair import repair_json

def json_extract(json_string):
    if not json_string:
        return None
    json_str = repair_json(json_string)
    if json_str:
        try:
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}")
            return None
    else:
        print("未找到 JSON 代码块")
        return None
    

def parse_json(json_str):
    
    try:
        obj = json_extract(json_str)
        return obj
    except:
        pass
    try:
        obj = json5.loads(json_str)
        return obj
    except:
        print(json_str)
        raise ValueError("Invalid json object")