import json
import os

### 주의사항 : 실행되는 가상환경이 있는 경우 가상환경에 접속하여 cd 등의 명령어로 해당 파일이 있는 폴더로 이동한 후 실행해주세요###
### 파일 명 설정 ###

### Compare 시에는 Compare하고자 하는 파일과 같은 파일을 Compare하지 않는지 확인해 주세요###
### Compare 기능 사용법은 Ctrl + Shift + P → Compare Files → 비교하고자 하는 두 파일 선택입니다 ###

INPUT_PATH = "pii_masked_openai_output part1.json"  ## 작업하고자 하는 파일 명을 맞춰주세요!

print("현재 작업 폴더 :", os.getcwd())
print("이 폴더 파일들 :", os.listdir("."))
print("읽을 JSON 파일:", INPUT_PATH)

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, list):
    raise TypeError("기존 JSON의 최상위 구조가 list가 아닙니다. (지금 코드는 list 전용)")

inputs = [{"input": item.get("input", "")} for item in data]
outputs = [{"output": item.get("output", "")} for item in data]


### split 되었을때 저장될 파일 명을 지정합니다 ###
inputs_path = f"masked_split_inputs.json"          # "pii_masked_openai_output part1_inputs.json"
outputs_path = f"masked_split_outputs.json"        # "pii_masked_openai_output part1_outputs.json"

with open(inputs_path, "w", encoding="utf-8") as f:
    json.dump(inputs, f, ensure_ascii=False, indent=2)

with open(outputs_path, "w", encoding="utf-8") as f:
    json.dump(outputs, f, ensure_ascii=False, indent=2)

print("=== split 완료 ===")
print("inputs  →", inputs_path)
print("outputs →", outputs_path)
