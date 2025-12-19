import json

### 합칠 파일 및 저장될 파일 명을 설정합니다 ### 

### 현재파일은 input된 처음 파일에 _merged 형식을 붙여서 저장하는 방식을 사용합니다. ###
### 구분이 필요할 시 masked_merged.json, masked_split_inputs.json, masked_split_outputs.json 처럼 지정하여  ###
### 구분해줄 수 있습니다. ###

INPUTS_PATH = "masked_split_inputs.json"
OUTPUTS_PATH = "masked_split_outputs.json"
MERGED_PATH = "pii_masked_openai_output part1_merged.json"

merged = []

with open(INPUTS_PATH, "r", encoding="utf-8") as f:
    inputs = json.load(f)

with open(OUTPUTS_PATH, "r", encoding="utf-8") as f:
    outputs = json.load(f)

if len(inputs) != len(outputs):
    raise ValueError(f"input과 output의 길이가 안맞습니다. input file과 output file이 맞는지 확인해주세요: inputs={len(inputs)}, outputs={len(outputs)}")


for i, (inp_item, out_item) in enumerate(zip(inputs, outputs)):
    # dict 형태로 처리 _ str 예외처리
    if isinstance(inp_item, dict):
        inp_text = inp_item.get("input", "")
    else:
        inp_text = str(inp_item)

    if isinstance(out_item, dict):
        out_text = out_item.get("output", "")
    else:
        out_text = str(out_item)

    merged.append({
        "input": inp_text,
        "output": out_text
    })

with open(MERGED_PATH, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print("merge 완료 ")
print("파일:", MERGED_PATH)
