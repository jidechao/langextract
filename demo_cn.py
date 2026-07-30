import langextract as lx
from langextract.factory import ModelConfig
import textwrap

# 1. Define the prompt and extraction rules
prompt = textwrap.dedent("""\
提取药物信息，包括药物名称、剂量、给药途径、频率和持续时间，按照它们在文本中出现的顺序提取。
""")

# 2. Provide a high-quality example to guide the model
examples_ner = [
    lx.data.ExampleData(
        text="患者被给予250毫克静脉注射头孢唑啉，每日三次，持续一周。",
        extractions=[
            lx.data.Extraction(extraction_class="dosage", extraction_text="250毫克"),
            lx.data.Extraction(extraction_class="route", extraction_text="静脉注射"),
            lx.data.Extraction(extraction_class="medication", extraction_text="头孢唑啉"),
            lx.data.Extraction(extraction_class="frequency", extraction_text="每日三次"),
            lx.data.Extraction(extraction_class="duration", extraction_text="持续一周")
        ]
    )
]



model_config = ModelConfig(
    model_id="deepseek-chat",  
    provider="OpenAILanguageModel",  # 明确指定使用OpenAI提供者
    provider_kwargs={
        "temperature": 0.1,
        "max_tokens": 2048,
        "base_url": "https://api.deepseek.com",
        "api_key": ""
    }
)

model = lx.factory.create_model(model_config)
# The input text to be processed
input_text = "患者服用了400毫克口服布洛芬，每4小时一次，持续两天。"

# Run the extraction
result_ner = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples_ner,
    model=model,
    fence_output=True,
    use_schema_constraints=False,
    resolver_params = {"enable_fuzzy_alignment": True,"fuzzy_alignment_threshold": 0.75},
    debug=True
)

print("提取的实体:")
for entity in result_ner.extractions:
    position_info = ""
    if entity.char_interval:
        start, end = entity.char_interval.start_pos, entity.char_interval.end_pos
        position_info = f" (位置: {start}-{end})"
    print(f"• {entity.extraction_class.capitalize()}: {entity.extraction_text}{position_info}")

# 保存结果
lx.io.save_annotated_documents([result_ner], output_name="医疗_ner_提取.jsonl")
print("\n基础NER结果已保存到 医疗_ner_提取.jsonl")

# 生成可视化 HTML
html_content_ner = lx.visualization.visualize("./test_output/医疗_ner_提取.jsonl")
with open("医疗_ner_可视化.html", "w", encoding='utf-8') as f:
    f.write(html_content_ner)
