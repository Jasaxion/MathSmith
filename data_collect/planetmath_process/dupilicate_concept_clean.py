import json

def deduplicate_concepts(input_path, output_path):
    seen = {}
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            if not line.strip():
                continue
            entry = json.loads(line)
            concept_lower = entry['Concept'].strip().lower()
            if concept_lower not in seen:
                seen[concept_lower] = entry

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for item in seen.values():
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')

# 示例使用
input_file = '/home/mnt/zhanshaoxiong/pipeline/RL_math_model/collection_concept_and_detail/collect_planetmath_grouped.jsonl'
output_file = '/home/mnt/zhanshaoxiong/pipeline/RL_math_model/collection_concept_and_detail/collect_planetmath_grouped_deduplicated.jsonl'
deduplicate_concepts(input_file, output_file)
