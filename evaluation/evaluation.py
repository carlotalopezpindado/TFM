import json
from factscore import FActScore
from transformers import pipeline

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
qa_pipeline = pipeline("question-answering", model=model_name)

# Load your dataset
with open("qa_dataset.json") as f:
    qa_data = json.load(f)

# Generate answers using the QA model
generated_answers = []
for item in qa_data:
    context = item['context']
    question = item['question']
    generated_answer = qa_pipeline(question=question, context=context)
    generated_answers.append({
        "question": question,
        "reference_answer": item['answer'],
        "generated_answer": generated_answer['answer']
    })

# Save generated answers
with open("generated_answers.json", "w") as f:
    json.dump(generated_answers, f, indent=4)

factscore = FActScore(model_name="deberta-large-mnli")

scores = []
for item in generated_answers:
    reference_answer = item["reference_answer"]
    generated_answer = item["generated_answer"]
    score = factscore.score(reference_answer, generated_answer)
    scores.append(score)

# Calculate average FActScore
average_factscore = sum(scores) / len(scores)

# Print or save results
print("Average FActScore:", average_factscore)

import difflib

def match_responses(reference_answer, generated_answer):
    sequence_matcher = difflib.SequenceMatcher(None, reference_answer, generated_answer)
    match_ratio = sequence_matcher.ratio()
    return match_ratio

# Calculate match ratios for each QA pair
match_ratios = []
for item in generated_answers:
    reference_answer = item["reference_answer"]
    generated_answer = item["generated_answer"]
    match_ratio = match_responses(reference_answer, generated_answer)
    match_ratios.append(match_ratio)

# Calculate average match ratio
average_match_ratio = sum(match_ratios) / len(match_ratios)

# Print or save results
print("Average Match Ratio:", average_match_ratio)

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def compute_sas(prediction, reference):
    pred_embedding = model.encode(prediction)
    ref_embedding = model.encode(reference)
    return util.pytorch_cos_sim(pred_embedding, ref_embedding).item()


import matplotlib.pyplot as plt

questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
exact_match_scores = [0.9, 0.8, 0.95, 0.85, 0.75]
f1_scores = [0.88, 0.85, 0.9, 0.8, 0.78]
sas_scores = [0.92, 0.87, 0.94, 0.82, 0.79]

x = range(len(questions))

plt.bar(x, exact_match_scores, width=0.2, label='Exact Match', align='center')
plt.bar(x, f1_scores, width=0.2, label='F1 Score', align='edge')
plt.bar(x, sas_scores, width=0.2, label='SAS', align='edge')

plt.xlabel('Questions')
plt.ylabel('Scores')
plt.title('Comparison of Evaluation Metrics')
plt.xticks(x, questions)
plt.legend()
plt.show()

plt.plot(questions, exact_match_scores, marker='o', label='Exact Match')
plt.plot(questions, f1_scores, marker='x', label='F1 Score')
plt.plot(questions, sas_scores, marker='s', label='SAS')

plt.xlabel('Questions')
plt.ylabel('Scores')
plt.title('Trend of Evaluation Metrics Across Questions')
plt.legend()
plt.show()
