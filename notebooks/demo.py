# Demo Notebook — Pandas QA System
# Copy each section into a Colab cell for recording

# ══════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

import os
import sys
os.chdir('/content/drive/MyDrive/pandas_qa')
sys.path.insert(0, 'src')

from retrieve import Retriever
from generate import Generator

print('Loading retriever and generator (takes ~1 minute)...')
retriever = Retriever(index_dir='data/index_400')
generator = Generator()

REFUSAL_THRESHOLD = 0.82
print('Ready!')

# ══════════════════════════════════════════════════════════════
# DEMO 1: Answerable Question - Should Give Answer with Citation
# ══════════════════════════════════════════════════════════════

question = "What does the 'how' parameter do in DataFrame.merge?"

print(f'Q: {question}')
print()

passages = retriever.retrieve(question, k=5)
print(f'Top passage score: {passages[0]["score"]:.3f}')
print(f'Top passage doc:   {passages[0]["doc_name"]}')
print()

answer = generator.generate(question, passages)
print(f'A: {answer}')

# ══════════════════════════════════════════════════════════════
# DEMO 2: Unanswerable Question - Should Refuse
# ══════════════════════════════════════════════════════════════

question = "How do I use pandas to compute matrix eigenvalues?"

print(f'Q: {question}')
print()

passages = retriever.retrieve(question, k=5)
print(f'Top passage score: {passages[0]["score"]:.3f}')

if passages[0]['score'] < REFUSAL_THRESHOLD:
    print(f'Score {passages[0]["score"]:.3f} < threshold {REFUSAL_THRESHOLD}')
    print('A: REFUSED - answer not found in documentation')
else:
    answer = generator.generate(question, passages)
    print(f'A: {answer}')

# ══════════════════════════════════════════════════════════════
# DEMO 3: Another Answerable - Show Consistency
# ══════════════════════════════════════════════════════════════

question = "What is the default sorting algorithm in DataFrame.sort_values?"

print(f'Q: {question}')
print()

passages = retriever.retrieve(question, k=5)
answer = generator.generate(question, passages)
print(f'Top score: {passages[0]["score"]:.3f}')
print(f'A: {answer}')
