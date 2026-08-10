# Demo Video Script — 2-3 Minutes

## What You'll Need

1. Colab notebook open with the pipeline loaded
2. Browser tab with pandas docs open:
   https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html
3. Screen recording software:
   - Windows: Xbox Game Bar (Win + G)
   - Mac: Cmd + Shift + 5
   - Linux: OBS Studio

---

## Recording Structure (2-3 minutes)

### Introduction (15 seconds)

Say: 'Hi, I'm Muhammad Asad. This is my CAID internship project — a
document QA system for pandas that only answers from official docs
and refuses when it doesn't know.'

### Demo 1: Answered Question with Citation (45 seconds)

Show the notebook. Type or paste:

    question = "What does the 'how' parameter do in DataFrame.merge?"
    passages = retriever.retrieve(question, k=5)
    answer = generator.generate(question, passages)
    print(answer)

Say while it runs:
'When I ask about the how parameter, the system retrieves the 5
most relevant passages from pandas docs, then Qwen 2.5 3B generates
a grounded answer. Notice it cites the source doc_name at the end.'

Point to the citation in the output: 'Source: pandas.DataFrame.merge'

### Demo 2: Verify Citation Against Source (45 seconds)

Switch to the browser tab with pandas.DataFrame.merge open.

Say: 'Let's verify that citation. This is the official pandas.DataFrame.merge
page. Scrolling to the how parameter...'

Scroll to the 'how' parameter section, highlight it with mouse.

'You can see the exact wording the system referenced.
The answer matches the docs. No hallucination.'

### Demo 3: Correct Refusal on Unanswerable (45 seconds)

Back to Colab. Type or paste:

    question = "How do I use pandas to compute matrix eigenvalues?"
    passages = retriever.retrieve(question, k=5)
    answer = generator.generate(question, passages)
    print(f'Top retrieval score: {passages[0]["score"]:.3f}')
    print(f'Answer: {answer}')

Say while it runs:
'But pandas doesn't do matrix eigenvalues — that's NumPy. The system
should refuse instead of making up an answer.'

Point to output:
'Top retrieval score is around 0.3-0.5, well below our threshold
of 0.82. The system correctly returns NOT_FOUND.'

### Conclusion (15 seconds)

Say: 'On the held-back set of 60 questions, this system correctly
refuses ALL 20 unanswerable questions — 100% refuse-when-absent.
Full code, data, and report are on GitHub. Thanks for watching.'

---

## Tips

- Zoom your terminal font to ~18pt so it's readable in video
- Practice once before recording
- Speak clearly - if English feels awkward, prepare word-for-word
- Keep it under 3 min - shorter is fine

## Uploading

Save the video as demo.mp4 in the project root, then upload to:
- Google Drive (shareable link)
- Or GitHub Release (attach as file)
- Add the link to README.md
