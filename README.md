# Egyptian Arabic ASR and Question Answering

A Nile University NLP course project investigating how automatic speech
recognition errors affect downstream question answering on Egyptian Arabic.

The project uses NileTTS audio and transcripts, Whisper-based ASR,
Qwen-generated QA pairs, and two QA models: AraT5 and AraELECTRA.

[Read the research paper](_docs/NLP_Project.pdf)

## Research question

How does QA performance change when the same questions and gold answers
are evaluated against reference transcripts and ASR-generated transcripts?

The questions and gold answers remain fixed; the context changes.

## Project stages

| Folder | Purpose |
| --- | --- |
| `0_ASR_NOOO` | Earlier Whisper comparisons and saved transcription metrics. The folder retains its historical name. |
| `1_ASR` | Whisper fine-tuning attempts, transcription code, and the final saved ASR evaluation CSV. |
| `2_QA_GENERATION` | QA-generation development notebooks, four-part training generation, checkpoints, filtering, and final QA datasets. |
| `3_FINE_TUNED_QA_MODEL` | Final AraT5 and AraELECTRA experiments and saved evaluation outputs. |
| `_docs` | Research paper and earlier document versions. |

Notebook numbering preserves the original development history.
Repeated numbers identify alternative versions or related workflows.

## Dataset and splits

The original NileTTS metadata contains:

- 8,571 training audio examples.
- 950 evaluation audio examples.

For QA generation, the original training metadata was split into:

- 7,713 training-source transcripts.
- 858 validation-source transcripts.

The split uses `random_state=42` and stratification by domain and speaker.
The 950 evaluation examples supply the QA test-source transcripts.

After QA generation and filtering:

| Split | QA pairs | Audio examples represented |
| --- | ---: | ---: |
| Train | 12,877 | 5,962 |
| Validation | 1,398 | 665 |
| Test | 1,554 | 738 |

An audio example is represented when at least one generated QA pair survives
filtering. These coverage counts differ from the original split sizes.

## QA generation and preparation

Qwen generates Arabic questions and short answers from reference transcripts.
Answers are checked against their reference contexts.

Training generation was divided into four independent parts using
`np.array_split`. Each device ran Qwen with a different `PART_ID`.
This distributes generation work across transcripts.

Main training-generation notebooks:

- `5_niletts_train_qwen_qa_generation_4_devices.ipynb`: configurable template.
- `5_niletts_llm_qa_train_part_3.ipynb`: the corresponding part-3 notebook.

The final data-preparation notebooks are:

- `6_checked_merge_niletts_train_qa_parts.ipynb`: merges the four filtered
  training parts and normalizes labels.
- `7_niletts_llm_qa_val.ipynb`: prepares validation data from its checkpoint.
- `8_niletts_llm_qa_test.ipynb`: prepares test data from its checkpoint.

The earlier `6_merge_niletts_train_qa_parts.ipynb` is an alternative merge
version; its output is not an input to the checked merge.

Earlier notebooks preserve Gemini and Qwen development attempts.
Some also contain prototype AraT5 training sections.
The final QA-model experiments are in `3_FINE_TUNED_QA_MODEL`.

## Canonical data files

| Data | Location |
| --- | --- |
| ASR evaluation results | `1_ASR/niletts_asr_eval_finetuning_transcriptions_metrics.csv` |
| Training QA | `2_QA_GENERATION/1_train_data/niletts_llm_qa_train.csv` |
| Validation QA | `2_QA_GENERATION/2_val_data/niletts_llm_qa_val.csv` |
| Test QA | `2_QA_GENERATION/3_test_data/niletts_llm_qa_test_reference.csv` |

Both final QA-model notebooks read these files directly.

Each model's output folder retains its own prepared test table, predictions,
comparison results, summary metrics, domain results, and error examples.

## Saved ASR results

Values below are means of the per-audio WER and CER values in the saved CSVs.

| ASR setting | WER | CER |
| --- | ---: | ---: |
| Whisper Base | 0.536309 | 0.172574 |
| MAdel fine-tuned Whisper-small | 0.372143 | 0.157277 |
| Project-fine-tuned Whisper-small, as reported in the paper | 0.116839 | 0.038978 |

The final ASR row summarizes all 950 evaluation examples.
ASR averages in the QA summary files are weighted by QA rows and can differ
because an audio example may have several questions.

## Saved QA results

Both models were trained using reference contexts and evaluated twice:
once with reference contexts and once with ASR contexts.

EM and F1 values are reported on a 0–1 scale.

| Model | Reference EM | Reference F1 | ASR EM | ASR F1 |
| --- | ---: | ---: | ---: | ---: |
| AraT5 | 0.0051 | 0.1547 | 0.0026 | 0.1469 |
| AraELECTRA | 0.2876 | 0.5558 | 0.1622 | 0.4858 |

AraELECTRA's F1 decreases by approximately 7.00 percentage points when
ASR contexts replace reference contexts.

The models differ in architecture and starting checkpoints:
AraT5 starts from `UBC-NLP/AraT5-base`, while AraELECTRA starts from an
existing Arabic QA checkpoint. This comparison does not isolate the effect
of architecture alone.

## Working with the repository

### Inspect saved results

The repository includes prepared datasets and prediction CSVs.
These can be inspected without downloading model weights or running training.

The final merge and filtering notebooks can also operate on the included
CSV checkpoints without loading an LLM.

### Configure execution environments

Python 3.11 is a suggested starting point for the compatibility configurations.

Use separate environments or sessions for ASR and QA.
Install platform-appropriate PyTorch builds before installing the remaining
requirements. ASR also uses TorchAudio.

- [PyTorch installation](https://pytorch.org/get-started/locally/)
- [TorchAudio compatibility information](https://docs.pytorch.org/audio/stable/installation.html)

For ASR:

```bash
python -m pip install -r requirements-asr.txt
```

For QA generation and training:

```bash
python -m pip install -r requirements-qa.txt
```

Open the notebooks in VS Code or Jupyter and select the corresponding kernel.

Whisper comparison experiments using `openai-whisper` also require the
FFmpeg command-line application.

### Execute selected stages

1. Obtain NileTTS audio when running ASR. The ASR setup cells download it to
   `NileTTS-dataset`; metadata copies for QA are already in
   `2_QA_GENERATION/0_data`.
2. For new QA generation, run the selected generation sections. Use the
   training template with `PART_ID` values 0–3 for the four training parts.
3. Prepare final CSVs with the checked merge notebook and notebooks 7 and 8.
4. Use the two notebooks in `3_FINE_TUNED_QA_MODEL` for the final model workflows.

Development notebooks contain alternative attempts and prototype training
sections. They are not a single sequence intended for indiscriminate
“Run All” execution.

The standalone `_generate_text_from_eval_sherif.ipynb` helper requires
`MODEL_PATH` to be configured explicitly.

## Included and excluded artifacts

Included:

- Notebook source and retained historical outputs.
- NileTTS metadata copies.
- QA-generation CSV checkpoints and filtered parts.
- Final QA datasets and missing-source reports.
- Saved predictions and evaluation summaries.
- Model configuration and tokenizer files.
- Project documents.

Excluded through `.gitignore`:

- Trained model weight files.
- Generated model ZIP archives.
- Intermediate training checkpoint directories.
- Downloaded audio and processed dataset caches.
- Local credentials and Python environments.

Configuration and tokenizer files do not include the trained model weights.
Loading a project-trained model requires its weights to be supplied separately.

## Reproducibility notes

- Paths, imports, dependency settings, and selected error handling were
  updated during repository cleanup.
- Saved training and generation outputs reflect the original runs.
  Heavy generation and training were not rerun after cleanup.
- The checked merge and final validation/test preparation logic were
  verified against the included CSVs. Their results matched the saved final
  tables, including row and column order.
- Final reference-answer offsets were checked, and no audio-file overlap
  was found between the prepared QA train, validation, and test sets.
- The ASR training notebook uses the evaluation split for validation and
  checkpoint selection; it is not an untouched held-out ASR test set.
- The exact checkpoint/export chain that produced the final ASR CSV was not
  recorded in the original transcription helper.
- Generated QA pairs are marked as requiring manual review.
- Dependency files provide compatibility settings rather than complete,
  historically verified environment lock files.

## Models and data sources

- [NileTTS dataset](https://huggingface.co/datasets/KickItLikeShika/NileTTS-dataset)
- [Whisper](https://github.com/openai/whisper)
- [MAdel Egyptian Arabic Whisper-small](https://huggingface.co/MAdel121/whisper-small-egyptian-arabic)
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [AraT5-base](https://huggingface.co/UBC-NLP/AraT5-base)
- [AraELECTRA Arabic QA checkpoint](https://huggingface.co/ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA)

## Team

Nile University — Department of Computer Science

- Haya Osama Nabil
- Abdelrahman Ahmed Fouad
- Sherif Ahmed Mahmoud
- Omar Abdelrahman Hossieny