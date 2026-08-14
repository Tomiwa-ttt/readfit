# ReadFit — Will This Text Fit This Reader?

**AASD 4016 — Full Stack Data Science Systems**
Applied A.I. Solutions Development, George Brown College

Moyinoluwa Ajibola (101610734) · Adelabu Emmanuel (101571971) · Ameer Hamza Manzoor (101630890)
Diwakar Saini (101592939) · Lalit Kumar (101572828) · Innocent Amos Mchechesi (101628648)

---

## 1. Introduction and problem statement

A Grade 1 teacher works with roughly twenty-five students who are all at different
points in a phonics sequence. To practise reading independently, a child needs
*decodable* text — a passage in which every word is built only from letter–sound
patterns (grapheme–phoneme correspondences, or GPCs) the child has already been
taught, plus a small permitted list of irregular sight words.

The rule is strict rather than approximate. If a passage contains a word the child
cannot sound out, the child falls back on guessing from the picture, the first
letter, or the context. That guessing habit is precisely what structured literacy
exists to eliminate, because it collapses as soon as books get harder. An
*almost*-decodable text is therefore not slightly less useful than a decodable one;
it actively trains the wrong reflex.

Existing tools answer a different question. Readability formulas and commercial
levelling systems tell a teacher what grade band a text belongs to. That is a
class-level answer to a student-level question. No widely available tool answers:
*can this specific child, at this specific point in their phonics sequence, decode
this specific passage — and if not, which words fail?*

ReadFit answers both halves. It returns a trained-model readability prediction for
the passage and a deterministic, word-by-word decodability report against a chosen
phonics lesson.

## 2. Why this problem

Ontario mandates systematic, explicit phonics instruction. The Ontario Human Rights
Commission's *Right to Read* inquiry led to a full revision of the Grade 1–8 Language
curriculum, released in June 2023 and implemented that September. Every Grade 1–2
classroom in the province now runs a structured phonics sequence, which means every
one of those classrooms has the text-matching problem described above.

The current workaround is manual. Teachers write passages by hand from a legal
vocabulary that may contain only forty or so words early in the sequence — a
genuinely hard constrained-writing puzzle, usually done outside working hours.
Purchased decodable readers follow a sequence that rarely matches the one a given
board or teacher uses, and they run out.

A check that takes two seconds and names the failing words is worth real
classroom time.

## 3. Literature review

**Past available projects.** Readability estimation has a long history of
formula-based approaches: Flesch Reading Ease and Flesch–Kincaid Grade Level (1975),
the New Dale–Chall formula, SMOG, and the Automated Readability Index. All operate on
surface counts — words per sentence, syllables per word, and in Dale–Chall's case
membership of a fixed easy-word list. Lexile provides a proprietary commercial
levelling scale used widely in North American schools.

**Similar projects.** The CommonLit Readability Prize (Kaggle, 2021) reframed
readability as a supervised machine learning problem, asking entrants to predict a
human-derived difficulty score for literary passages. That competition produced the
CLEAR corpus, which this project trains on. Competition entries were dominated by
large transformer models fine-tuned on the excerpt text.

**Drawbacks of the existing work.**

1. *Every one of these systems outputs a single number for a whole passage.* None
   identifies which individual words will fail which individual learner. A grade
   band cannot tell a teacher that `quickly` is the problem because `ck`, `qu` and
   vowel-`y` have not been taught yet.
2. *Formulas use only surface statistics.* Our own benchmarking quantifies the cost:
   fitted against human judgement on held-out data, Flesch–Kincaid grade level alone
   achieves R² = 0.267. It explains roughly a quarter of the variance in how hard
   humans actually find a passage.
3. *No published tool models grapheme-level decodability against a phonics
   sequence.* This is the gap ReadFit occupies, and it is the reason the system pairs
   a learned model with a deterministic verifier rather than using either alone.
4. *Transformer approaches from the Kaggle competition are impractical here.* They
   require GPU inference or large CPU memory footprints, which rules them out for a
   free-tier cloud VM.

## 4. Data

**Source.** CLEAR Corpus version 6.01 (CommonLit Ease of Readability), developed by
CommonLit with Georgia State University and published openly. The corpus contains
4,724 reading passage excerpts drawn from grades 3–12 material.

**Target variable.** `BT_easiness` — a Bradley–Terry score derived from a large body
of pairwise human comparisons in which readers judged which of two excerpts was
easier. Higher values mean easier text.

This choice matters for the integrity of the result. `BT_easiness` is **not** a
formula. Had we trained against Flesch–Kincaid, we would have been training a model
to approximate a function we already possessed exactly, and beating it would have
been meaningless. Training against human judgement makes the comparison against
Flesch–Kincaid a genuine test: both are attempts to predict the same human quantity,
and one wins.

**Licensing.** Corpus metadata is distributed under an MIT license. Individual
excerpts vary — most are public domain, some carry Creative Commons terms. We use
the excerpts for model training only and do not redistribute them.

**Preprocessing.** Rows missing `Excerpt` or `BT_easiness` were dropped, as were
excerpts under twenty words. The corpus was split 80/20 into training and held-out
test sets with a fixed random seed of 42; 3,779 training and 945 test excerpts.

**Secondary data assets.** A hand-built grapheme–phoneme correspondence table
covering 60 graphemes, and a reconstruction of a UFLI-style phonics scope and
sequence at four checkpoints (lessons 10, 34, 60 and 98). The sequence is a
reconstruction from public descriptions, not the official UFLI document; this is a
stated limitation.

## 5. Methodology and model benchmarking

### 5.1 Features

Two feature families are concatenated into a single 3,015-dimensional
representation.

*Fifteen handcrafted linguistic features*, standardised: word count, sentence count,
mean and maximum sentence length, mean word length, mean syllables per word,
proportion of polysyllabic words, proportion of long words, type–token ratio,
proportion of common high-frequency words, comma rate per sentence, Flesch Reading
Ease, Flesch–Kincaid Grade Level, proportion of capitalised tokens, and apostrophe
rate.

*Three thousand TF-IDF features* over unigrams and bigrams, minimum document
frequency 3, with sublinear term-frequency scaling. Vocabulary is fitted on the
training split only.

Syllable counting uses a vowel-group heuristic with a silent-`e` correction. It is
not linguistically perfect, but it is deterministic and consistent, which is what a
feature requires.

### 5.2 Candidate models and results

Five candidates plus an uninformed floor were evaluated on the same held-out split.

| Model | RMSE ↓ | MAE | R² |
|---|---|---|---|
| Baseline — predict the mean | 1.0703 | 0.8816 | −0.005 |
| Flesch–Kincaid grade level only | 0.9142 | 0.7292 | 0.2667 |
| Ridge — 15 linguistic features | 0.8467 | 0.6746 | 0.3710 |
| RandomForest — linguistic features | 0.8441 | 0.6642 | 0.3749 |
| GradientBoosting — linguistic features | 0.8340 | 0.6620 | 0.3897 |
| **Ridge — linguistic + TF-IDF (tuned, α = 1.0)** | **0.7061** | **0.5521** | **0.5625** |

The second row is the meaningful comparison. Flesch–Kincaid grade level is the
formula schools use today; fitted linearly against the human target it reaches
RMSE 0.9142. The deployed model reaches 0.7061 — a **22.8% reduction in error** —
and lifts R² from 0.267 to 0.563.

Two secondary observations are worth recording. First, the linguistic features alone
already beat the formula (0.8467 versus 0.9142), so part of the gain comes simply
from using fifteen surface statistics rather than two. Second, adding lexical content
via TF-IDF produces the largest single jump (0.8340 → 0.7061), which indicates that
*which words* a passage uses carries substantial information beyond sentence and
syllable length.

### 5.3 Tuning and optimisation

The deployed model's regularisation strength was selected by `GridSearchCV` over
α ∈ {0.3, 1.0, 3.0, 10.0, 30.0} with five-fold cross-validation on the training split,
scored by negative RMSE. α = 1.0 won. The full cross-validation curve is written to
`models/metrics.json` and plotted on the dashboard.

Model selection was also constrained by the deployment target. Sentence-embedding
approaches were considered and rejected: the free-tier VM has one gigabyte of RAM,
and the tuned Ridge model plus its vectoriser occupy 136 KB in total, loading in
well under a second.

### 5.4 Feature importance

The RandomForest was retained purely to produce interpretable importances. The
strongest drivers, in order, are Flesch Reading Ease, maximum sentence length,
proportion of polysyllabic words, type–token ratio, and proportion of common words.
Maximum sentence length outranking mean sentence length is a useful practical
finding: one very long sentence appears to affect perceived difficulty more than a
uniformly moderate average.

## 6. The decodability verifier

The verifier is deliberately model-free. It segments each word into graphemes using
greedy longest-match against an ordered table of 60 GPCs, with a silent-`e` rule,
then tests set inclusion of those graphemes against the lesson's taught set. A word
passes if it is on the lesson's sight-word list or if all of its graphemes have
been taught.

The design property that matters is that it **fails closed**. If a word cannot be
segmented at all, it is reported as undecodable rather than waved through. A false
alarm costs a teacher a glance; a missed undecodable word costs a child a guess.

Worked example at lesson 10, which teaches short `a`/`i`/`o` and core consonants:

> *He was very happy. Then he saw a big dog and ran away quickly!*

scores 35.7% decodable, and the system names each failure with its cause —
`He` (untaught `e`), `very` (untaught `er`, `v`, `y`), `saw` (untaught `aw`),
`quickly` (untaught `ck`, `qu`, `y`). This per-word diagnosis is the capability no
readability formula provides.

## 7. System architecture and API

The service is a single Flask application serving nine endpoints.

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | HTML client — passage box and lesson selector |
| POST | `/analyze` | HTML result page |
| POST/GET | `/predict` | **JSON data service** — the model offered as an API |
| GET | `/benchmark` | Model comparison, tuning curve, feature importances |
| GET | `/metrics` | Aggregate statistics over logged predictions |
| GET | `/history` | Recent predictions |
| GET | `/lessons` | Available phonics lessons |
| GET | `/health` | Liveness and model-load status |
| GET | `/reload` | Re-read model artifacts without a restart |

The model and vectoriser are loaded once at process start rather than per request.
Model loading is wrapped so that a missing artifact degrades the service to
verifier-only operation rather than preventing startup — this is what allowed the
container to be deployed to the cloud before the model existed.

Every prediction is written to SQLite with its timestamp, word count, lesson,
predicted readability, band, decodability and Flesch–Kincaid score. This log is the
data source for the dashboard's live views and would be the basis of any future
retraining.

## 8. Deployment

The deployment follows the pipeline taught in Modules 1–5.

1. **Notebook → script.** Exploratory analysis in a Jupyter notebook, then
   `src/train.py` as a runnable script.
2. **Script → Flask.** `app.py` loads the serialised model and serves inference.
3. **Containerisation.** `python:3.11-slim` base image; dependencies from
   `requirements.txt`; a `.dockerignore` excluding the 3.3 MB corpus and notebooks,
   keeping the build context near 180 KB. The container runs under gunicorn rather
   than the Flask development server, which prints an explicit production warning.
4. **Registry.** Image tagged and pushed to Docker Hub.
5. **Cloud.** Google Cloud Platform Compute Engine, `e2-micro` instance in
   `us-east1`. A firewall rule with source range `0.0.0.0/0` and target tag
   `demo-1` opens TCP 5000 and 8501; the instance carries the matching network tag.
   Docker installed over SSH, image pulled, container run with
   `--restart on-failure`.
6. **Verification.** Public endpoints tested with curl, POSTMAN, and the browser
   client, from a network outside the deployment environment.

Python 3.11 was used rather than the 3.8 shown in the module slides because
scikit-learn 1.6 requires a modern interpreter.

## 9. Dashboard

A Streamlit application provides the stakeholder-facing view. It reads exclusively
over HTTP from the deployed API's public address — never from local files — so that
what it displays is by construction the state of the deployed model.

It presents a KPI row (passages analysed, mean readability, mean decodability, mean
Flesch–Kincaid grade), the model benchmark comparison with the winning model
highlighted, the hyper-parameter tuning curve, the top ten feature importances, a
scatter of readability against decodability across all logged predictions, the
distribution of predicted grade bands, and a live scoring control that submits a
passage to production and displays the response.

The API address is a sidebar input rather than a hard-coded constant, so the
dashboard can be re-pointed between local and deployed instances without a rebuild.

## 10. Challenges and limitations

**Compressed timeline.** The build was executed in a single working day rather than
across the planned three weeks. This forced explicit scope cuts, which are recorded
here rather than hidden: no sentence-embedding benchmark row, no continuous
integration, no generation feature, a single phonics sequence rather than several,
and no hand-validation of the segmenter against a pronunciation dictionary.

**Memory ceiling.** One gigabyte of RAM on the free-tier instance ruled out
transformer-based approaches outright and directed model selection toward compact
linear models.

**Segmenter accuracy is unmeasured.** The GPC table is hand-built and validated only
by inspection on a small set of words. The original design called for validation
against CMUdict by comparing grapheme counts to phoneme counts; this was cut. Some
words are certainly mis-segmented. Because the verifier fails closed, the error
direction is conservative, but the rate is unknown.

**The phonics sequence is a reconstruction.** Lesson checkpoints approximate a
UFLI-style progression from public descriptions rather than reproducing an official
scope and sequence document.

**Bands are relative, not absolute.** `BT_easiness` is an abstract scale. The five
grade bands are quantile cut-points calibrated on the training distribution, so they
describe position within this corpus rather than validated grade equivalence.

**Substantial variance remains unexplained.** R² = 0.563 means the model accounts for
a little over half the variance in human readability judgement. Readability is
genuinely hard, and this figure should not be presented as a solved problem.

**No user testing.** No teacher has used the system. Validation is entirely against
the corpus and against published decodable text.

## 11. Future work

Validate the segmenter against CMUdict and report its measured accuracy. Add
additional phonics sequences (Wilson, Orton–Gillingham variants) as data files rather
than code. Introduce a teacher feedback endpoint so that flagged words enter a review
queue and corrections to the GPC table can be measured before and after. Explore a
distilled sentence-embedding model if a larger instance becomes available. Most
importantly, put the tool in front of practising teachers, whose priorities will
almost certainly differ from those assumed here.

## 12. Contributions

> **Note before completing this section:** the paragraph below states what actually
> happened. Replace `[NAME]` with your own name and adjust any line where a teammate
> genuinely did contribute — an accurate record that shows uneven effort is safer
> than a tidy one that does not match reality, and this section is the one a grader
> is most likely to test in Q&A.

Roles for this project were assigned across six members as follows: cloud
deployment and infrastructure (Moyinoluwa Ajibola), model training and benchmarking
(Lalit Kumar), API and feature engineering (Ameer Hamza Manzoor), decodability
verifier and demonstration content (Diwakar Saini), dashboard and visualisation
(Innocent Amos Mchechesi), and documentation, ML Canvas and presentation
(Adelabu Emmanuel).

Owing to scheduling and availability constraints in the final week, the
implementation was carried out by **[NAME]**, who executed all six workstreams:
provisioning and securing the GCP instance, training and benchmarking the model,
building the Flask API and feature pipeline, implementing the decodability verifier,
building the Streamlit dashboard, and preparing the report and presentation. The role
assignments above are recorded because they reflect the agreed division of work and
the sections each member is responsible for presenting.

Generative AI was used throughout, in line with the course's Generative AI Usage
Guidelines, for architecture design, code generation, benchmarking setup and
document drafting. The completed AI Usage Declaration form accompanies this
submission. All reported metrics were produced by executing the code in this
repository; no result in this report is estimated or reproduced from another source.

## 13. References

- Ontario Human Rights Commission, *Right to Read Inquiry Report* — Curriculum and Instruction.
- Crossley, S. et al., *The CommonLit Ease of Readability (CLEAR) Corpus*. https://github.com/scrosseye/CLEAR-Corpus
- CommonLit, *Introducing the CLEAR Corpus*. https://www.commonlit.org/blog/introducing-the-clear-corpus-an-open-dataset-to-advance-research-28ff8cfea84a/
- CommonLit Readability Prize, Kaggle, 2021. https://www.kaggle.com/c/commonlitreadabilityprize
- Kincaid, J.P. et al. (1975), *Derivation of New Readability Formulas*.
- Dorard, L. (2019), *The Machine Learning Canvas*, draft v0.1.
- UFLI Foundations, University of Florida Literacy Institute. https://ufli.education.ufl.edu/foundations/
