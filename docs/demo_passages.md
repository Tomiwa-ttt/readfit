# Demo passages — paste these, in this order

Tested against the live API. Expected values recorded so you know instantly if
something is wrong on the day.

---

## 1. HARD — set lesson to 34
Expected: band **Grades 11-12**, decodability **~74%**, long flagged-word table.

```
The quantum mechanical model of the atom, notwithstanding its counterintuitive
implications, has demonstrated remarkable predictive efficacy across a broad
spectrum of experimental configurations. Researchers continue to investigate
the boundaries of this framework, particularly where classical intuitions
about locality and determinism appear to break down entirely.
```

---

## 2. EASY AND FULLY DECODABLE — keep lesson at 34
Expected: band **Grades 2-3**, decodability **100.0%**, empty flag table, green "Fully decodable" message.

```
Tim the cat sat on a mat. A big dog ran at Tim. Tim ran to the shop.
The dog is sad. Tim has a fish. Tim is not sad.
```

> This is the money shot. Same endpoint, same UI, opposite result on both scales.

---

## 3. THE NEAR-MISS — set lesson to 10
Expected: decodability **~36%**, flags `He, was, very, happy, Then, saw, away, quickly`
with the specific untaught grapheme named for each.

```
He was very happy. Then he saw a big dog and ran away quickly!
```

> Use this one to make the point that "almost decodable" is not slightly worse —
> it is the thing that trains guessing. Point at the `reason` column: `untaught: aw`,
> `untaught: ck, qu, y`. That per-word diagnosis is what no readability formula does.

---

## 4. MID-RANGE — lesson 60
Expected: band around **Grades 6-8**, decodability **~85-95%**.

```
The old lighthouse stood at the edge of the bay, its lamp turning slowly through
the fog. Sailors had trusted that light for nearly a hundred years, and the
keeper climbed the stairs every evening without fail.
```

---

## The curl command — have this typed and ready in a second terminal

```bash
curl -X POST http://<VM_IP>:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"Tim the cat sat on a mat. A big dog ran at Tim.","lesson":34}'
```

Expected response shape:

```json
{"code":200,"readability":1.2934,"band":"Grades 2-3","fk_grade":-1.84,
 "decodability":100.0,"illegal_words":[],"word_count":30,
 "lesson_label":"Lesson 34 - all short vowels + sh/ch/th digraphs",
 "model_version":"...","message":"ok"}
```

---

## Pre-flight checklist — run 15 minutes before you present

- [ ] `curl http://<VM_IP>:5000/health` → `"model_loaded": true`
- [ ] Browser tab 1: `http://<VM_IP>:5000/`
- [ ] Browser tab 2: `http://<VM_IP>:8501` (dashboard) — **or** dashboard running locally with `API_URL` set to the public IP
- [ ] Terminal open with the curl command typed but not run
- [ ] Passages 1–3 in a scratch file, ready to copy
- [ ] Fallback screen recording open in a media player, minimised
- [ ] Laptop plugged in, notifications off
