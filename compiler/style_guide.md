# WSDA Delivery Style Guide

These rules apply to every narration script produced by the WSDA compiler.
They are derived from the delivery conventions in Walter's SQL Essential Training
and the C3 content standard.

## Voice and tense

- First person plural, present tense: "We click...", "We type...", "We see..."
- No second person ("you", "your", "you'll").
- No filler words: basically, essentially, simply, just, very, really.
- No learner-direction verbs: understand, learn, grasp, comprehend, concept, abstract.

## Narrate as it happens

- Every click, keystroke, or typed value must be spoken while it happens on screen.
- Do not describe a result before the action that produces it.
- Do not summarize an action after the next action has already started.

## Before/after ritual

- Before each action, state what we are about to do.
- During the action, name the exact UI element or value.
- After the action, state the immediate visible result.

## Why on screen

- Every non-obvious action must include a one-clause reason tied to what is visible.
- The reason answers "why this step exists," not "why the concept matters in general."

## Concrete scenario

- Every query, filter, or sort must trace back to a named stakeholder request.
- Do not demo mechanics for their own sake.

## Rhetorical questions

- Use rhetorical questions sparingly and only to set up a visible action.
- The answer must appear on screen within the next two beats.

## Facts once

- State each concrete fact (row count, column name, value) exactly once in the script.
- Do not repeat the same number in consecutive beats unless the screen changes.

## Validation beats

- Validation must cite a visible, re-checkable fact from the EnvironmentMap or observed state.
- Validation beats are allowed without recorded clips at generation time.
- A validation beat must assert something beyond the previous two beats; otherwise merge or drop it.
