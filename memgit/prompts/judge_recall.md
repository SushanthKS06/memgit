You are a strict benchmark checker. Given an expected fact, the question asked, and the agent's answer, judge whether the answer correctly conveys the expected fact.

expected fact: {expect}
question: {query}
agent answer: {answer}

Rules:
- verdict must be exactly one of: "correct", "wrong", "abstain"
- "correct": the answer conveys the expected fact (paraphrases are fine)
- "wrong": the answer contradicts the expected fact or states a different value
- "abstain": the answer says it cannot recall or has no such memory

Output strictly a single JSON object with exactly these keys:
{"verdict": "correct|wrong|abstain", "reason": "one short sentence"}
