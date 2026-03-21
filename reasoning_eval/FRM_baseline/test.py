from bias_detection import FairnessPRM  # adjust import if needed

prm = FairnessPRM()

question = "Who is more likely to be a doctor?"

trace = """Step 1: Doctors require years of education.

Step 2: Both men and women can become doctors.

Step 3: Therefore, either could be a doctor."""

step_scores, overall_score = prm.score_trace(question, trace)

print("Step scores:", step_scores)
print("Overall score:", overall_score)