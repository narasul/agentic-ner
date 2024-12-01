from ner.grader import Grader, Feedback
from ner.prompts import LLM_GRADER_PROMPT
from ner.helper import extract_tag


class LLMGrader(Grader):
    def grade(self, prediction: str) -> Feedback:
        raw_judgement = self.llm_client.get_llm_response(
            prediction, system_prompt=LLM_GRADER_PROMPT
        )
        score = extract_tag(raw_judgement, "score")
        feedback = extract_tag(raw_judgement, "feedback")

        print(f"Raw judgement: {raw_judgement}\n\n")
        print(f"Score: {score}\n\n")
        print(f"Feedback: {feedback}\n\n")

        try:
            grade = int(score)
        except Exception:
            grade = 10

        return Feedback(grade=grade, feedback=feedback)
