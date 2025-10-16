from typing import Optional
from textwrap import dedent

BASE_EVALUATION_PROMPT = dedent(
    """
    You are an expert evaluator. 
    Your task is to evaluate if a proposed answer matches the ground truth answer for a given question.

    You will be provided with information between special tags:
    1. The original question <question>
    2. The ground truth (correct) answer <ground_truth>
    3. A proposed answer to evaluate <proposed_answer>
    4. (Optional) the proposed answer's cited information <ground_truth_citations>

    Please evaluate the proposed answer based on the following criteria:
    - Abstain: If the proposed answer is "I don't know" or "I don't have enough information to answer this question", then the label is 'abstain'.
    - Accuracy: Does it contain the same key information as the ground truth?
    - Completeness: Does it cover all important points from the ground truth?
    - Correctness: Are there any factual errors compared to the ground truth?

    Provide your evaluation as a JSON object with the following fields:
    {{
        \"label\": \"correct\" | \"possible_correct\" | \"incorrect\" | \"abstain\",
        \"reasoning\": string // A very brief explanation of your evaluation
    }}

    Here are some examples:

    <question>
    Who was the first person to walk on the moon?
    </question>
    <ground_truth>
    Neil Armstrong was the first person to walk on the moon on July 20, 1969 during the Apollo 11 mission.
    </ground_truth>
    <proposed_answer>
    Neil Armstrong walked on the moon in 1969.
    </proposed_answer>
    <ground_truth_citations>
    Apollo 11 was a spaceflight conducted from July 16 to July 24, 1969 by the United States and launched by NASA. It marked the first time that humans landed on the Moon. Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969, at 20:17 UTC, and Armstrong became the first person to step onto the Moon's surface six hours and 39 minutes later, on July 21 at 02:56 UTC. Aldrin joined him 19 minutes later, and they spent about two and a quarter hours together exploring the site they had named Tranquility Base upon landing. Armstrong and Aldrin collected 47.5 pounds (21.5 kg) of lunar material to bring back to Earth as pilot Michael Collins flew the Command Module Columbia in lunar orbit, and were on the Moon's surface for 21 hours, 36 minutes, before lifting off to rejoin Columbia.
    </ground_truth_citations>
    Evaluation:
    {{
        \"label\": \"correct\",
        \"reasoning\": \"The proposed answer correctly identifies Neil Armstrong and the year 1969. While it omits the specific date and mission name, it captures the key information that Neil Armstrong was the first moon walker.\"
    }}

    <question>
    What is the capital of France?
    </question>
    <ground_truth>
    Paris is the capital city of France.
    </ground_truth>
    <proposed_answer>
    The capital of France is London.
    </proposed_answer>
    Evaluation:
    {{
        \"label\": \"incorrect\",
        \"reasoning\": \"The proposed answer is incorrect. It states London is the capital of France, when the ground truth clearly states that Paris is the capital.\"
    }}

    <question>
    {question}
    </question>

    <ground_truth>
    {ground_truth}
    </ground_truth>

    <proposed_answer>
    {proposed_answer}
    </proposed_answer>

    {citations}

    Your evaluation:
    """
)

CITATION_TEMPLATE = dedent(
    """
    <ground_truth_citations>
    {ground_truth_citations}
    </ground_truth_citations>
    """
)


def get_evaluation_prompt(
    question: str,
    ground_truth: str,
    proposed_answer: str,
    ground_truth_citations: Optional[list[str]] = None,
) -> str:
    if ground_truth_citations:
        citations = '\n\n'.join(ground_truth_citations)
        citations = CITATION_TEMPLATE.format(ground_truth_citations=citations)
    else:
        citations = ''
    return BASE_EVALUATION_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        proposed_answer=proposed_answer,
        citations=citations,
    )
