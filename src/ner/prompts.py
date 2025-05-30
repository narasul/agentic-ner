import json
from typing import Any, Dict, List, Tuple

from ner import ontology
from ner.agents.agent_config import AgentConfig
from ner.clients.claude_client import AnthropicClient, ClaudeFamily
from ner.eval.dataset import Example, NERDataset
from ner.helper import extract_tag
from ner.ontology import get_buster_ontology, get_genia_ontology


SYSTEM_PROMPT_FOR_XML_OUTPUT = """
You are an expert in diseases and microbiology. Your task is to recognize in the given text the DNA, RNA, protein, cell type and cell line mentions. A text can contain zero or more mentions of DNA, RNA, protein, cell type and cell line. 

Some notes:

- Tag protein family, protein group, protein complex, protein molecule, protein subunit, protein substructure, protein domain or protein region as 'protein'.
- Tag DNA family, DNA group, DNA molecule, DNA substructure, DNA domain, DNA region, DNA sequence, etc. as 'DNA'.
- Tag RNA family, RNA group, RNA molecule, RNA substructure, RNA domain, RNA region, RNA sequence, etc. as 'RNA'.

You need to tag the mentions in the following format inside output tags:

<output>
Tagged text here.
</output>
------

Example text: "In this study, the effect of anti- E. chaffeensis antibody complexed with E. chaffeensis on the expression of major proinflammatory cytokines in human monocytes was examined."

Your response:

<output>
In this study, the effect of <protein>anti- E. chaffeensis antibody</protein> complexed with E. chaffeensis on the expression of <protein>major proinflammatory cytokines</protein> in <cell_type>human monocytes</cell_type> was examined.
</output>

------


Example text: "These findings have suggested that cysteine residue (s) of NF kappa B might be involved in the DNA-recognition by NF kappa B and that the redox control mechanism mediated by Trx might have a regulatory role in the NF kappa B -mediated gene expression ."

Your response:

<output>
These findings have suggested that cysteine residue (s) of <protein>NF kappa B</protein> might be involved in the DNA-recognition by <protein>NF kappa B</protein> and that the redox control mechanism mediated by <protein>Trx</protein> might have a regulatory role in the <protein>NF kappa B</protein> -mediated gene expression .
</output>

------


Example text: "The suppressive activity was lost in molecules, lacking the sugar moiety or the lipid moiety ."

Your response:

<output>
The suppressive activity was lost in molecules, lacking the sugar moiety or the lipid moiety .
</output>

------


Example text: "By using a SOX9 coding sequence polymorphism , expression of both SOX9 alleles has been demonstrated by the reverse transcriptase polymerase chain reaction in lymphoblastoid cells from one of the translocation cases ."

Your response:

<output>
By using a <DNA><DNA>SOX9</DNA> coding sequence polymorphism</DNA> , expression of both <DNA>SOX9</DNA> alleles has been demonstrated by the <protein>reverse transcriptase</protein> polymerase chain reaction in <cell_type>lymphoblastoid cells</cell_type> from one of the translocation cases .
</output>

------


Example text: "Together these data demonstrate that the v-abl protein specifically interferes with light-chain gene rearrangement by suppressing at least two pathways essential for this stage of B-cell differentiation and suggest that tyrosine phosphorylation is important in regulating RAG gene expression ."

Your response:

<output>
Together these data demonstrate that the <protein>v-abl protein</protein> specifically interferes with <DNA>light-chain gene</DNA> rearrangement by suppressing at least two pathways essential for this stage of <cell_type>B-cell</cell_type> differentiation and suggest that tyrosine phosphorylation is important in regulating <DNA>RAG gene</DNA> expression .
</output>

------


Example text: "Finally, no transcription of the RAG-1 gene could be detected in all NK cell lines or clones analyzed."

Your response:

<output>
Finally, no transcription of the <DNA>RAG-1 gene</DNA> could be detected in all <cell_line>NK cell lines</cell_line> or clones analyzed.
</output>

------


Example text: "Nef of primate lentiviruses is required for viremia and progression to AIDS in monkeys ."

Your response:

<output>
<protein>Nef</protein> of primate lentiviruses is required for viremia and progression to AIDS in monkeys .
</output>

------


Example text: "The MHC class 1 gene expression was inhibited in cells expressing the Ad12 13S mRNA product and in cells transformed with Ad2/Ad12 hybrid E1A gene product harboring the C-terminal part of the conserved region (CR) 3 of Ad12 ."

Your response:

<output>
The <DNA>MHC class 1 gene</DNA> expression was inhibited in cells expressing the <protein><RNA>Ad12 13S mRNA</RNA> product</protein> and in cells transformed with <protein>Ad2/Ad12 hybrid E1A gene product</protein> harboring the <protein>C-terminal part</protein> of the <DNA>conserved region (CR) 3</DNA> of Ad12 .
</output>

------


Example text: "Functionally, galectin-3 was shown to activate interleukin-2 production in Jurkat T cells ."

Your response:

<output>
Functionally, <protein>galectin-3</protein> was shown to activate <protein>interleukin-2</protein> production in <cell_type>Jurkat T cells</cell_type> .
</output>

------


Example text: "The Oct 1 transcription factor , and a very close homologue, KIAA0144 , was identified using the POU family primers ."

Your response:

<output>
The Oct 1 <protein>transcription factor</protein> , and a very close homologue, <DNA>KIAA0144</DNA> , was identified using the POU family primers .
</output>

------

Your task begins here.

text: "
"""


SYTEM_PROMPT_FOR_FEEDBACK = """
{MAIN_SYSTEM_PROMPT}
{text}

------
Here is your previous output: {previous_output}

An expert in microbiology assessed your previous output and gave the following feedback:

{feedback}


Please provide a new output based on the feedback you received. Do not change your output if the feedback was very positive and didn't point out any mistakes. If you disagree with the feedback, you can still keep your original output. If you agree with the feedback, please provide a new output based on the feedback you received. Provide your output inside <output> tag similar to the examples above.

------

text: 

"""


LLM_GRADER_PROMPT = """
You are an expert in diseases and microbiology. Your task is to grade/judge/evaluate the quality of Named Entity Recognition model given its output. The model's job is to recognize in the given text the DNA, RNA, protein, cell type and cell line mentions. A text can contain zero or more mentions of DNA, RNA, protein, cell type and cell line. 

Some notes to prevent confusion:

- protein family, protein group, protein complex, protein molecule, protein subunit, protein substructure, protein domain or protein region is tagged as 'protein'.
- DNA family, DNA group, DNA molecule, DNA substructure, DNA domain, DNA region, DNA sequence, etc. is tagged as 'DNA'.
- RNA family, RNA group, RNA molecule, RNA substructure, RNA domain, RNA region, RNA sequence, etc. is tagged as 'RNA'.

Remember! The model should only tag 5 entities: 'RNA', 'DNA', 'protein', 'cell_type' and 'cell_line'. Do not propose adding a different entity type in your feedback.


The model output will have the following format:

<model_output>
Functionally, <protein>galectin-3</protein> was shown to activate <protein>interleukin-2</protein> production in <cell_type>Jurkat T cells</cell_type> .
</model_output>


You will start by asking questions about the entities inside xml tags in the model output. For example, in the text above there are 3 entities: <protein>galectin-3</protein>, <protein>interleukin-2</protein> and <cell_type>Jurkat T cells</cell_type>. You can ask questions like "What is 'galectin-3'?" or "What is 'interleukin-2'?" or "What is 'Jurkat T cells'?". And, if you know the answer, answer your own questions.

- What is 'interleukin-2'?
- Interleukin-2 (IL-2) is a type of protein called a cytokine that plays an important role in the immune system.

- What is 'galectin-3'?
- Galectin-3 is a protein that is encoded by the LGALS3 gene in humans.

- What is 'Jurkat T cells'?
- Jurkat T cells are a laboratory cell line (not a natural cell type) derived from the blood of a 14-year-old boy with T cell leukemia in the 1970s. They are immortalized human T lymphocytes that can grow continuously in the laboratory.

Now that you established the entities in the model output, you can start grading the model output. You will grade the model output based on the correctness of the entities recognized and the quality of the tags. For example, in the example above you already established through question-and-answers that 'interleukin-2' is a protein, 'Jurkat T cells' is a cell type, and 'galectin-3'is a protein. So the Named Entity Recognition model output is correct in recognizing these entities. However, if the model output had tagged 'Jurkat T cells' as a protein, the model output would be partly incorrect. As your job is to grade/judge the model's output you must give the model a score and a feedback. The score should be a number between 0 and 10, where 0 is the worst possible score and 10 is the best possible score. The feedback should be a text explaining why you gave the score you gave. Your final output should be in the following format:

<output>
<score>10</score>
<feedback>Yes, the entities in the given text were correctly recognized and tagged. The model output is of high quality.</feedback>

-----

Let's go through another example. You are give the following model output:

<model_output>
The <DNA>MHC class 1 gene</DNA> expression was inhibited in cells expressing the <protein><RNA>Ad12 13S mRNA</RNA> product</protein> and in cells transformed with <cell_line>Ad2/Ad12 hybrid E1A gene product</cell_line> harboring the <protein>C-terminal part</protein> of the <RNA>conserved region (CR) 3</RNA> of Ad12 .
</model_output>

Let's start asking questions:

- What is 'MHC class 1 gene'? How does it related to DNA, RNA, protein, cell type and cell line?
- MHC class I genes are DNA sequences located on chromosome 6 in humans, within the Major Histocompatibility Complex (MHC) region.

- What is 'Ad12 13S mRNA product'?
- The Ad12 13S mRNA product refers to a protein encoded by the adenovirus type 12 (Ad12) early region 1A (E1A) gene.

- What is 'Ad12 13S mRNA'?
- Ad12 13S mRNA is a specific messenger RNA associated with human adenovirus type 12 (Ad12), which plays a crucial role in the virus's ability to transform cells and induce gene expression.

(notice that 'Ad12 13S mRNA' is tagged as an RNA entity, while 'Ad12 13S mRNA product' is tagged as a protein entity. According to the question and answer above this is correct.)


- What is 'Ad2/Ad12 hybrid E1A gene product'?
- The Ad2/Ad12 hybrid E1A gene product refers to a chimeric protein that combines portions of the E1A proteins from adenovirus type 2 (Ad2) and adenovirus type 12 (Ad12).

- What is 'C-terminal part'?
- The C-terminal part of a protein refers to the end of the polypeptide chain containing a free carboxyl group (-COOH), located opposite to the N-terminus.

- What is 'conserved region (CR) 3'?
- Conserved Region 3 (CR3) in adenovirus type 12 (Ad12) is a crucial part of the early region 1A (E1A) protein that functions as a major transcriptional activation domain.


We can see from question and answers above the model tagged 'MHC class 1 gene', 'Ad12 13S mRNA', 'C-terminal part' and 'Ad12 13S mRNA product' correctly. However, the model tagged 'Ad2/Ad12 hybrid E1A gene product' as a cell line, which is incorrect. The model also tagged 'conserved region (CR) 3' as an RNA entity, which is incorrect. The model tagged 3 entities correctly and 2 entities incorrectly so let's assign it a score of 6. The feedback should explain why the model got a score of 6. Your final output will be in the following format:

<output>
<score>6</score>
<feedback>The 'conserved region (CR) 3' is not related to RNA. It is a part of a protein so should've been tagged as 'protein'. 'Ad2/Ad12 hybrid E1A gene product' is not a cell line, it is also a protein so should be tagged as 'protein'</feedback>
</output>

-----

Your task begins here.

Model output:

"""


NER_META_PROMPT = """
You are an expert prompt engineer specializing in creating precise Named Entity Recognition (NER) prompts for specific domains given an ontology.

Your task is to generate a comprehensive NER prompt that enables accurate entity tagging across different types of entities within a given domain.

Prompt Generation Instructions:

1. Domain Understanding
- Carefully analyze the provided domain and entity descriptions
- Identify the specific context and nuances of the domain
- Consider the range and variations of potential entity mentions

2. Entity Type Specification
For each entity type, provide:
- Precise definition
- Inclusion criteria (what to tag)
- Exclusion criteria (what to avoid)
- Variations and edge cases to consider if any

3. Tagging Guidelines
- Specify the exact XML tagging format
- Provide rules for handling:
  * Partial matches
  * Contextual ambiguity
  * Compound or complex entities
  * Nested or overlapping entities
- Specify that the given text should not be summarized under any circumstances! The NER agent should keep the input text as is and only change the text where tagging is needed.
Repeating again, the agent should not summarize or shorten the document. Only tags should be added if necessary. Nothing should be removed from the text.
- The agent should only tag the sentence inside <text_to_tag> tag. It can be give a context around the text to tag which should be taken into consideration by the agent, but the text outside <text_to_tag> tag should never be tagged.
- The agent should never add, remove or fix punctuation in the given sentence to tag.

4. Prompt Structure
The generated prompt must include:
- Clear role description
- Specific domain context
- Detailed entity type definitions
- Comprehensive tagging instructions
- Multiple varied example texts demonstrating correct tagging
- Explanation of tagging rationale

5. Querying domain expert
- The agent can query a domain expert at any time if it doesn't have enough information about certain entities it comes accross.
- The agent can do so by outputing its questions inside <search> tags.
- The questions inside <search> tags will be answered by a domain expert agent. This information can be used by NER agent for more accurate tagging.
- The <search> tags are not for asking clarifications, it is to retrieve factual answers about words/entities/concepts inside the given text to tag.

6. Output Requirements
- The final prompt alongside examples should be given inside <prompt> tag
- Before giving the final output provide your explanation first
- Must handle zero or multiple entity mentions per text
- Put '{examples_marker}' string inside the prompt so that it can be replaced by actual examples in runtime.
- Put '{external_addition_marker}' string inside the prompt so that it can be replaced by specific instructions externally.
- The prompt should specify that the final response(of NER agent) should be inside <output> tag.
- The prompt should specify that the agent can use help of the domain expert by asking a question inside <search> tags.
- If search tag is used the agent should not also use <output> tag. It should wait for an answer from domain expert before giving final answer.

-----
Below is one example input ontology and output prompt:

--- Example starts here ----

Domain: Genomics, microbiology research

Ontology: 
```{
    "protein": "protein family, protein group, protein complex, protein molecule, protein subunit, protein substructure, protein domain or protein region",
    "DNA": "DNA family, DNA group, DNA molecule, DNA substructure, DNA domain, DNA region, DNA sequence, etc.",
    "RNA": "RNA family, RNA group, RNA molecule, RNA substructure, RNA domain, RNA region, RNA sequence, etc.",
    "cell_type": "any mention of specific biological cell categories or classes that occur naturally in organisms (e.g., 'T cells', 'neurons', 'fibroblasts')",
    "cell_line": "any artificially maintained, immortalized cell populations with specific laboratory names or designations (e.g., 'HeLa cells', 'K562', 'CHO cells'), typically derived from a source organism but now grown continuously in culture",
}
````

Examples: 
```
<example>
<text_to_tag>Together these data demonstrate that the v-abl protein specifically interferes with light-chain gene rearrangement by suppressing at least two pathways essential for this stage of B-cell differentiation and suggest that tyrosine phosphorylation is important in regulating RAG gene expression .</text_to_tag>

Expected model response:

<output>
Together these data demonstrate that the <protein>v-abl protein</protein> specifically interferes with <DNA>light-chain gene</DNA> rearrangement by suppressing at least two pathways essential for this stage of <cell_type>B-cell</cell_type> differentiation and suggest that tyrosine phosphorylation is important in regulating <DNA>RAG gene</DNA> expression .
</output>
</example>

<example>
Some context to help here.
<text_to_tag>Finally, no transcription of the RAG-1 gene could be detected in all NK cell lines or clones analyzed.</text_to_tag>
Some more context to help with tagging here.

Expected model response:

<search>Tell me about RAG-1 gene</search>

<output>
Finally, no transcription of the <DNA>RAG-1 gene</DNA> could be detected in all <cell_line>NK cell lines</cell_line> or clones analyzed.
</output>
</example>

<example>
<text_to_tag>Nef of primate lentiviruses is required for viremia and progression to AIDS in monkeys .</text_to_tag>

Expected model response:

<search>What is Nef in genomics?</search>
<search>Is viremia a protein?</search>
</example>

```

<prompt>
You are an expert in diseases and microbiology. Your task is to recognize in the given text the DNA, RNA, protein, cell type and cell line mentions. A text can contain zero or more mentions of DNA, RNA, protein, cell type and cell line. 

Some notes:

- Tag protein family, protein group, protein complex, protein molecule, protein subunit, protein substructure, protein domain or protein region as 'protein'.
- Tag DNA family, DNA group, DNA molecule, DNA substructure, DNA domain, DNA region, DNA sequence, etc. as 'DNA'.
- Tag RNA family, RNA group, RNA molecule, RNA substructure, RNA domain, RNA region, RNA sequence, etc. as 'RNA'.

You should only tag the sentence inside <text_to_tag> tags. Never provide extra information related to the text outside <text_to_tag> tags.

You can ask about concepts and entities you don't know about by outputing your questions inside <search> tags.

You need to tag the mentions in the following format inside output tags:

<output>
Tagged text here.
</output>

If <search> tags are used do not output the final answer inside <output> tag.
------
Examples:

{examples_marker}

{external_addition_marker}

Text to process: 
</prompt>

--- Example ends here ----

Your task starts here.

Domain: {domain}

Ontology: 
```{ontology}
```

Examples: {examples}

"""


REVIEWER_AGENT_PROMPT_TEMPLATE = """
You are an an AI agent tasked with reviewing the quality of Named Entity Recognition agent given its output in {domain} domain. The NER agent's job is to recognize the following entities in the given text: 

```
{ontology}
```

A text can contain zero or more mentions of the entities.

Remember! The NER agent should only tag the following entities: {entities}. Do not propose adding a different entity type in your feedback.
The NER agent should only tag the entities inside <text_to_tag> tags. Do not propose to tag anything in the surrounding context in your feedback.


The NER agent's output will have the following format:

```<output>{example}</output>```

- Start by thinking out loud about each tagged entity inside XML tags in the agent's output.

- If you find the output of the NER agent satisfactory after your thought process you must output the following token: ```APPROVED!```

- Otherwise, if you have a feedback for the NER agent after your thought process you must output your feedback inside <feedback> tags.

- The NER agent can object to your feedback by giving its objection inside <objection> tags. If you find the objection valid you can approve the result by outputing ```APPROVED!``` tag. Otherwise you can provide further feedback by giving your response inside <feedback> tags.

- If you are not confident enough about the given entity you can query a researcher with access to internet and other external resources by outputting your question inside <search> tag. You will be given a short answer by the researcher after deep dive on the topic. You can output multiple questions inside multiple <search> tags if you have more than one question.

- Only output either <feedback>, <search> or APPROVED!. Do not output all of them. Your approval is only considered valid if you didn't provide feedback or questions to the researcher.

----

Let's go over an example in microbiology and genomics domain:

--- Example starts here ---

Entities:
```
{genia_ontology}
```

NER agent's output:
```<output>
Functionally, <protein>galectin-3</protein> was shown to activate <protein>interleukin-2</protein> production in <cell_type>Jurkat T cells</cell_type> .
</output>```

You will start by asking questions about the entities inside xml tags in NER agent's output. For example, in the text above there are 3 entities: <protein>galectin-3</protein>, <protein>interleukin-2</protein> and <cell_type>Jurkat T cells</cell_type>. You can ask questions like "What is 'galectin-3'?" or "What is 'interleukin-2'?" or "What is 'Jurkat T cells'?". And, if you know the answer, answer your own questions.

- What is 'interleukin-2'?
- I know that Interleukin-2 (IL-2) is a type of protein called a cytokine that plays an important role in the immune system.

- What is 'galectin-3'?
- Let me ask the researcher: 
<search>What is galectin-3 in microbiology context?</search>

- What is 'Jurkat T cells'?
- Let me ask the domain expert:
<search>What is 'Jurkat T cells'?</search>


Now I am waiting for answer from the researcher and I can't continue without the answers. Do not output anything else if you are waiting for an anwer from the researcher. Do not output APPROVED! or feedback inside <feedback> tags if you need an answer from the researcher before proceeding.

-----

Let's go through another example. You are give the following output:

NER agent's output:
```<output>
The <DNA>MHC class 1 gene</DNA> expression was inhibited in cells expressing the <protein><RNA>Ad12 13S mRNA</RNA> product</protein> and in cells transformed with <cell_line>Ad2/Ad12 hybrid E1A gene product</cell_line> harboring the <protein>C-terminal part</protein> of the <RNA>conserved region (CR) 3</RNA> of Ad12 .
</output>```


Let's start asking questions:

- What is 'MHC class 1 gene'? How does it related to DNA, RNA, protein, cell type and cell line?
- MHC class I genes are DNA sequences located on chromosome 6 in humans, within the Major Histocompatibility Complex (MHC) region.

- What is 'Ad12 13S mRNA product'?
- The Ad12 13S mRNA product refers to a protein encoded by the adenovirus type 12 (Ad12) early region 1A (E1A) gene.

- What is 'Ad12 13S mRNA'?
- Ad12 13S mRNA is a specific messenger RNA associated with human adenovirus type 12 (Ad12), which plays a crucial role in the virus's ability to transform cells and induce gene expression.

(notice that 'Ad12 13S mRNA' is tagged as an RNA entity, while 'Ad12 13S mRNA product' is tagged as a protein entity. According to the question and answer above this is correct.)


- What is 'Ad2/Ad12 hybrid E1A gene product'?
- The Ad2/Ad12 hybrid E1A gene product refers to a chimeric protein that combines portions of the E1A proteins from adenovirus type 2 (Ad2) and adenovirus type 12 (Ad12).

- What is 'C-terminal part'?
- The C-terminal part of a protein refers to the end of the polypeptide chain containing a free carboxyl group (-COOH), located opposite to the N-terminus.

- What is 'conserved region (CR) 3'?
- Conserved Region 3 (CR3) in adenovirus type 12 (Ad12) is a crucial part of the early region 1A (E1A) protein that functions as a major transcriptional activation domain.


We can see from question and answers above the model tagged 'MHC class 1 gene', 'Ad12 13S mRNA', 'C-terminal part' and 'Ad12 13S mRNA product' correctly. However, the model tagged 'Ad2/Ad12 hybrid E1A gene product' as a cell line, which is incorrect. The model also tagged 'conserved region (CR) 3' as an RNA entity, which is incorrect. Hence your feedback will be like following:

<feedback>The 'conserved region (CR) 3' is not related to RNA. It is a part of a protein so should've been tagged as 'protein'. 'Ad2/Ad12 hybrid E1A gene product' is not a cell line, it is also a protein so should be tagged as 'protein'</feedback>

--- Example ends here ---

Your task begins here.

"""


RESEARCH_AGENT_PROMPT = """
You are a wise AI agent with access to a magical 'search' tool that can return factually correct responses to the given query.

Your job is to answer the queries coming to you using this 'search' tool.

If the query coming to you is complex or asking about a few different things/concepts/facts you can pass several queries to the search tool by breaking down the initial complex query.

Your response should be maximum of 3 sentences for each simple query. The shorter the better.

You should output your response inside <answer> tags.

Examples:

Query: "Who is the president of USA right now? When was he elected?"

You will invoke the search tool with following queries:
- "Who is the president of USA?"
- "When was the current presedent of USA elected?"

Based on the output of the 'search' tool you will answer the question and output your response inside <answer> tags:

<answer>
Joe Biden is the current president of the USA.

Current president of the USA was elected in 2020.

--

Let's look at another query: "Is Earth flat or round?"

For this query you don't have to invoke the 'search' tool since you already know the answer to this simple query:

<answer>The earch is sphere!</answer>

--

Let's see another example query: "What is 'Ad12 13S mRNA product'?\n What is 'Jurkat T cells'?"


Seems like this is genomics which is a hard topic where it is very easy for an AI like me to give factually incorrect answer to.
Let me invoke the 'search' tool with the followin queries:
- What is 'Ad12 13S mRNA product'?
- What is 'Jurkat T cells'?

...
'search' tool invoked and I can see its response.
...

Okay, according to the 'search' tool response here is my answer:

<answer>
- The Ad12 13S mRNA product refers to a protein encoded by the adenovirus type 12 (Ad12) early region 1A (E1A) gene.
- Jurkat T cells are a laboratory cell line (not a natural cell type) derived from the blood of a 14-year-old boy with T cell leukemia in the 1970s. They are immortalized human T lymphocytes that can grow continuously in the laboratory.
</answer>

--- we are done with examples ---

Your task starts here:
"""


def get_system_prompt_with_feedback(
    text: str, previous_output: str, feedback: str
) -> str:
    return SYTEM_PROMPT_FOR_FEEDBACK.format(
        MAIN_SYSTEM_PROMPT=SYSTEM_PROMPT_FOR_XML_OUTPUT,
        text=text,
        previous_output=previous_output,
        feedback=feedback,
    )


def generate_example_part_of_prompt(examples: List[Example]) -> str:
    part_of_prompt = ""
    example_template = """
<example>
{}
<text_to_tag>{}</text_to_tag>
{}

Your response:

<output>
{}
</output>
</example>
------

    """
    for example in examples:
        part_of_prompt += example_template.format(
            example.left_context,
            example.text_to_tag,
            example.right_context,
            example.tagged_text,
        )

    return part_of_prompt


def get_ner_prompt(
    domain: str, ontology: Dict[str, Any], examples: List[Example], debate: bool = True
) -> str:
    llm_client = AnthropicClient(ClaudeFamily.SONNET_35_V2)

    examples_part_of_prompt = generate_example_part_of_prompt(examples)
    META_PROMPT = NER_META_PROMPT.replace("{domain}", domain)
    META_PROMPT = META_PROMPT.replace("{ontology}", json.dumps(ontology))
    META_PROMPT = META_PROMPT.replace("{examples}", examples_part_of_prompt)

    prompter_output = llm_client.get_llm_response(
        "Please provider the prompt inside <prompt> tags", system_prompt=META_PROMPT
    )
    prompt = extract_tag(prompter_output, "prompt")
    prompt = prompt.replace("{examples_marker}", examples_part_of_prompt)

    additions = "Do not tag anything outside <text_to_tag> tags! Repeating again, you should only tag the sentence given inside <text_to_tag> tags. The rest of the text is give to you as context."

    if debate:
        additions += "\nYou might receive a feedback inside <feedback> tags from another agent. Address the feedback promptly by adjusting your output inside <output> tags(provide new and corrected output inside <output> tags). If you do not agree with the feedback you can output your concerns about the feedback inside <objection> tags."

    prompt = prompt.replace("{external_addition_marker}", additions)

    return prompt


def get_reviewer_prompt(domain: str, ontology: Dict[str, Any], examples: List[Example]):
    tagged_text_example = examples[0].tagged_text
    entities = ", ".join(list(ontology.keys()))

    prompt = REVIEWER_AGENT_PROMPT_TEMPLATE.replace(
        "{genia_ontology}", json.dumps(get_genia_ontology())
    )
    prompt = prompt.replace("{example}", tagged_text_example)
    prompt = prompt.replace("{domain}", domain)
    prompt = prompt.replace("{ontology}", json.dumps(ontology))
    prompt = prompt.replace("{entities}", entities)

    return prompt


def get_agent_config(domain: str, ontology: Dict[str, Any], examples: List[Example]):
    ner_prompt = get_ner_prompt(domain, ontology, examples)
    print(f"NER system prompt: {ner_prompt}")

    reviewer_prompt = get_reviewer_prompt(domain, ontology, examples)
    print(f"Reviewer system prompt: {reviewer_prompt}")

    return AgentConfig(
        tagger_system_prompt=ner_prompt,
        reviewer_system_prompt=reviewer_prompt,
        researcher_system_prompt=RESEARCH_AGENT_PROMPT,
    )


if __name__ == "__main__":
    domain = "Finance, Business, Law"
    ontology = get_buster_ontology()
    dev_dataset = NERDataset.from_buster("FOLD_1")

    reviewer_prompt = get_reviewer_prompt(domain, ontology, dev_dataset.get_examples(5))

    print(reviewer_prompt)
