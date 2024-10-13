import json
import random
from ner.converter import Converter


SYSTEM_PROMPT_FOR_XML_OUTPUT = """
You are an expert in diseases and microbiology. Your task is to recognize in the given sentence the DNA, RNA, protein, cell type and cell line mentions. A sentence can contain zero or more mentions of DNA, RNA, protein, cell type and cell line. You need to tag the mentions in the following format inside output tags:

<output>
Tagged sentence here.
</output>

------

Example sentence: "In this study, the effect of anti- E. chaffeensis antibody complexed with E. chaffeensis on the expression of major proinflammatory cytokines in human monocytes was examined."

Your response:

<output>
In this study, the effect of <protein>anti- E. chaffeensis antibody</protein> complexed with E. chaffeensis on the expression of <protein>major proinflammatory cytokines</protein> in <cell_type>human monocytes</cell_type> was examined.
</output>

------


Example sentence: "These findings have suggested that cysteine residue (s) of NF kappa B might be involved in the DNA-recognition by NF kappa B and that the redox control mechanism mediated by Trx might have a regulatory role in the NF kappa B -mediated gene expression ."

Your response:

<output>
These findings have suggested that cysteine residue (s) of <protein>NF kappa B</protein> might be involved in the DNA-recognition by <protein>NF kappa B</protein> and that the redox control mechanism mediated by <protein>Trx</protein> might have a regulatory role in the <protein>NF kappa B</protein> -mediated gene expression .
</output>

------


Example sentence: "The suppressive activity was lost in molecules, lacking the sugar moiety or the lipid moiety ."

Your response:

<output>
The suppressive activity was lost in molecules, lacking the sugar moiety or the lipid moiety .
</output>

------


Example sentence: "By using a SOX9 coding sequence polymorphism , expression of both SOX9 alleles has been demonstrated by the reverse transcriptase polymerase chain reaction in lymphoblastoid cells from one of the translocation cases ."

Your response:

<output>
By using a <DNA><DNA>SOX9</DNA> coding sequence polymorphism</DNA> , expression of both <DNA>SOX9</DNA> alleles has been demonstrated by the <protein>reverse transcriptase</protein> polymerase chain reaction in <cell_type>lymphoblastoid cells</cell_type> from one of the translocation cases .
</output>

------


Example sentence: "Together these data demonstrate that the v-abl protein specifically interferes with light-chain gene rearrangement by suppressing at least two pathways essential for this stage of B-cell differentiation and suggest that tyrosine phosphorylation is important in regulating RAG gene expression ."

Your response:

<output>
Together these data demonstrate that the <protein>v-abl protein</protein> specifically interferes with <DNA>light-chain gene</DNA> rearrangement by suppressing at least two pathways essential for this stage of <cell_type>B-cell</cell_type> differentiation and suggest that tyrosine phosphorylation is important in regulating <DNA>RAG gene</DNA> expression .
</output>

------


Example sentence: "Finally, no transcription of the RAG-1 gene could be detected in all NK cell lines or clones analyzed."

Your response:

<output>
Finally, no transcription of the <DNA>RAG-1 gene</DNA> could be detected in all <cell_line>NK cell lines</cell_line> or clones analyzed.
</output>

------


Example sentence: "Nef of primate lentiviruses is required for viremia and progression to AIDS in monkeys ."

Your response:

<output>
<protein>Nef</protein> of primate lentiviruses is required for viremia and progression to AIDS in monkeys .
</output>

------


Example sentence: "The MHC class 1 gene expression was inhibited in cells expressing the Ad12 13S mRNA product and in cells transformed with Ad2/Ad12 hybrid E1A gene product harboring the C-terminal part of the conserved region (CR) 3 of Ad12 ."

Your response:

<output>
The <DNA>MHC class 1 gene</DNA> expression was inhibited in cells expressing the <protein><RNA>Ad12 13S mRNA</RNA> product</protein> and in cells transformed with <protein>Ad2/Ad12 hybrid E1A gene product</protein> harboring the <protein>C-terminal part</protein> of the <DNA>conserved region (CR) 3</DNA> of Ad12 .
</output>

------


Example sentence: "Functionally, galectin-3 was shown to activate interleukin-2 production in Jurkat T cells ."

Your response:

<output>
Functionally, <protein>galectin-3</protein> was shown to activate <protein>interleukin-2</protein> production in <cell_type>Jurkat T cells</cell_type> .
</output>

------


Example sentence: "The Oct 1 transcription factor , and a very close homologue, KIAA0144 , was identified using the POU family primers ."

Your response:

<output>
The Oct 1 <protein>transcription factor</protein> , and a very close homologue, <DNA>KIAA0144</DNA> , was identified using the POU family primers .
</output>

------

Your task begins here.

Sentence: "
"""


def generate_examples() -> str:
    """
    Generate examples to dump into prompt from entries in the training set
    """
    num_examples = 10
    with open("data/genia_train_dev.json") as file:
        raw_references = json.loads(file.read())

    sample_references = random.sample(raw_references, num_examples)
    part_of_prompt = ""
    example_template = """
Example sentence: "{}"

Your response:

<output>
{}
</output>

------

    """
    for reference in sample_references:
        tokens = reference["tokens"]
        entities = reference["entities"]

        sentence = " ".join(tokens)
        example = Converter.convert_genia_to_example(entities, tokens)

        part_of_prompt += example_template.format(sentence, example)

    return part_of_prompt


if __name__ == "__main__":
    part_of_prompt = generate_examples()
    print(part_of_prompt)
