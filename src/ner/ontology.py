from typing import Dict, Any


def get_buster_ontology() -> Dict[str, Any]:
    return {
        "BUYING_COMPANY": "The company which is acquiring the target.",
        "SELLING_COMPANY": "The company which is selling the target.",
        "ACQUIRED_COMPANY": "The company target of the transaction.",
        "LEGAL_CONSULTING_COMPANY": "A law firm providing advice on the transaction, such as: government regulation, litigation, anti-trust, structured finance, tax etc.",
        "GENERIC_CONSULTING_COMPANY": "A general firm providing any other type of advice, such as: financial, accountability, due diligence, etc.",
        "ANNUAL_REVENUES": "The past or present annual revenues of any company or asset involved in the transaction.",
    }


def get_genia_ontology() -> Dict[str, Any]:
    return {
        "protein": "protein family, protein group, protein complex, protein molecule, protein subunit, protein substructure, protein domain or protein region",
        "DNA": "DNA family, DNA group, DNA molecule, DNA substructure, DNA domain, DNA region, DNA sequence, etc.",
        "RNA": "RNA family, RNA group, RNA molecule, RNA substructure, RNA domain, RNA region, RNA sequence, etc.",
        "cell_type": "any mention of specific biological cell categories or classes that occur naturally in organisms (e.g., 'T cells', 'neurons', 'fibroblasts')",
        "cell_line": "any artificially maintained, immortalized cell populations with specific laboratory names or designations (e.g., 'HeLa cells', 'K562', 'CHO cells'), typically derived from a source organism but now grown continuously in culture",
    }
