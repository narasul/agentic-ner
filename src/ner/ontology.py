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


def get_musicner_ontology() -> Dict[str, Any]:
    return {
        "Artist": "Names of music artists (e.g. bands, singers, composers). Additionally, movie directors, filmmakers, and so on in music recommendation context",
        "WoA": "Means 'Work of Art'. Titles of works of art (e.g. albums, tracks, playlists, soundtracks). Additionally movies and video game titles in music recommendation context",
    }


def get_astroner_ontology() -> Dict[str, Any]:
    return {
        "AstrObject": "All concepts representing astronomical objects, e.g. black holes.",
        "AstroPortion": "All concepts representing portions of astronomical objects which are not astro- nomical objects themselves, e.g. sunspots.",
        "ChemicalSpecies": "Atomic elements such as element names from the periodic table, atoms, nuclei, dark matter, e.g. Fe.",
        "Instrument": "Names of measurement instruments, including telescopes, e.g. Large Hadron Collider.",
        "Measurement": "Measured observational parameters or properties (both property and value), e.g. frequency.",
        "Method": "Abstractions which are commonly used to support the solution of the investiga- tion, e.g. minimal supersymmetrical model. In case of overlap Method is selected over all other entity types except 'ResearchProblem'",
        "Morphology": "Geometry or morphology of astronomical objects or physical phenomena, e.g. asymmetrical.",
        "PhysicalQuantity": "Properties of physical phenomena interacting, e.g. gravity.",
        "Process": "Phenomenon or associated process, e.g. Higgs boson decay.",
        "Project": "Survery or research mission, e.g. the dark energy survey",
        "ResearchProblem": "The theme of the investigation, e.g. final state hadronic interactions. In case of overlap, ResearchProblem is selected over all other entity types.",
        "SpectralRegime": "Observed or analyzed electromagnetic spectrum, e.g. mega electron volt.",
    }
