from dataclasses import dataclass
from typing import Any
import gripql


@dataclass
class BMEGClient:
    def search(self, query: str) -> Any:
        g = gripql.Connection(
            "https://bmeg.io", credential_file="bmeg_credentials.json"
        ).graph("rc5")

        results = (
            g.query().V().hasLabel("Gene").has(gripql.eq("symbol", query)).execute()
        )
        print(results)
        print()

        print(g.listLabels())
        print()

        gids = [result["gid"] for result in results]
        for ent in g.query().V(gids).out("gene_ontology_terms").limit(10):
            print(ent["gid"], ent["data"]["definition"])


if __name__ == "__main__":
    BMEGClient().search("CD28")
