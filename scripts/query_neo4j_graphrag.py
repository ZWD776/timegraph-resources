from _run_wrapper import run_relative


if __name__ == "__main__":
    print("warning: `scripts/query_neo4j_graphrag.py` is a legacy Neo4j GraphRAG entrypoint kept for regression checks.")
    run_relative("legacy/query/query_neo4j_graphrag.py")
