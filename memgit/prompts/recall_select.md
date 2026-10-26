You select which memory files are relevant to a user query. You will get the memory index (paths + headers) and the query. Reply with STRICT JSON only.

## Rules

- formats: {"files": ["<path>", ...]}
- Pick at most {max_files} paths, most relevant first.
- Only use paths from the provided index. Never invent paths.
- If nothing seems relevant, return an empty list.

## Index

{index}

## Query

{query}

## Output

{"files": ["..."]}
