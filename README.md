# Entifier
CS6120

> Extracting quality named entity relations in textual documents for visualisation and querying in a performant graph database

Output Columns of ReVerb from OpenIE:
    1. filename
    2. sentence number
    3. arg1
    4. rel
    5. arg2
    6. arg1 start
    7. arg1 end
    8. rel start
    9. rel end
    10. arg2 start
    11. arg2 end
    12. conf
    13. sentence words
    14. sentence pos tags
    15. sentence chunk tags
    16. arg1 normalized
    17. rel normalized
    18. arg2 normalized


## Installation Instructions

This requires pandas, neo4j
Install anaconda if in doubt.

## How to run reverb

```
java -Xmx512m -jar reverb.jar yourfile.txt

java -Xmx512m -jar reverb.jar yourfile.txt > outputfile.tsv
```

Note that this can cause parsing errors when done in windows