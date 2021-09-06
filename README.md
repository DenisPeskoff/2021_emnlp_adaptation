# Adapting Entities Across Languages and Cultures
## published in **EMNLP Findings 2021**



We provide the following files for future experiments:

**Human Generated Adaptations**

**Human Evaluations of All Adaptations**

**Embedding-Based Adaptations**

**WikiData Adaptations**

The final generated data for our Veale and Wikiepedia sourced entities:
[DE Veale](https://obj.umiacs.umd.edu/adaptation/wikidata_de_veale.txt)
[DE Wikipedia](https://obj.umiacs.umd.edu/adaptation/wikidata_de_wiki.txt)
[EN Veale](https://obj.umiacs.umd.edu/adaptation/wikidata_us_veale.txt)
[EN Wikipedia](https://obj.umiacs.umd.edu/adaptation/wikidata_us_wiki.txt)

To create it, you need:

German Matrix
American Matrix


Optionally, we provide the original WikiData dump from 10-26-2020 (processed to remove everything unnecessary to Properties and Values):
https://obj.umiacs.umd.edu/adaptation/10-26-20-wikidata.jsonl



**FAQ**

**1) I want to create my own WikiData dump:**

1) download a specific date from  https://dumps.wikimedia.org/wikidatawiki/entities/
You're looking for the file titled: e.g., wikidata-20210830-all.json.bz2  under a recent date.
2) process the data to get it into .jsonl format (WikiData is unsurprisingly large, so removing unrelated attributes and making it into a JSONLines format---which can be loaded item by item---is a helpful preprocessing step.  
We use https://github.com/EntilZha/wikidata-rust to make this conversion.  

