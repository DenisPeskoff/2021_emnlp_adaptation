# Adapting Entities Across Languages and Cultures
## EMNLP Findings 2021

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

To create it, you will need (combined 50GB):

[German Matrix](https://obj.umiacs.umd.edu/adaptation/all_german_matrix.npy)

[American Matrix](https://obj.umiacs.umd.edu/adaptation/all_american_matrix.npy)


Optionally, we provide the original WikiData dump from 10-26-2020 (processed to remove everything unnecessary to Properties and Values):
https://obj.umiacs.umd.edu/adaptation/10-26-20-wikidata.jsonl



**FAQ**

**1) What enviornment do I need?**
Wikipedia/Wikidata are obviously large.  The code for Wikidata used a large RAM CPU (100+ GB) for pre-processing the data, and a GPU for computing Faiss distance.  
Since the data is provided in a .jsonl format, the code could likely be reworked to require less CPU memory if needed.  
The Faiss distance calculation is tractable (~1 hour) on a CPU.  

**2) I want to create my own WikiData dump:**

1) download a specific date from  https://dumps.wikimedia.org/wikidatawiki/entities/
You're looking for the file titled: e.g., wikidata-20210830-all.json.bz2  under a recent date.
2) process the data to get it into .jsonl format (WikiData is unsurprisingly large, so removing unrelated attributes and making it into a JSONLines format---which can be loaded item by item---is a helpful preprocessing step.  
We use https://github.com/EntilZha/wikidata-rust to make this conversion.  

