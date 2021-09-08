# Adapting Entities Across Languages and Cultures
## EMNLP Findings 2021

We provide the following files for future experiments:

Predictions are evaluated as:
python evaluate.py --golds <gold_annotations> --predictions <predictions>
e.g., python evaluate.py --golds evaluation_data/gold_American_VealeNOC.txt --predictions embedding_predictions/predictions_3cosadd_American_VealeNOC.txt
    
**Human Generated Adaptations**

**Human Evaluations of All Adaptations**

**Embedding-Based Adaptations**
The final generated data for our VealeNOC and Wikipedia sourced entities is available under embedding_predictions.
There are 6 files: German Veale, German Wiki, American Veale, American Wiki created with the 3cosadd.  And 2 files ("learned") that are trained on Wikipedia and tested on VealeNOC.   

**WikiData Adaptations**

The final generated data for our VealeNOC and Wikipedia sourced entities is available under wikidata_predictions.
There are 4 files: German Veale, German Wiki, American Veale, American Wiki created with our WikiData method.  

To create them yourself, you will need (combined 50GB):
[German Matrix](https://obj.umiacs.umd.edu/adaptation/all_german_matrix.npy)

[American Matrix](https://obj.umiacs.umd.edu/adaptation/all_american_matrix.npy)


Optionally, we provide the original WikiData dump from 10-26-2020 (processed to remove everything unnecessary to Properties and Values):
https://obj.umiacs.umd.edu/adaptation/10-26-20-wikidata.jsonl

**FAQ**

**1) What python environment do I need?**
```
pip install -r requirements.txt
```

**2) How do I produce embeddings based modulations?**

We provide ```modulate.py``` which supports both the unsupervised ```3cosadd``` and the supervised ```learned``` modulation modes. For detailed parameters run:
```
python modulate.py -h
```

* Example American to German modulation with ```3cosadd```:
```
python modulate.py \
    --input input_American_Wiki.txt \
    --output predictions_3cosadd_American_Wiki.txt \
    --src_emb vectors-en.txt \
    --trg_emb vectors-de.txt \
    --method add \
    --src_pos Germany \
    --src_neg United_States \
    --trg_pos Deutschland \
    --trg_neg USA
```

* Example German to American modulation with ```learned```:
```
python modulate.py \
    --input input_German_VealeNOC.txt \
    --output predictions_learned_German_VealeNOC.txt \
    --src_emb vectors-de.txt \
    --trg_emb vectors-en.txt \
    --method ridge \
    --train_file train_German_Wiki.txt
```

**3) How do I get my own WikiData dump?:**

1) download a specific date from  https://dumps.wikimedia.org/wikidatawiki/entities/
You're looking for the file titled: e.g., wikidata-20210830-all.json.bz2  under a recent date.
2) process the data to get it into .jsonl format (WikiData is unsurprisingly large, so removing unrelated attributes and making it into a JSONLines format---which can be loaded item by item---is a helpful preprocessing step.  
We use https://github.com/EntilZha/wikidata-rust to make this conversion.  
    
**4) What computing environment do I need?**

Wikipedia and Wikidata are obviously large.  The code for Wikidata used a large RAM CPU (100+ GB) for pre-processing the data, and a GPU for computing Faiss distance.  Since the data is provided in a .jsonl format, the code could likely be reworked to require less CPU memory if needed.  The Faiss distance calculation is tractable (~1 hour) on a CPU.  

Please contact dpeskov.work@gmail.com with any questions.  
