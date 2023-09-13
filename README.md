## Install all the required packages
```bash
pip install -r requirements.txt
```

## Dataset Preparation
To put the dataset[CNN/DailyMail, Reddit, XSum, WikiHow, PubMed, NYT] in ./data/original directory, please guarantee there are 'src_txt' and 'tgt_txt', and each dataset contains train.json, valid.json, and test.json

Example of each sample in the original dataset:
```json
{
 "src_txt": ["i went with my dad to buy some new fish for our 50 gal .", "tank and our old siphon stopped working .", "we come back about an hour later with food and i test the new siphon ( works great . )", "it even has a clip so it can shoot the water straight into the tub .", "when i was done testing the siphon i sat the new fish in the water in the bags so they would be ready to accomodate themselves .", "30 mins later after finishing my sandwhich , i come out to see the tank is sitting with 5 inches of water inside of it .", "my dad was upset , told me to start cleaning it and to take it as a lesson .", "gathering old towels i see that a power strip sitting by the tank was smoking and was about to short circuit .", "luckily i turned it off ... with all the water around my feet .", "the water went from the dining room to the kitchen , i was lucky it was just tiling .", "i 've been wanting to clean our tank but now i have to wait two weeks just to clean it so the water rebalances .", "hopefully the new fish do n't die , i 'm surprised they did n't die by stress with the water flowing out ."],
  "tgt_txt": "got new fish and siphon , left the siphon in the tank , water spilled over in the second tub and nearly caused a short circuit , fish might die and ca n't clean the tank for two weeks .",
  "src_sent_labels": [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
}
```

## Create a new label for each dataset
To get svs label for CNN/DailyMail, Reddit, XSum, WikiHow, PubMed, NYT datasets
```bash
python create_svs.py \
    --soruce_path ./data/original/  \
    --save_path  ./data/new/   \
    --summary_sent_count 3 "sentence count included in the summary, which is different for each dataset" \
    --times 10 "sample times"   \
    --max_sent_count 10 "sample size" 
```

              
Example of each sample in the new dataset:
```json
{
 "src_txt": ["i went with my dad to buy some new fish for our 50 gal .", "tank and our old siphon stopped working .", "we come back about an hour later with food and i test the new siphon ( works great . )", "it even has a clip so it can shoot the water straight into the tub .", "when i was done testing the siphon i sat the new fish in the water in the bags so they would be ready to accomodate themselves .", "30 mins later after finishing my sandwhich , i come out to see the tank is sitting with 5 inches of water inside of it .", "my dad was upset , told me to start cleaning it and to take it as a lesson .", "gathering old towels i see that a power strip sitting by the tank was smoking and was about to short circuit .", "luckily i turned it off ... with all the water around my feet .", "the water went from the dining room to the kitchen , i was lucky it was just tiling .", "i 've been wanting to clean our tank but now i have to wait two weeks just to clean it so the water rebalances .", "hopefully the new fish do n't die , i 'm surprised they did n't die by stress with the water flowing out ."],
  "tgt_txt": "got new fish and siphon , left the siphon in the tank , water spilled over in the second tub and nearly caused a short circuit , fish might die and ca n't clean the tank for two weeks .",
  "src_sent_labels": [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
  "svs_labels": [0.0, 0.0657628106544322, 0.07155925121390644, 0.058491514661438135, 0.16259747670660712, 0.05618844715399162, 0.0, 0.15283144349987832, 0.005011071536673256, 0.005712542486406249, 0.11591425565468534, 0.1529232245791513],

}
```



