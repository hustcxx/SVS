# 1 install all the required packages:
pip install -r requirements.txt

# 2 Dataset Preparation:
put the dataset in ./data/origianl directory, please guarantee there is 'src_txt' and 'tgt_txt', and each dataset contains train.json, valid.json and test.json

# 3 Create new label for each dataset:
create_svs.py --soruce_path ./data/original/ 
              --save_path  ./data/new/ 
              --summary_sent_count 3 ## The parameter denotes the sentence count included in the summary, which is different for each dataset
              --times 10 # sample times 
              --max_sent_count 10 # sample size
