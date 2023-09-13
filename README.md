## Install all the required packages:
pip install -r requirements.txt

## Dataset Preparation:
put the dataset in ./data/origianl directory, please guarantee there is 'src_txt' and 'tgt_txt', and each dataset contains train.json, valid.json and test.json

Example of each sample in the original dataset:
{
    "src_txt": ["i went with my dad to buy some new fish for our 50 gal .", "tank and our old siphon stopped working .", "we come back about an hour later with food and i test the new siphon ( works great . )", "it even has a clip so it can shoot the water straight into the tub .", "when i was done testing the siphon i sat the new fish in the water in the bags so they would be ready to accomodate themselves .", "30 mins later after finishing my sandwhich , i come out to see the tank is sitting with 5 inches of water inside of it .", "my dad was upset , told me to start cleaning it and to take it as a lesson .", "gathering old towels i see that a power strip sitting by the tank was smoking and was about to short circuit .", "luckily i turned it off ... with all the water around my feet .", "the water went from the dining room to the kitchen , i was lucky it was just tiling .", "i 've been wanting to clean our tank but now i have to wait two weeks just to clean it so the water rebalances .", "hopefully the new fish do n't die , i 'm surprised they did n't die by stress with the water flowing out ."], \
  "tgt_txt": "; got new fish and siphon , left the siphon in the tank , water spilled over in the second tub and nearly caused a short circuit , fish might die and ca n't clean the tank for two weeks .", \
  "src_sent_labels": [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
}

## Create new label for each dataset:
create_svs.py --soruce_path ./data/original/  \
              --save_path  ./data/new/   \
              --summary_sent_count 3 ## The parameter denotes the sentence count included in the summary, which is different for each dataset \
              --times 10 # sample times   \
              --max_sent_count 10 # sample size 


The following code example shows a workflow that posts a welcome comment on a pull request when it is opened.

```yaml annotate
# The name of the workflow as it will appear in the "Actions" tab of the GitHub repository.
name: Post welcome comment
# The `on` keyword lets you define the events that trigger when the workflow is run.
on:
  # Add the `pull_request` event, so that the workflow runs automatically
  # every time a pull request is created.
  pull_request:
    types: [opened]
# Modifies the default permissions granted to `GITHUB_TOKEN`.
permissions:
  pull-requests: write
# Defines a job with the ID `build` that is stored within the `jobs` key.
jobs:
  build:
    name: Post welcome comment
    # Configures the operating system the job runs on.
    runs-on: ubuntu-latest
    # The `run` keyword tells the job to execute a command on the runner.
    steps:
      - run: gh pr comment $PR_URL --body "Welcome to the repository!"
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          PR_URL: ${{ github.event.pull_request.html_url }}
```
