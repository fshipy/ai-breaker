# AI-breaker
A web application powered by adverserial examples generation engine to trick reputed image classification models.

# Preview
<img src="https://github.com/fshipy/ai-breaker/blob/main/screenshots/Try%20Your%20Own%20Page%20(After%20Run).png" width="500">

# Run the web app
If you have a linux kernel:

```./run.sh```

Then nevigate to the URL on the console. 

Otherwises:

1. ```cd services```

2. ```pip install -r requirements.txt```

3. ```python app.py``` (Require python 3)

4. Nevigate to the URL on the console. 

Note:

1. Screenshots can be found in `screenshots` folder.
2. For the engine (try-your-own) part, using a GPU can be much more efficent than CPU only. Please be patient if using CPU only as it takes time to compute.
3. For the engine (try-your-own) part, the first run takes extra time because it may needs to download
required models from `torchvision`.

# By:

Group 17 of CS 492 (Spring 2021) Final Project: Ayman, Frank, Sahil
