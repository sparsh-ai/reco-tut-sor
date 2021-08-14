# Starbucks Offers Recommender

## Project structure
```
.
├── artifacts
│   └── models
│       ├── BaselineModel.pth
│       └── RecommendationModel2.pth
├── code
│   ├── __init__.py
│   └── nbs
│       ├── reco-tut-sor-02-recommender-model.py
│       ├── reco-tut-sor-02-xgboost.py
│       ├── reco-tut-sor-t1-01-eda.py
│       └── reco-tut-sor-t1-02-baseline.py
├── data
│   ├── bronze
│   │   ├── portfolio.json
│   │   ├── profile.json
│   │   └── transcript.json
│   └── silver
│       ├── useractions.csv
│       └── userdata.csv
├── docs
│   └── track1_report.pdf
├── extras
│   └── images
│       ├── heatmap-event.png
│       ├── heatmap-general.png
│       └── income-age-dist-binned.png
├── LICENSE
├── notebooks
│   ├── reco-tut-sor-02-recommender-model.ipynb
│   ├── reco-tut-sor-02-xgboost.ipynb
│   ├── reco-tut-sor-t1-01-eda.ipynb
│   └── reco-tut-sor-t1-02-baseline.ipynb
├── outputs
│   ├── Recommendation-cm.png
│   ├── RecommendationEngine-cm.png
│   ├── RecommendationXGB-cm.png
│   ├── RF-model-cm.png
│   └── XGB-model-cm.png
├── README.md
├── requirements.txt
├── setup.cfg
└── setup.py  
```