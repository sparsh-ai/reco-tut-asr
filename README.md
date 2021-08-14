# Amazon Stationary Recommender

## Project structure
```
.
├── code
│   ├── __init__.py
│   ├── metrics.py
│   └── nbs
│       ├── reco-tut-asr-01-evaluation-version-1.py
│       ├── reco-tut-asr-01-evaluation-version-2.py
│       ├── reco-tut-asr-99-01-non-personalised-and-stereotyped-recommendation.py
│       ├── reco-tut-asr-99-02-recommendation-metrics-rating-evaluation.py
│       ├── reco-tut-asr-99-03-content-based-recommendations.py
│       ├── reco-tut-asr-99-04-user-user-cf.py
│       ├── reco-tut-asr-99-05-item-item-cf.py
│       ├── reco-tut-asr-99-06-similarity-metrics-for-cf.py
│       ├── reco-tut-asr-99-07-offline-evaluation.py
│       ├── reco-tut-asr-99-08-matrix-factorization.py
│       ├── reco-tut-asr-99-09-hybrid-recommendations.py
│       └── reco-tut-asr-99-10-metrics-calculation.py
├── data
│   ├── bronze
│   │   ├── items.csv
│   │   ├── other
│   │   └── ratings.csv
│   ├── gold
│   │   ├── cbf.csv
│   │   ├── item-item.csv
│   │   ├── mf.csv
│   │   ├── pers-bias.csv
│   │   └── user-user.csv
│   └── silver
│       ├── items.csv
│       └── ratings.csv
├── docs
│   ├── notes
│   │   ├── Month 1 - HW1 Notes.pdf
│   │   ├── Month 1 - HW 2 Notes.pdf
│   │   ├── Month 2 - HW 1 Notes.pdf
│   │   ├── Month 2 - HW2 Notes.pdf
│   │   ├── Month 3 - HW1 Notes.pdf
│   │   ├── Month 3 - HW2 Quizz Long.pdf
│   │   └── Month 4 - HW1 Notes.pdf
│   ├── papers
│   │   ├── An Algorithmic Framework for Performing Collaborative Filtering.pdf
│   │   ├── A Survey of Serendipity in Recommender Systems.pdf
│   │   ├── Collaborative Recommendations Using Item-to-Item Similarity Matrix.pdf
│   │   ├── Evaluating Collaborative Filtering Recommender Systems.pdf
│   │   ├── Explaining Collaborative Filltering Recommendations.pdf
│   │   ├── Improving Recommendation Lists Through Topic Diversification.pdf
│   │   ├── Is Seeing Believing - How Recommender Systems Interfaces bias User's Opinions.pdf
│   │   └── Item-Based Collaborative Filtering Algorithm.pdf
│   └── slides
│       ├── Coursera - 0 - Slides Introduction to RS.pdf
│       ├── Coursera 10 - Capstone Assignment.pdf
│       ├── Coursera - 1 - Non-Personalised and Stereotyped Rec.pdf
│       ├── Coursera - 2 - Content Based Recommendation.pdf
│       ├── Coursera - 3 - Slides User User Collaborative Filtering.pdf
│       ├── Coursera - 4 - Slides Item Item Collaborative Filtering.pdf
│       ├── Coursera - 5 - Additional Item and List‐Based Metrics.pdf
│       ├── Coursera - 6 - Top-N Protocols and Unary Data.pdf
│       ├── Coursera - 7 - Online-Evaluation-and-User-Studies.pdf
│       ├── Coursera - 8 - AB-Testing.pdf
│       └── Coursera - 9 - Evaluation-Cases.pdf
├── images
│   ├── notebook1_image1.jpeg
│   ├── notebook1_image2.jpg
│   ├── notebook2_image1.jpg
│   ├── notebook2_image2.jpg
│   ├── notebook2_image3.png
│   ├── notebook3_image1.jpg
│   ├── notebook4_image1.png
│   ├── notebook5_image1.jpeg
│   ├── notebook5_image2.png
│   ├── notebook6_image1.png
│   ├── notebook6_image2.png
│   ├── notebook6_image3.png
│   ├── notebook7_image1.png
│   ├── notebook7_image20.png
│   ├── notebook7_image3.png
│   ├── notebook7_image4.png
│   ├── notebook7_image50.png
│   ├── notebook7_image6.png
│   └── notebook7_image7.png
├── LICENSE
├── model
└── notebooks
    ├── reco-tut-asr-01-evaluation-version-1.ipynb
    ├── reco-tut-asr-01-evaluation-version-2.ipynb
    ├── reco-tut-asr-99-01-non-personalised-and-stereotyped-recommendation.ipynb
    ├── reco-tut-asr-99-02-recommendation-metrics-rating-evaluation.ipynb
    ├── reco-tut-asr-99-03-content-based-recommendations.ipynb
    ├── reco-tut-asr-99-04-user-user-cf.ipynb
    ├── reco-tut-asr-99-05-item-item-cf.ipynb
    ├── reco-tut-asr-99-06-similarity-metrics-for-cf.ipynb
    ├── reco-tut-asr-99-07-offline-evaluation.ipynb
    ├── reco-tut-asr-99-08-matrix-factorization.ipynb
    ├── reco-tut-asr-99-09-hybrid-recommendations.ipynb
    └── reco-tut-asr-99-10-metrics-calculation.ipynb
```