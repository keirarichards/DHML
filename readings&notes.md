## Readings and Notes
### Algorithm types:
- k-means clustering
- neural networks
- probabilistic clustering
- Dimensionality reduction
- principle component analysis
- Single value decomposition

### Week 1:

[**IBM Machine learning basics**](https://www.ibm.com/cloud/learn/machine-learning)
- Branch of AI trying to imitate human learning “gradually improving accuracy”
- Used in recommendation engines and self driving cars
- trained to make classifications or predictions in “data mining projects”  (isnt that a bad word?)
- “classical” (non deep) machine learning relies more on human intervention in which humans identify “features” in a structured data set to learn
- Deep Learning: sub field of machine learning. “scalable machine learning” eliminates the need for human “feature extraction”  which allows for larger data sets to be processed. 
- Can use labeled or unstructured data (text and images) and automatically determine features that distinguish categories 
- used in computer vision, speech recognition and natural language processing
Neural Networks: also known as artificial neural networks (ANNs), subfield of deep learning. -Uses “node layers”,  each node an artificial neuron, organized in at least 3 layers (input, hidden, and output)
- nodes connect to each other with “associated weight and threshold” in which if a node value is above a certain threshold, it is activated, sending data to the next layer (if/else trees?)
- more than three layers is classified as a “deep learning algorithm” vs a basic neural network
- Supervised Learning- use of labeled data set to train predictive or classifying algorithms
- Unsupervised Learning- use of unlabeled datasets, discovers similarities and difference on its own
- Reinforcement learning- behavioral ML through trial and error, sequences of success reinforce pathways
- overfitting: a statistical model fits its source data too well, making it useless for prediction. Caused by training on one dataset too long. Low error rates in training set and high in test data
- underfitting: model can’t determine the relationship between input and outcome, high bias and low variance 

[**Machine learning for the humanities**](https://latex-ninja.com/2020/10/25/machine-learning-for-the-humanities-a-very-short-introduction-and-a-not-so-short-reflection/):
- Need to understand what questions your asking and be able to explain/interpret results
- frameworks hide how ML process data to be aware of downfalls/issues
- ML program learns from experience of a task while performance is measured (E= user marking junk, T= marking junk P= amount of emails correctly labled)
- makes functions(models) for incomplete data, it fills in points until there is a continuous function
- In DH, results depend on how well model fits reality and data is represented
- test function by putting in Xs (features) to see if Y(target value) fits model
Selecting training data/measuring success
- training data needs to represent the data we are trying to process later ie sampling
- automatic functions for accuracy rate , how well model performs on new test data
- minimize error by giving penalties using error function like least squares error
- Regression learning: prediction of continuous valued output ex. Housing price vs size
- Classification Learning: discrete valued output, values can be assigned to classes for grouping
- most standard training sets work poorly for DH (or unobvious, theoretical applications)
- should generate a ML in DH data set (.... could I? If not what should i use?)

[**Latent spaces of culture**]:(https://tedunderwood.com/2021/10/21/latent-spaces-of-culture/)
- danger of stochastic parrots: large language models pose “social risk” & misdirected effort, don’t understand language just chinese room
- Argues AI still beneficial to historians bc used to extrapolating meaning/patterns from text
- ”not to mimic individual language understanding, but to represent specific cultural practices (like styles or expository templates) so they can be studied and creatively remixed.” models of culture not language/intelligence
- less worried about bias bc history is about comparing perspectives, can train models on different sources/ time periods to compare bias
- Lightweight models less power intensive to train, more conducive for multiple models in small projects
- [BERT](https://arxiv.org/abs/1810.04805v2) Can be trained to categorize perspectives “designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers”....WTF?
- danger of oligopoly by tech companies in ML
- CLIP “learns visual concepts from natural language supervision”

[AI Dungeon](https://play.aidungeon.io/main/adventurePlay?publicId=3f94167c-0083-4fe5-8db9-5b11045c1e44): 
- I played this for a bit from Dr. Klein's syllabus.it is interesting and seems rather ambitious but the AI is far from perfect, the story seems to lack any continuity and it often misinterprets what you say….I wonder why that is and if it will get better with more players and more input. I assume this is a classification model looking for certain words. Can a model use both classification and regression?
### Week  2:

[Time aware language models as temporal knowledge bases](https://arxiv.org/abs/2106.15110)
-CUSTOMNEWS, “webdocuments” determined to be news with date stamps. Used to train a time aware model. I feel such a model may be used to look  at change over time, very useful for historical study
Readings:
- [Poems with pattern and VADER](https://scholarslab.lib.virginia.edu/blog/poems-with-pattern-and-vader-part-1-quincy-troupe/)
- [Formulating research questions for DH methods](https://latex-ninja.com/2020/03/29/formulating-research-questions-for-using-dh-methods/)
- [Leveraging alignment between machine learning and intersectionality](https://www.sciencedirect.com/science/article/abs/pii/S0304422X21000115)
- [How words lead to justice](https://www.publicbooks.org/how-words-lead-to-justice/)
- [Textblob sentiment analyzer](https://textblob.readthedocs.io/en/dev/advanced_usage.html)
