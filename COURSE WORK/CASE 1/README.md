# Traffic-Flow-Prediction-Using-Deep-Learning

### Problem Definition and Background
Traffic flow prediction largely relies on historical and real-time traffic data collected from various sensor sources, including sensor coils, radars, cameras, mobile global positioning system, crowdsourcing, social media, etc. With the extensive application of traditional traffic sensors and emerging traffic sensor technologies, and the explosive growth of traffic data, transportation management and control is becoming more and more data-driven. 

Although there are many traffic flow prediction systems and models, most of them adopt shallow traffic models, like one hidden layer neural network, which still have some shortcomings. This inspires us to rethink the traffic flow prediction based on deep structure models, which have such abundant traffic data. And it is also worth to try different models to predict traffic flow and compare these models performance, considering the number of parameters, the training speed, the evaluation metrics and etc., to choose a best model.

### Understand the Data
#### Data Introduction
The data is from the Caltrans Performance Measurement System (PeMS) database, http://pems.dot.ca.gov/. The traffic data are collected every 5 min from over 15000 individual detectors, which are deployed statewide in freeway systems across California.

#### Data Preprocessing

The collected data are aggregated 5-min interval each for each detector station. I plan to use the traffic flow data in the weekdays of the July, August, September of the year 2020 for the experiments.

July and August data as the training set, and September data as the testing set.

For freeways with multiple detectors, the traffic data collected by different detectors are aggregated to get the average traffic flow of this freeway.


Data source:
```
Armstrong County HW 2 File.csv
```
For more code details:
```
Traffic Flow Prediction.ipynb
```
Report:
```
Independent Study Final Report.pdf
```
