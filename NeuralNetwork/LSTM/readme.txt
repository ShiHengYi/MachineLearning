How to run my code: just run "python LSTM.py" to run the code within the same directory of "kddcup.data.corrected".
If using other data set please rename the data set to "kddcup.data.corrected" or change the hard coding data set name in LSTM.py
    data = pd.read_csv("kddcup.data.corrected", names = col_names)
When executing the program, there will be serial warning messages, which will not cause any side-effect.
Data loading will take about 30s on purdue scholar machine.

Citation:
https://www.researchgate.net/publication/279770740_Applying_long_short-term_memory_recurrent_neural_networks_to_intrusion_detection
https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.loc.html
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
https://www.youtube.com/watch?v=2PAFVKA-OWY&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN&index=41
https://stackoverflow.com/questions/35365007/tensorflow-precision-recall-f1-score-and-confusion-matrix
https://www.tensorflow.org/api_docs/python/tf/math/count_nonzero
