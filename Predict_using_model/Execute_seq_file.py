import subprocess
import pandas as pd


subprocess.run(["/bin/python3", "predict_using_model.py"])
subprocess.run(["/bin/python3", "predict_using_model_single_speech_at_a_time.py"])
subprocess.run(["/bin/python3", "predict_sentiment_full.py"])
subprocess.run(["/bin/python3", "predict_using_lexical_algo.py"])
subprocess.run(["/bin/python3", "predict_using_pos_rule_based_algo.py"])
subprocess.run(["/bin/python3", "predict_using_pos_single_speech_at_a_time.py"])
subprocess.run(["/bin/python3", "predict_using_model_only_lexical_data.py"])
subprocess.run(["/bin/python3", "predict_using_model_reduced_emt.py"])
subprocess.run(["/bin/python3", "predict_sentiment_reduced_emt.py"])
# subprocess.run(["/bin/python3", "find_roberta_best_threshold_full (max as default).py"])
# subprocess.run(["/bin/python3", "find_roberta_best_threshold_full (neutral as default).py"])
# subprocess.run(["/bin/python3", "find_roberta_best_threshold_reduced_emt (max as default).py"])
# subprocess.run(["/bin/python3", "find_roberta_best_threshold_reduced_emt (neutral as default).py"])
subprocess.run(["/bin/python3", "predict_using_model_reduced_emt_LSTM (GLoVe-embeddings).py"])
subprocess.run(["/bin/python3", "predict_using_model_reduced_emt_LSTM (BERT embeddings).py"])
subprocess.run(["/bin/python3", "predict_using_model_reduced_emt_LSTM_smote.py"])
# subprocess.run(["/bin/python3", "predict_using_model_reduced_emt_LSTM_MOWWD.py"])
# subprocess.run(["/bin/python3", "predict_using_model_reduced_emt_MOWWD_wholetext.py"])




