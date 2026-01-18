import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


df = pd.read_excel("/home/sambeg/human_labelled/human_labeled_predicted.xlsx")
df_reduced = pd.read_excel("/home/sambeg/reduced_emotions/human_labelled/human_labeled_predicted.xlsx")


true_labels = df['label'].str.lower()  #human labeled
true_labels_reduced = df_reduced['label'].str.lower()  #human labeled

predicted_labels_sentiment= df['predicted_sentiment'].str.lower() #human labeled sentiment
predicted_labels_sentiment_reduced = df_reduced['predicted_sentiment'].str.lower() #human labeled sentiment

predicted_labels = df['predicted_emotion'].str.lower()  # predicted using neural network
predicted_labels_neural_only_lexical = df['predicted_emotion_only_lexical'].str.lower()  # predicted using neural network built using only lexical data
predicted_labels_lexical = df['detected_emotion_lexical_algo'].str.lower() # predicted using lexical algorithm
predicted_labels_pos_based = df['detected_emotion_pos_rulebased_algo'].str.lower() # predicted using pos rule based
predicted_labels_roberta_full_max = df['detected_sentiment_roberta (Max as default)'].str.lower()  # predicted sentiment using roberta full emotion with max as default
predicted_labels_roberta_full_neutral = df['detected_sentiment_roberta (Neutral as default)'].str.lower()  # predicted sentiment using roberta full emotion with neutral as default 


predicted_labels_reduced = df_reduced['predicted_emotion'].str.lower() #predicted using neural network reduced
predicted_labels_reduced_LSTM = df_reduced ['predicted_emotion_LSTM'].str.lower() #predicted using neural network reduced and LSTM
predicted_labels_reduced_LSTM_SMOTE = df_reduced['predicted_emotion_LSTM_SMOTE'].str.lower()
predicted_labels_reduced_LSTM_CVSMOTE = df_reduced['predicted_emotion_LSTM_CVSMOTE'].str.lower()
predicted_labels_reduced_LSTM_MOWWD = df_reduced['predicted_emotion_LSTM_MOWWD'].str.lower()
predicted_labels_reduced_LSTM_MOWWD_whole_text = df_reduced['predicted_emotion_LSTM_MOWWD_whole_text'].str.lower()

predicted_labels_roberta_reduced_max = df_reduced['detected_sentiment_roberta (Max as default)'].str.lower()  # predicted sentiment using roberta reduced emotion with max as default
predicted_labels_roberta_reduced_neutral = df_reduced['detected_sentiment_roberta (Neutral as default)'].str.lower()  # predicted sentiment using roberta reduced emotion with neutral as default 




#combine labels neural network full
class_names_neural_network = sorted(list(set(true_labels) | set(predicted_labels)))  

# Generate confusion matrix neural network full
cm = confusion_matrix(true_labels, predicted_labels, labels=class_names_neural_network)

plt.figure(figsize=(20, 8)) 

# Display confusion matrix for neural network full
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names_neural_network)
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())

# Add a title and show the plot
plt.title('Confusion Matrix (predicted using neural network model)')

plt.savefig('/home/sambeg/results_confusion_matrix_line_chart/confusion_matrix_neural_network.png', dpi=300, bbox_inches='tight')




#combine labels neural network built using only lexical data
class_names_neural_network_only_lexical = sorted(list(set(true_labels) | set(predicted_labels_neural_only_lexical)))  

# Generate confusion matrix neural network full
cm_neural_network_only_lexical = confusion_matrix(true_labels, predicted_labels_neural_only_lexical, labels=class_names_neural_network_only_lexical)

plt.figure(figsize=(20, 8)) 

# Display confusion matrix for neural network full
disp = ConfusionMatrixDisplay(confusion_matrix=cm_neural_network_only_lexical, display_labels=class_names_neural_network_only_lexical)
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())

# Add a title and show the plot
plt.title('Confusion Matrix (predicted using neural network model built using only lexical data)')

plt.savefig('/home/sambeg/results_confusion_matrix_line_chart/confusion_matrix_neural_network_only_lexical_data.png', dpi=300, bbox_inches='tight')





#combine labels_lexical
class_names_lexical = sorted(list(set(true_labels) | set(predicted_labels_lexical)))  

# Generate confusion matrix for lexical algo
cm_lexical = confusion_matrix(true_labels, predicted_labels_lexical, labels=class_names_lexical)

plt.figure(figsize=(20, 8)) 

# Display confusion matrix for lexical algo
disp = ConfusionMatrixDisplay(confusion_matrix=cm_lexical, display_labels=class_names_lexical)
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())


# Add a title and show the plot
plt.title('Confusion Matrix (predicted using lexical algorithm)')

plt.savefig('/home/sambeg/results_confusion_matrix_line_chart/confusion_matrix_lexical_algo.png', dpi=300, bbox_inches='tight')




# combine labels_pos_based
class_names_pos_based = sorted(list(set(true_labels) | set(predicted_labels_pos_based)))  

# Generate confusion matrix for pos based algo
cm_pos_based = confusion_matrix(true_labels, predicted_labels_pos_based, labels=class_names_pos_based)
 
plt.figure(figsize=(20, 8)) 

# Display confusion matrix for pos based algo
disp = ConfusionMatrixDisplay(confusion_matrix=cm_pos_based, display_labels=class_names_pos_based)
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())


# Add a title and show the plot
plt.title('Confusion Matrix (predicted using pos based algorithm)')

plt.savefig('/home/sambeg/results_confusion_matrix_line_chart/confusion_matrix_pos_based_algo.png', dpi=300, bbox_inches='tight')




# combine labels sentiments/roberta full with max as default
class_names_sentiment_roberta_full_max = sorted(list(set(predicted_labels_sentiment) | set(predicted_labels_roberta_full_max)))  

# Generate confusion matrix for sentiments/roberta full with max as default
cm_sentiment_roberta_full_max = confusion_matrix(predicted_labels_sentiment, predicted_labels_roberta_full_max, labels=class_names_sentiment_roberta_full_max)
 
plt.figure(figsize=(20, 8)) 

# Display confusion matrix for sentiments/roberta full with max as default
disp = ConfusionMatrixDisplay(confusion_matrix=cm_sentiment_roberta_full_max, display_labels=class_names_sentiment_roberta_full_max)
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())

# Add a title and show the plot
plt.title('Confusion Matrix (compare predicted sentiment using neural network and predicted sentiment using roberta Full_max as default)')

plt.savefig('/home/sambeg/results_confusion_matrix_line_chart/confusion_matrix_sentiment_roberta_full_max.png', dpi=300, bbox_inches='tight')




# combine labels sentiments/roberta full with neutral as default
class_names_sentiment_roberta_full_neutral = sorted(list(set(predicted_labels_sentiment) | set(predicted_labels_roberta_full_neutral)))  

# Generate confusion matrix for sentiments/roberta full with neutral as default
cm_sentiment_roberta_full_neutral = confusion_matrix(predicted_labels_sentiment, predicted_labels_roberta_full_neutral, labels=class_names_sentiment_roberta_full_neutral)
 
plt.figure(figsize=(20, 8)) 

# Display confusion matrix for sentiments/roberta full with neutral as default
disp = ConfusionMatrixDisplay(confusion_matrix=cm_sentiment_roberta_full_neutral, display_labels=class_names_sentiment_roberta_full_neutral)
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())


# Add a title and show the plot
plt.title('Confusion Matrix (compare predicted sentiment using neural network and predicted sentiment using roberta Full_neutral as default)')

plt.savefig('/home/sambeg/results_confusion_matrix_line_chart/confusion_matrix_sentiment_roberta_full_neutral.png', dpi=300, bbox_inches='tight')




# combine labels for neural network reduced
class_names_neural_network_reduced = sorted(list(set(true_labels_reduced) | set(predicted_labels_reduced)))  

# Generate confusion matrix for neural network reduced
cm_neural_network_reduced = confusion_matrix(true_labels_reduced, predicted_labels_reduced, labels=class_names_neural_network_reduced)
 
plt.figure(figsize=(20, 8)) 

# Display confusion matrix for neural network reduced
disp = ConfusionMatrixDisplay(confusion_matrix=cm_neural_network_reduced, display_labels=class_names_neural_network_reduced)
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())


# Add a title and show the plot
plt.title('Confusion Matrix (predicted using neural network reduced emotions)')

plt.savefig('/home/sambeg/results_confusion_matrix_line_chart/confusion_matrix_neural_network_reduced_emotion.png', dpi=300, bbox_inches='tight')



# combine labels for neural network reduced LSTM
class_names_neural_network_reduced_LSTM = sorted(list(set(true_labels_reduced) | set(predicted_labels_reduced_LSTM)))  

# Generate confusion matrix for neural network reduced LSTM
cm_neural_network_reduced_LSTM = confusion_matrix(true_labels_reduced, predicted_labels_reduced_LSTM, labels=class_names_neural_network_reduced_LSTM)
 
plt.figure(figsize=(20, 8)) 

# Display confusion matrix for neural network reduced LSTM
disp = ConfusionMatrixDisplay(confusion_matrix=cm_neural_network_reduced_LSTM, display_labels=class_names_neural_network_reduced_LSTM)
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())


# Add a title and show the plot
plt.title('Confusion Matrix BiLSTM)')

plt.savefig('/home/sambeg/results_confusion_matrix_line_chart/confusion_matrix_BiLSTM.png', dpi=300, bbox_inches='tight')
 



# combine labels for neural network reduced LSTM along with SMOTE
class_names_neural_network_reduced_LSTM_SMOTE = sorted(list(set(true_labels_reduced) | set(predicted_labels_reduced_LSTM_SMOTE)))  

# Generate confusion matrix for neural network reduced LSTM
cm_neural_network_reduced_LSTM_SMOTE = confusion_matrix(true_labels_reduced, predicted_labels_reduced_LSTM_SMOTE, labels=class_names_neural_network_reduced_LSTM_SMOTE)
 
plt.figure(figsize=(20, 8)) 

# Display confusion matrix for neural network reduced LSTM
disp = ConfusionMatrixDisplay(confusion_matrix=cm_neural_network_reduced_LSTM_SMOTE, display_labels=class_names_neural_network_reduced_LSTM_SMOTE)
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())


# Add a title and show the plot
plt.title('Confusion Matrix BiLSTM(SMOTE for oversampling)')

plt.savefig('/home/sambeg/results_confusion_matrix_line_chart/confusion_matrix_BiLSTM_SMOTE.png', dpi=300, bbox_inches='tight')




# combine labels for neural network reduced LSTM along with CVSMOTE
class_names_neural_network_reduced_LSTM_CVSMOTE = sorted(list(set(true_labels_reduced) | set(predicted_labels_reduced_LSTM_CVSMOTE)))  

# Generate confusion matrix for neural network reduced LSTM
cm_neural_network_reduced_LSTM_CVSMOTE = confusion_matrix(true_labels_reduced, predicted_labels_reduced_LSTM_CVSMOTE, labels=class_names_neural_network_reduced_LSTM_CVSMOTE)
 
plt.figure(figsize=(20, 8)) 

# Display confusion matrix for neural network reduced LSTM
disp = ConfusionMatrixDisplay(confusion_matrix=cm_neural_network_reduced_LSTM_CVSMOTE, display_labels=class_names_neural_network_reduced_LSTM_CVSMOTE)
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())


# Add a title and show the plot
plt.title('Confusion Matrix BiLSTM(CVSMOTE for oversampling)')

plt.savefig('/home/sambeg/results_confusion_matrix_line_chart/confusion_matrix_BiLSTM_CVSMOTE.png', dpi=300, bbox_inches='tight')





# combine labels for neural network reduced LSTM along with MOWWD
class_names_neural_network_reduced_LSTM_MOWWD = sorted(list(set(true_labels_reduced) | set(predicted_labels_reduced_LSTM_MOWWD)))  

# Generate confusion matrix for neural network reduced LSTM
cm_neural_network_reduced_LSTM_MOWWD = confusion_matrix(true_labels_reduced, predicted_labels_reduced_LSTM_MOWWD, labels=class_names_neural_network_reduced_LSTM_MOWWD)
 
plt.figure(figsize=(20, 8)) 

# Display confusion matrix for neural network reduced LSTM
disp = ConfusionMatrixDisplay(confusion_matrix=cm_neural_network_reduced_LSTM_MOWWD, display_labels=class_names_neural_network_reduced_LSTM_MOWWD)
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())


# Add a title and show the plot
plt.title('Confusion Matrix BiLSTM (MOWWD for oversampling))')

plt.savefig('/home/sambeg/results_confusion_matrix_line_chart/confusion_matrix_BiLSTM_MOWWD.png', dpi=300, bbox_inches='tight')
 



# combine labels for neural network reduced LSTM along with MOWWD whole text
class_names_neural_network_reduced_LSTM_MOWWD_wholetext = sorted(list(set(true_labels_reduced) | set(predicted_labels_reduced_LSTM_MOWWD_whole_text)))  

# Generate confusion matrix for neural network reduced LSTM
cm_neural_network_reduced_LSTM_MOWWD_wholetext = confusion_matrix(true_labels_reduced, predicted_labels_reduced_LSTM_MOWWD_whole_text, labels=class_names_neural_network_reduced_LSTM_MOWWD_wholetext)
 
plt.figure(figsize=(20, 8)) 

# Display confusion matrix for neural network reduced LSTM
disp = ConfusionMatrixDisplay(confusion_matrix=cm_neural_network_reduced_LSTM_MOWWD_wholetext, display_labels=class_names_neural_network_reduced_LSTM_MOWWD_wholetext)
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())


# Add a title and show the plot
plt.title('Confusion Matrix BiLSTM (MOWWD-whole text for oversampling)')

plt.savefig('/home/sambeg/results_confusion_matrix_line_chart/confusion_matrix_neural_BiLSTM_MOWWD_wholetext.png', dpi=300, bbox_inches='tight')



# combine labels sentiments/roberta reduced with max as default
class_names_sentiment_roberta_reduced_max = sorted(list(set(predicted_labels_sentiment_reduced) | set(predicted_labels_roberta_reduced_max)))  

# Generate confusion matrix for sentiments/roberta reduced with max as default
cm_sentiment_roberta_reduced_max = confusion_matrix(predicted_labels_sentiment_reduced, predicted_labels_roberta_reduced_max, labels=class_names_sentiment_roberta_reduced_max)
 
plt.figure(figsize=(20, 8)) 

# Display confusion matrix for sentiments/roberta reduced with max as default
disp = ConfusionMatrixDisplay(confusion_matrix=cm_sentiment_roberta_reduced_max, display_labels=class_names_sentiment_roberta_reduced_max)
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())

# Add a title and show the plot
plt.title('Confusion Matrix (compare predicted sentiment using neural network and predicted sentiment using roberta reduced_max as default)')

plt.savefig('/home/sambeg/results_confusion_matrix_line_chart/confusion_matrix_sentiment_roberta_reduced_max.png', dpi=300, bbox_inches='tight')




# combine labels sentiments/roberta reduced with neutral as default
class_names_sentiment_roberta_reduced_neutral = sorted(list(set(predicted_labels_sentiment_reduced) | set(predicted_labels_roberta_reduced_neutral)))  

# Generate confusion matrix for sentiments/roberta reduced with neutral as default
cm_sentiment_roberta_reduced_neutral = confusion_matrix(predicted_labels_sentiment_reduced, predicted_labels_roberta_reduced_neutral, labels=class_names_sentiment_roberta_reduced_neutral)
 
plt.figure(figsize=(20, 8)) 

# Display confusion matrix for sentiments/roberta reduced with neutral as default
disp = ConfusionMatrixDisplay(confusion_matrix=cm_sentiment_roberta_reduced_neutral, display_labels=class_names_sentiment_roberta_reduced_neutral)
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())


# Add a title and show the plot
plt.title('Confusion Matrix (compare predicted sentiment using neural network and predicted sentiment using roberta reduced_neutral as default)')

plt.savefig('/home/sambeg/results_confusion_matrix_line_chart/confusion_matrix_sentiment_roberta_reduced_neutral.png', dpi=300, bbox_inches='tight')
