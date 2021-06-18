# -------------------------------------------------------
# Assignment (2)
# Written by (Hualin Bai : 40053833 , Jiawei Zeng : 40079344)
# For COMP 472 Section (LAB AI-X) – Summer 2021
# --------------------------------------------------------

'''
This run_model.py is to (1.2) extract the data and build the model
and (1.3) test your dataset, and (2) Experiments with your classifier
'''
import sys
import math
import copy # for deepcopy
import re # for regex
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # for progressbar

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def main():
    # (1.2) extract the data and build the model
    # read review_data_ordered.csv
    reviews_csv = pd.read_csv('review_data_ordered.csv')

    # assign training and testing datasets
    num_negative = (reviews_csv['positive-negative'] == 0).astype(int).sum()
    num_positive = (reviews_csv['positive-negative'] == 1).astype(int).sum()
    training_dataset_negative = reviews_csv.iloc[0:math.ceil(num_negative / 2)]
    testing_dataset_negative = reviews_csv.iloc[math.ceil(num_negative / 2) : num_negative]
    training_dataset_positive = reviews_csv.iloc[num_negative : num_negative + math.ceil(num_positive / 2)]
    testing_dataset_positive = reviews_csv.iloc[num_negative + math.ceil(num_positive / 2) : ]

    # counter word frequency and conditional probability
    processed_training_negative, removed_words_negative = process_text(training_dataset_negative)
    processed_training_positive, removed_words_positive = process_text(training_dataset_positive)

    # set numbers for calculate P(positive) and P(negative)
    positive_train_nums = len(processed_training_positive)
    negative_train_nums = len(processed_training_negative)
    # create remove.txt
    save_remove_txt(removed_words_positive,removed_words_negative)
    # build model
    my_result_model = build_model(processed_training_negative, processed_training_positive)

    # (1.3) Test your dataset
    df_test_result = naive_bayes_classifier(my_result_model,testing_dataset_positive,testing_dataset_negative,positive_train_nums,negative_train_nums)
    # save result.txt
    print("[news] saving result.txt file")
    save_result_txt(df_test_result, 'result.txt')
    # store prediction and words' amount
    prediction_list = []
    words_amount_list = []
    # get original result
    prediction_list.append(calc_prediction(df_test_result))
    words_amount_list.append(len(my_result_model))

      
    while True:
        # choose Task 2.1, 2.2, 2.3
        task = input("Please choose a Task : [1 : Task 2.1, 2 : Task 2.2, 3 : Task 2.3, 9 : Exit]")
        if task == '1':
            # Task 2.1 
            prediction_list_1 = copy.deepcopy(prediction_list)
            words_amount_list_1 = copy.deepcopy(words_amount_list)
            infrequency = [1,10,20]
            print("[Task 2.1] Infrequent Word Filtering")
            # deepcopy original result model
            df_infrequent_model = copy.deepcopy(my_result_model)
            for item in tqdm(infrequency, desc='Removing Frequency'):
                print(f'\n[news] cut frequency is : {item}')
                df_infrequent_model, size_volcabulary = rebuild_model_by_frequency(df_infrequent_model,item,0)
                # print(df_infrequent_model.info)
                df_test_result = naive_bayes_classifier(df_infrequent_model,testing_dataset_positive,testing_dataset_negative,positive_train_nums,negative_train_nums)
                prediction_list_1.append(calc_prediction(df_test_result))
                words_amount_list_1.append(size_volcabulary)
            print("\n[news] removing top frequency words...")   
            top_frequency = [0.05,0.1,0.2]
            for item in tqdm(top_frequency,desc='Removing Top Frequency'):
                print(f'\n[news] cut top is : {item*100}%')
                df_infrequent_model,size_volcabulary = rebuild_model_by_frequency(df_infrequent_model,0,item)
                # print(df_infrequent_model.info)
                df_test_result = naive_bayes_classifier(df_infrequent_model,testing_dataset_positive,testing_dataset_negative,positive_train_nums,negative_train_nums)
                prediction_list_1.append(calc_prediction(df_test_result))
                words_amount_list_1.append(size_volcabulary)

            print(f'\n[Finished Task 2.1] prediction correctness is : {prediction_list}')
            print(f'\t\t words_amount is : {words_amount_list}')
            # save the final results
            save_model_txt(df_infrequent_model,'frequency-model.txt')
            save_result_txt(df_test_result,'frequency-result.txt')
            print("\n[TXT] Success saved frequency-model.txt and frequency-result.txt !!!\n")
            # save image of task 2.1
            print("[news] saving the image of the results...")
            # draw graph
            fig, ax = plt.subplots(figsize=(10,5))
            plt.xticks(range(len(words_amount_list_1)), words_amount_list_1)
            # set title and axis lables
            ax.set_xlabel("The Number of Words left in vocabulary (w) ")
            ax.set_ylabel("Correctness of Prediction (%) for the number of words")
            ax.set_title("Task 2.1 Infrequent Word Filtering Result", fontsize=16, color='black', verticalalignment="baseline")

            plt.bar(range(len(prediction_list_1)), prediction_list_1)
            ax.grid(True)
            fig.savefig(r'./images/task_2_1_graph.jpg')
            print("\n[Image] please check (task_2_1_graph.jpg) in the image folder!\n")
        elif task == '2':
            # Task 2.2 Word smoothing filter
            # smoothing is from 1.2 to 2 instead of 1 to 2, since default smoothing=1
            print("[Task 2.2] Word smoothing filter")
            prediction_list_2 = copy.deepcopy(prediction_list)
            smoothing_values = [1,1.2,1.4,1.6,1.8,2]
            for s_i in tqdm(smoothing_values,desc='processing smooth'):
                if s_i == 1: 
                    pass
                else:
                    df_smoothing_model = rebuild_model_by_smoothing(my_result_model,s_i)
                    # print(df_smoothing_model.info)
                    df_test_result = naive_bayes_classifier(df_smoothing_model,testing_dataset_positive,testing_dataset_negative,positive_train_nums,negative_train_nums)
                    prediction_list_2.append(calc_prediction(df_test_result))
                    # save the result when smoothing = 1.6
                    if s_i == 1.6:
                        print("[Saving] the result and model of the smoothing is 1.6")
                        save_model_txt(df_smoothing_model,'smooth-model.txt')
                        save_result_txt(df_test_result,'smooth-result.txt')
                        print("[TXT] Success saved smooth-model.txt and smooth-result.txt !!!\n")
            # draw graph
            fig, ax = plt.subplots(figsize=(10,5))
            plt.xticks(range(len(smoothing_values)), smoothing_values)
            # set title and axis lables
            ax.set_xlabel("Different Smoothing Values  ")
            ax.set_ylabel("Correctness of Prediction (%) for different smoothing values ")
            ax.set_title("Task 2.2 Word Smoothing Filtering Result", fontsize=16, color='black', verticalalignment="baseline")

            plt.bar(range(len(prediction_list_2)), prediction_list_2)
            ax.grid(True)
            # save image
            fig.savefig(r'./images/task_2_2_graph.jpg')
            print("\n[Image] please check (task_2_2_graph.jpg) in the image folder!\n")
        elif task == '3':
            # Task 2.3
            prediction_list_3 = copy.deepcopy(prediction_list)
            words_amount_list_3 = copy.deepcopy(words_amount_list)
            word_length_values = [2,4,9]
            print("[Task 2.3] Word Length Filtering")
            # deepcopy original result model
            df_word_length_model = copy.deepcopy(my_result_model)
            for length in tqdm(word_length_values,desc='processing word length'):
                print(f'\n[news] remove word length is : {length}')
                df_word_length_model, size_volcabulary = rebuild_model_by_word_length(df_word_length_model,length)
                # print(df_word_length_model.info)
                df_test_result = naive_bayes_classifier(df_word_length_model,testing_dataset_positive,testing_dataset_negative,positive_train_nums,negative_train_nums)
                prediction_list_3.append(calc_prediction(df_test_result))
                words_amount_list_3.append(size_volcabulary)
            # save txt files
            print("[Saving] the result and model of the word length")
            save_model_txt(df_word_length_model,'length-model.txt')
            save_result_txt(df_test_result,'length-result.txt')
            print("[TXT] Success saved length-model.txt and length-result.txt !!!\n")

            # draw graph
            fig, ax = plt.subplots(figsize=(10,5))
            plt.xticks(range(len(words_amount_list_3)), words_amount_list_3)
            # set title and axis lables
            ax.set_xlabel("The number of words left in Vocabulary")
            ax.set_ylabel("Correctness of Prediction (%) for different length ")
            ax.set_title("Task 2.3 Word Length Filtering Result", fontsize=16, color='black', verticalalignment="baseline")
            plt.bar(range(len(prediction_list_3)), prediction_list_3)
            ax.grid(True)
            # save image
            fig.savefig(r'./images/task_2_3_graph.jpg')
            print("\n[Image] please check (task_2_3_graph.jpg) in the image folder!\n")

        elif task == '9':
            sys.exit(0)
        else:
            pass

# Task 2.1 Infrequent word filtering
def rebuild_model_by_frequency(a_df_model, cut_frequency = 0, cut_top = 0):
    '''
    The method is to rebuild model including re-calculate the frequency of each word in the dataset, and conditional probability 
    of P(wi|positive) or P(wi|negative).
    :param: a_df_model
    :param: cut frequency number
    :param: cut top num for most frequency words
    :return: df_infrequent_model : pandas.dataframe -- column: word's name, rows: |freq in positive | freq in negative | P(wi|positive)| P(wi|negative)|
    :return: new_size_vocabulary : int
    '''
    # TODO: Task 2.1 Infrequent word filtering
    df_infrequent_model = copy.deepcopy(a_df_model)
    # remove vocabulary words according the cut_frequency
    # replace 0 for all words which frequency = cut_frequency such as 1,10,20
    if cut_frequency == 0: pass
    elif cut_frequency == 1:
        df_infrequent_model.loc[df_infrequent_model["freq-positive"] == cut_frequency] = 0
        df_infrequent_model.loc[df_infrequent_model["freq-negative"] == cut_frequency] = 0  
    else:
        df_infrequent_model.loc[df_infrequent_model["freq-positive"] <= cut_frequency] = 0
        df_infrequent_model.loc[df_infrequent_model["freq-negative"] <= cut_frequency] = 0 
    # remove word from the vocabulary
    df_infrequent_model = df_infrequent_model.drop(df_infrequent_model[(df_infrequent_model["freq-positive"] == 0) & (df_infrequent_model["freq-negative"] == 0)].index)
    # reorder the df_model for cut top frequency
    df_infrequent_model = df_infrequent_model.sort_values(by=["freq-positive","freq-negative"],ascending=False)
    if cut_top == 0: pass
    else:
        num_remove_rows = math.ceil(cut_top*len(df_infrequent_model))
        # remove top frequency words
        df_infrequent_model = df_infrequent_model.drop(df_infrequent_model.head(num_remove_rows).index)
    
    # smoothing 1
    smoothing = 1
    # re-calculate new_size_vocabulary
    new_size_vocabulary = len(df_infrequent_model) * smoothing # add 1 smoothing

    # re-calculate new_total_nb_words_positive and new_total_nb_words_positive
    new_total_nb_words_positive = df_infrequent_model["freq-positive"].sum() + new_size_vocabulary
    new_total_nb_words_negative = df_infrequent_model["freq-negative"].sum() + new_size_vocabulary
    print(f'\n[Rebuild Model by Frequency] new_total_nb_words_positive : {new_total_nb_words_positive}, new_total_nb_words_negative : {new_total_nb_words_negative}')

    print(f'[Rebuild Model by Frequency] new_size_vocabulary is : {new_size_vocabulary}')
    # print(df_infrequent_model)

    for i in range(len(df_infrequent_model)):
        # rewrite the p-wi-positive and p-wi-negative
        df_infrequent_model.iloc[i,3] = np.round( (df_infrequent_model.iloc[i,1] + smoothing) / new_total_nb_words_positive,6)
        df_infrequent_model.iloc[i,4] = np.round( (df_infrequent_model.iloc[i,2] + smoothing) / new_total_nb_words_negative,6)

    print("[news] Success rebuild model by frequency!")
    return df_infrequent_model, new_size_vocabulary

# Task 2.2 Word Smoothing Filtering
def rebuild_model_by_smoothing(a_df_model, current_smoothing):
    '''
    The method is to rebuild model by smoothing including re-calculate the frequency of each word in the dataset, and conditional probability 
    of P(wi|positive) or P(wi|negative).
    :param: a_df_model
    :param: current_smoothing
    :return: df_smoothing_model : pandas.dataframe -- column: word's name, rows: |freq in positive | freq in negative | P(wi|positive)| P(wi|negative)|
    '''
    # TODO: Task 2.2 Word Smoothing Filtering
    df_smoothing_model = copy.deepcopy(a_df_model)
    # set smoothing
    smoothing = current_smoothing
    
    # calculate new_size_vocabulary
    new_size_vocabulary = len(df_smoothing_model) * smoothing # add smoothing

    # re-calculate new_total_nb_words_positive and new_total_nb_words_positive
    new_total_nb_words_positive = np.round(df_smoothing_model["freq-positive"].sum() + new_size_vocabulary,1)
    new_total_nb_words_negative = np.round(df_smoothing_model["freq-negative"].sum() + new_size_vocabulary,1)
    print(f'\n[Rebuild Model by Smoothing] Current Smoothing is : {smoothing}')
    print(f'[Rebuild Model by Smoothing] new_total_nb_words_positive : {new_total_nb_words_positive}, new_total_nb_words_negative : {new_total_nb_words_negative}')
    # print(df_infrequent_model)

    for i in range(len(df_smoothing_model)):
        # rewrite the p-wi-positive and p-wi-negative
        df_smoothing_model.iloc[i,3] = np.round( (df_smoothing_model.iloc[i,1] + smoothing) / new_total_nb_words_positive,6)
        df_smoothing_model.iloc[i,4] = np.round( (df_smoothing_model.iloc[i,2] + smoothing) / new_total_nb_words_negative,6)

    print("[news] Success rebuild model by smoothing!\n")
    return df_smoothing_model

# Task 2.3 Word Length Filtering
def rebuild_model_by_word_length(a_df_model, word_length):
    '''
    The method is to rebuild model by word length including re-calculate the frequency of each word in the dataset, and conditional probability 
    of P(wi|positive) or P(wi|negative).
    :param: a_df_model
    :param: word_length
    :return: df_word_length_model : pandas.dataframe -- column: word's name, rows: |freq in positive | freq in negative | P(wi|positive)| P(wi|negative)|
    :return: new_size_vocabulary : int
    '''
    # TODO: Task 2.3 word length filtering
    df_word_length_model = copy.deepcopy(a_df_model)
    current_word_length = word_length
    # remove vocabulary words according the word length
    if current_word_length == 9:
        df_word_length_model = df_word_length_model.drop(df_word_length_model[(df_word_length_model["word-name"].str.len() >= current_word_length)].index)
    else:
        df_word_length_model = df_word_length_model.drop(df_word_length_model[(df_word_length_model["word-name"].str.len() <= current_word_length)].index)
    
    # smoothing 1
    smoothing = 1
    # calculate new_size_vocabulary
    new_size_vocabulary = len(df_word_length_model) * smoothing # add 1 smoothing

    # re-calculate new_total_nb_words_positive and new_total_nb_words_positive
    new_total_nb_words_positive = df_word_length_model["freq-positive"].sum() + new_size_vocabulary
    new_total_nb_words_negative = df_word_length_model["freq-negative"].sum() + new_size_vocabulary
    print(f'\n[Rebuild Model by Word Length] new_total_nb_words_positive : {new_total_nb_words_positive}, new_total_nb_words_negative : {new_total_nb_words_negative}')
    print(f'[Rebuild Model by Word Length] new_size_vocabulary is : {new_size_vocabulary}')
    # print(df_infrequent_model)

    for i in range(len(df_word_length_model)):
        # rewrite the p-wi-positive and p-wi-negative
        df_word_length_model.iloc[i,3] = np.round( (df_word_length_model.iloc[i,1] + smoothing) / new_total_nb_words_positive,6)
        df_word_length_model.iloc[i,4] = np.round( (df_word_length_model.iloc[i,2] + smoothing) / new_total_nb_words_negative,6)

    print("[news] Success rebuild model by Word Length!")
    return df_word_length_model, new_size_vocabulary

def build_model(processed_dataset_negative,processed_dataset_positive):
    '''
    The method is to build model including calculate the frequency of each word in the dataset, and conditional probability 
    of P(wi|positive) or P(wi|negative).
    :param: processed dataset_negative and dataset_positive
    :return: result_model : pandas.dataframe -- column: word's name, rows: |freq in positive | freq in negative | P(wi|positive)| P(wi|negative)|

    '''
    # calculate frequency negative and positive
    freq_negative = nltk.FreqDist(processed_dataset_negative)
    freq_positive = nltk.FreqDist(processed_dataset_positive)
    
    # create dictionary for frequency negative and positive
    dict_freq_negative = {k:v for k,v in freq_negative.items()}
    dict_freq_positive = {k:v for k,v in freq_positive.items()}
    # use set to get all words from positive and negative
    total_words_set = set(dict_freq_negative.keys()) | set(dict_freq_positive.keys())

    # set initial value is 0
    row_size = len(total_words_set) # row size of default_total_words_dict
    default_total_words_dict = dict.fromkeys(list(total_words_set),0)

    # update the dictionary's value
    dict_freq_positive_updated = {**default_total_words_dict, **dict_freq_positive}
    dict_freq_negative_updated = {**default_total_words_dict, **dict_freq_negative}

    # calculate conditinal probability of negative and positive
    size_vocabulary = len(default_total_words_dict) * 1 # add 1 smoothing
    # total_nb_words_negative = sum(dict_freq_negative_updated.values()) + size_vocabulary
    dict_p_wi_negative = {k : 0 for k, v in dict_freq_negative_updated.items()}

    # total_nb_words_positive = sum(dict_freq_positive_updated.values()) + size_vocabulary
    dict_p_wi_positive = {k : 0 for k, v in dict_freq_positive_updated.items()}
    
    # print(f'dict_p_wi_negative \n {dict_p_wi_negative.items()}')
    print(f'\n[Build Model] size_vocabulary : {size_vocabulary}')

    # create dataframe of model
    columns_name = ['word-name','freq-positive','freq-negative','p-wi-positive','p-wi-negative']

    df_freq_positive = pd.DataFrame(dict_freq_positive_updated.items(),columns=[columns_name[0],columns_name[1]])
    df_freq_negative = pd.DataFrame(dict_freq_negative_updated.items(),columns=[columns_name[0],columns_name[2]])
    df_p_wi_positive = pd.DataFrame(dict_p_wi_positive.items(),columns=[columns_name[0],columns_name[3]])
    df_p_wi_negative = pd.DataFrame(dict_p_wi_negative.items(),columns=[columns_name[0],columns_name[4]])

    # concate DataFrames
    df_all = [df_freq_positive,df_freq_negative,df_p_wi_positive,df_p_wi_negative]
    result_model = df_all[0].merge(df_all[1].merge(df_all[2]).merge(df_all[3]))

    # optimal vocabulary
    print("[news] processing optimal vocabulary model")
    result_model = optimal_vocabulary_model(result_model)

    # save model.txt
    save_model_txt(result_model,'model.txt')

    return result_model

def optimal_vocabulary_model(a_nb_model):
    '''
    The method is to optimize the nb model.
    :param: a_nb_model
    :return: optimal_nb_model
    '''
    # find same frequency words, then remove them from nb_model
    optimal_nb_model = copy.deepcopy(a_nb_model)
    list_same_words = optimal_nb_model[optimal_nb_model["freq-positive"] == optimal_nb_model["freq-negative"] ]["word-name"]
    list_same_words = list_same_words.tolist()

    # update the remove.txt
    my_remove_file = open('remove.txt','w+')
    for item in list_same_words:
        my_remove_file.write(f'{item}\n')
    # close file
    my_remove_file.close()
    print("\n[TXT] remove.txt has updated success!")

    # drop same words from model
    optimal_nb_model = optimal_nb_model.drop(optimal_nb_model[(optimal_nb_model["freq-positive"] == optimal_nb_model["freq-negative"])].index)

    # calculate p-wi-positive and p-wi-negative
    # smoothing 1
    smoothing = 1
    # re-calculate new_size_vocabulary
    new_size_vocabulary = len(optimal_nb_model) * smoothing # add 1 smoothing

    # re-calculate new_total_nb_words_positive and new_total_nb_words_positive
    new_total_nb_words_positive = optimal_nb_model["freq-positive"].sum() + new_size_vocabulary
    new_total_nb_words_negative = optimal_nb_model["freq-negative"].sum() + new_size_vocabulary
    print(f'\n[Build optimal model] new_total_nb_words_positive : {new_total_nb_words_positive}, new_total_nb_words_negative : {new_total_nb_words_negative}')

    print(f'[Build optimal model] new_size_vocabulary is : {new_size_vocabulary}')
    # print(df_infrequent_model)

    for i in range(len(optimal_nb_model)):
        # rewrite the p-wi-positive and p-wi-negative
        optimal_nb_model.iloc[i,3] = np.round( (optimal_nb_model.iloc[i,1] + smoothing) / new_total_nb_words_positive,6)
        optimal_nb_model.iloc[i,4] = np.round( (optimal_nb_model.iloc[i,2] + smoothing) / new_total_nb_words_negative,6)

    return optimal_nb_model


def save_model_txt(result_model,file_name):
    '''
    The method is to write a model.txt to store the info of the model
    :param: model : DataFrame
    :param: file_name
    '''

    # create txt file for model
    my_model_file = open(file_name,'w')
    # add info of the model in the model.txt
    for index, row in result_model.iterrows():
        title = f'No.{index+1} {row["word-name"]}\n'
        info = f'{row["freq-positive"]}, {row["p-wi-positive"]}, {row["freq-negative"]}, {row["p-wi-negative"]}\n'
        my_model_file.write(title + info)
    # close file
    my_model_file.close()
    print(f'\n[TXT] {file_name} has saved success!')

def save_remove_txt(remove_positive, remove_negative):
    '''
    The method is to store info of remove words
    :param: removed_words_positive and negative
    '''
    file_name = 'remove.txt'
    my_remove_file = open(file_name,'w')
    # use set to write file
    set_total_removed_words = remove_positive | remove_negative
    for item in set_total_removed_words:
        my_remove_file.write(f'{item}\n')
    # close file
    my_remove_file.close()
    print("\n[TXT] remove.txt has saved success!")

def save_result_txt(a_dataframe,file_name):
    '''
    The method is to store info of test result
    :param: a dataframe
    :param: file_name
    '''

    my_result_file = open(file_name,'w')
    temp = 0
    for index, row in a_dataframe.iterrows():
        temp += 1
        my_result_file.write(f'No.{temp} {row["review-title"]}\n')
        my_result_file.write(f'{row["p-ri-positive"]}, {row["p-ri-negative"]}, {row["my-result"]}, {row["correct-result"]}, {row["prediction"]}\n')
    # add the prediction correctness
    prediction_correctness = calc_prediction(a_dataframe)
    my_result_file.write(f'The prediction correctness is {prediction_correctness}%')

    print(f'\n[TXT] {file_name} has saved success!\n')

def calc_prediction(a_dataframe):
    '''
    The method is to calculate the prediction correctness.
    :param: a test dataframe
    :return: prediction_correctness (float)
    '''
    # add the prediction correctness
    df_freq_group = a_dataframe['prediction'].value_counts()
    freq_correct_prediction = df_freq_group[df_freq_group.index == 'right']
    prediction_correctness = np.round(np.round(freq_correct_prediction / len(a_dataframe),4)*100,2)
    print(f'[prediction correctness] is {prediction_correctness[0]}%')
    return prediction_correctness[0]



def process_text(dataset):
    '''
    This method is to process text including extract data, lowercase and tokenize the words.
    Removed words will be saved in remove.txt
    :param: dataset
    :return: processed_dataset
    :return: removed_words
    '''
    processed_dataset = []
    removed_words = set()
    # use stopwords of nltk
    stop_words = stopwords.words('english')
    for review_item in dataset['review-comment']:
        # use regex to replace some pattern
        regex_review_item = re.sub(r'[\.\/\-\+]','',review_item)
        # lowercase word and check if it is alpha
        word_data = [ word.lower() for word in word_tokenize(regex_review_item) if word.isalpha() and len(word) > 1]
        # update the removed_words set
        # removed_words.update([word.lower() for word in word_tokenize(review_item) if not word.isalpha()])
        clean_tokens = word_data[:]
        for token in word_data:
            if token in stop_words:
                clean_tokens.remove(token)
                # add the removed_words
                removed_words.add(token)
        # add word in processed_dataset        
        processed_dataset.extend(clean_tokens)
    # print(processed_dataset)
    # print(len(processed_dataset))

    return processed_dataset, removed_words

def process_testing_text(dataset,nb_model):
    '''
    This method is to process testing text including extract data, lowercase and tokenize the words.
    :param: testing dataset : DataFrame
    :return: list_sum_result positive and negative
    '''
   
    list_sum_result_positive = []
    list_sum_result_negative = []
    # process testing text including extract data, lowercase and tokenize the words.
    # use stopwords of nltk
    stop_words = stopwords.words('english')
    for index, row in tqdm(dataset.iterrows(),desc='processing testing text'):
        # use regex to replace some pattern
        row["review-comment"] = re.sub(r'[\.\/\-\+]',' ',row["review-comment"])
        # lowercase word and check if it is alpha
        word_data = [ word.lower() for word in word_tokenize(row["review-comment"]) if word.isalpha() and len(word) > 1]
        clean_tokens = word_data[:]
        for token in word_data:
            if token in stop_words:
                clean_tokens.remove(token)
        # print(clean_tokens)
       
        # calculate sum(p(wi|positive)) and sum(p(wi|negative)) using log10
        sum_result_positive = 0
        sum_result_negative = 0
        for word_name in clean_tokens:
            # check the word is in the nb_model
            if len(nb_model.loc[nb_model["word-name"] == word_name]) == 1:
                sum_result_positive += math.log10(nb_model.loc[nb_model["word-name"] == word_name]["p-wi-positive"])
                sum_result_negative += math.log10(nb_model.loc[nb_model["word-name"] == word_name]["p-wi-negative"])
        # print(f'{sum_result_positive}')
        list_sum_result_positive.append(np.round(sum_result_positive,2))
        list_sum_result_negative.append(np.round(sum_result_negative,2))

    return list_sum_result_positive, list_sum_result_negative

# testing dataset
def naive_bayes_classifier(nb_model, testing_dataset_p, testing_dataset_n,num_train_positive, num_train_negative):
    '''
    This method is to use naive bayes classifier to classify the testing dataset.
    :param: nb_model, testing_dataset_p, testing_dataset_n,num_train_positive, num_train_negative
    :return: df_test_result : DataFrame
    '''
    # test dataframe to store the test info
    list_review_title = []
    list_p_ri_positive = []
    list_p_ri_negative = []
    list_my_result = []
    list_correct_result = []
    list_prediction = []
    # calculate p(positive) and p(negative)
    p_positive = np.round(math.log10(num_train_positive/(num_train_positive+num_train_negative)),2)
    p_negative = np.round(math.log10(num_train_negative/(num_train_positive+num_train_negative)),2)
    # first, process the reviews of testing datasets
    print("[news] processing the reviews of testing datasets")
    # positive testing dataset
    sum_testing_positive_list_p, sum_testing_negative_list_p = process_testing_text(testing_dataset_p,nb_model)
    list_p_ri_positive.extend(sum_testing_positive_list_p)
    list_p_ri_negative.extend(sum_testing_negative_list_p)
    # store correct result
    for _ in range(len(testing_dataset_p)):
        list_correct_result.append('positive')
    
    # negative testing dataset
    sum_testing_positive_list_n, sum_testing_negative_list_n = process_testing_text(testing_dataset_n,nb_model)
    list_p_ri_positive.extend(sum_testing_positive_list_n)
    list_p_ri_negative.extend(sum_testing_negative_list_n)
    # calculate p(ri|positive), p(ri|negative)
    list_p_ri_positive = np.round(np.array(list_p_ri_positive) + p_positive,2)
    list_p_ri_negative = np.round(np.array(list_p_ri_negative) + p_negative,2)
    # store correct result
    for _ in range(len(testing_dataset_n)):
        list_correct_result.append('negative')
    
    # calculate my result
    list_result_temp = list_p_ri_positive - list_p_ri_negative
    for item in list_result_temp:
        if item > 0:
            list_my_result.append('positive')
        else:
            list_my_result.append('negative')
    # calculate prediction is right or wrong (based on comparing your result with correctresult)
    for i, item in enumerate(list_correct_result):
        if list_my_result[i] == item:
            list_prediction.append('right')
        else:
            list_prediction.append('wrong')
            
    # store review title
    for index, title in testing_dataset_p.iterrows():
        list_review_title.append(title["review-title"])
    for index, title in testing_dataset_n.iterrows():
        list_review_title.append(title["review-title"])
    
    
    # save the info in DataFrame
    df_test_result = pd.DataFrame({
         'review-title':list_review_title ,
         'p-ri-positive':list_p_ri_positive,
         'p-ri-negative':list_p_ri_negative,
         'my-result':list_my_result,
         'correct-result':list_correct_result,
         'prediction':list_prediction})
    # print(df_test_result.info)
    print("[news] finished Naive Bayes Classifier!")
    # print("[news] saving result.txt file")

    return df_test_result


if __name__ == '__main__':
    main()