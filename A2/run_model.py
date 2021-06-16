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
import re # for regex
import pandas as pd
import numpy as np

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

    # create remove.txt
    save_remove_txt(removed_words_positive,removed_words_negative)
    # build model
    my_result_model = build_model(processed_training_negative, processed_training_positive)

    # (1.3) Test your dataset
    df_test_result = naive_bayes_classifier(my_result_model,testing_dataset_positive,testing_dataset_negative,math.ceil(num_positive / 2),math.ceil(num_negative / 2))
    

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
    total_nb_words_negative = sum(dict_freq_negative_updated.values()) + size_vocabulary
    dict_p_wi_negative = {k : np.round((v+1)/total_nb_words_negative,6) for k, v in dict_freq_negative_updated.items()}

    total_nb_words_positive = sum(dict_freq_positive_updated.values()) + size_vocabulary
    dict_p_wi_positive = {k : np.round((v+1)/total_nb_words_positive,6) for k, v in dict_freq_positive_updated.items()}
    
    # print(f'dict_p_wi_negative \n {dict_p_wi_negative.items()}')
    print(f'\n[Build Model] size_vocabulary : {size_vocabulary}, total_nb_words_positive : {total_nb_words_positive}, total_nb_words_negative : {total_nb_words_negative}')

    # create dataframe of model
    columns_name = ['word-name','freq-positive','freq-negative','p-wi-positive','p-wi-negative']

    df_freq_positive = pd.DataFrame(dict_freq_positive_updated.items(),columns=[columns_name[0],columns_name[1]])
    df_freq_negative = pd.DataFrame(dict_freq_negative_updated.items(),columns=[columns_name[0],columns_name[2]])
    df_p_wi_positive = pd.DataFrame(dict_p_wi_positive.items(),columns=[columns_name[0],columns_name[3]])
    df_p_wi_negative = pd.DataFrame(dict_p_wi_negative.items(),columns=[columns_name[0],columns_name[4]])

    # concate DataFrames
    df_all = [df_freq_positive,df_freq_negative,df_p_wi_positive,df_p_wi_negative]
    result_model = df_all[0].merge(df_all[1].merge(df_all[2]).merge(df_all[3]))

    # save model.txt
    save_model_txt(result_model)

    return result_model


def save_model_txt(result_model):
    '''
    The method is to write a model.txt to store the info of the model
    :param: model : DataFrame
    '''
    # create model.txt
    file_name = 'model.txt'
    my_model_file = open(file_name,'w')
    # add info of the model in the model.txt
    for index, row in result_model.iterrows():
        title = f'No.{index+1} {row["word-name"]}\n'
        info = f'{row["freq-positive"]}, {row["p-wi-positive"]}, {row["freq-negative"]}, {row["p-wi-negative"]}\n'
        my_model_file.write(title + info)
    # close file
    my_model_file.close()
    print("\n[TXT] model.txt has saved success!")

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

def save_result_txt(a_dataframe):
    '''
    The method is to store info of test result
    :param: a dataframe
    '''
    file_name = 'result.txt'
    my_result_file = open(file_name,'w')
    temp = 0
    for index, row in a_dataframe.iterrows():
        temp += 1
        my_result_file.write(f'No.{temp} {row["review-title"]}\n')
        my_result_file.write(f'{row["p-ri-positive"]}, {row["p-ri-negative"]}, {row["my-result"]}, {row["correct-result"]}, {row["prediction"]}\n')
    # add the prediction correctness
    prediction_correctness = calc_prediction(a_dataframe)
    my_result_file.write(f'The prediction correctness is {prediction_correctness}%')

    print("\n[TXT] result.txt has saved success!\n")

def calc_prediction(a_dataframe):
    '''
    The method is to calculate the prediction correctness.
    :param: a dataframe
    :return: prediction_correctness (float)
    '''
    # add the prediction correctness
    df_freq_group = a_dataframe['prediction'].value_counts()
    freq_correct_prediction = df_freq_group[df_freq_group.index == 'right']
    prediction_correctness = np.round(np.round(freq_correct_prediction / len(a_dataframe),4)*100,2)
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
    for index, row in dataset.iterrows():
        # use regex to replace some pattern
        row["review-comment"] = re.sub(r'[\.\/\'\-\+]',' ',row["review-comment"])
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
        list_sum_result_negative.append(np.round(sum_result_positive,2))

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
    print("[news] saving result.txt file")
    save_result_txt(df_test_result)

    return df_test_result


if __name__ == '__main__':
    main()