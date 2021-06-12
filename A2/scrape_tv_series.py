# -------------------------------------------------------
# Assignment (2)
# Written by (Hualin Bai : 40053833 , Jiawei Zeng : 40079344)
# For COMP 472 Section (LAB AI-X) – Summer 2021
# --------------------------------------------------------

'''
This scrape_tv_series.py is to scrape your favorite tv series from IMDB website.
Saving data as data.csv, the columns as follow:
| Name(episode) | Season | Review Link | Year |
'''
import sys
import requests
from bs4 import BeautifulSoup
import re # for regex
import pandas as pd


def download_menu():
    '''
    The method is to display the download menu.
    :return: tv_url
    '''
    print('*' * 100)
    print('\t\t\t\Scraping Menu:')
    print('*' * 100)
    while True:
        try:
            option_name = int(input('please choose one option: (eg. 1)\n [1]: Game of Thrones, [2]: Type URL by myself, [3]: Scrape reviews in data.csv, [9]: Exit\n'))
            # set tv_url
            if option_name == 1:
                tv_url = 'https://www.imdb.com/title/tt0944947/'
                return tv_url
            elif option_name == 9:
                print('[Exit]')
                sys.exit(0)
            elif option_name == 2:
                tv_url = input('Please input a valid TV series link: (eg. https://www.imdb.com/title/tt0944947/)\n')
                # regex check if the link is valid
                if re.match(r'^https://www.imdb.com/title/',tv_url,re.I):
                    return tv_url
                else:
                    print('[Warning] Please input a valid TV series link!')
            elif option_name == 3:
                # crawling reviews
                # crawling the review links from data.csv
                print('\n[crawling] all reviews info from data.csv')
                scrape_all_reviews()
                return None

            else:
                print('[Warning] Please input a valid number!!!')
            
        except ValueError or TypeError:
            print('[Warning] Please input a valid number!!!')
        
def get_season_nums_list(html_soup):
    '''
    The method is to find the maximum season num of the TV series
    :param: htmp_soup
    :return: list of season nums
    '''
    # get all option values of the select id='bySeason'
    season_option_values = html_soup.find('select',id = 'bySeason').find_all('option')
    season_num_list = [item.get('value') for item in season_option_values]
    print(f'[news] Total season nums is : {season_num_list}')
    return season_num_list


def request_url(url):
    '''
    The method is to use request lib to get text for html
    :param: url
    :return: response.text
    '''
    # set user agent
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36',
    }
    try:
        response = requests.get(url,headers=headers)
        # status 200 means connected success
        if response.status_code == 200:
            print('[news] Connected Success!')
            return response.text
    except requests.RequestException:
        print('[Error] Connected Fail')
        return None

def start_scrape_seasons(tv_url):
    '''
    The method is to scrape seasons' all episodes.
    :param: tv_url
    '''
    num_season = 1
    # set full season url
    season_url = tv_url + 'episodes?season=' + str(num_season)
    # get response.text
    response_text = request_url(season_url)
    # parse response.text by creating a BeautifulSoup Object, and assign this object to html_soup
    html_soup = BeautifulSoup(response_text,'html.parser')
    # print(html_soup)
    # (1) find the max num of the seasons
    list_season_nums = get_season_nums_list(html_soup)
    
    # lists to store data for data.csv
    name_episode = []
    season_num = []
    review_link = []
    year = []
    url_head = 'https://www.imdb.com/title/'
    
    # (2) scrape all episodes from each season
    for i in list_season_nums:
        # crawling
        print(f'\n[Crawling] Season: No. {i}')
        # we have already use season 1, so directly use current html_soup
        if i == '1':
            episode_lists = html_soup.find('div', class_='list detail eplist')
            # find eoisode id
            episode_id_lists = [item.get('data-const') for item in episode_lists.find_all('div',class_='hover-over-image zero-z-index')]
            # combine url_head and review links
            episode_review_link_lists = [ url_head + item + '/reviews' for item in episode_id_lists ]
            
            # print(episode_review_link_lists)
        else:
            # set full season url
            season_url = tv_url + 'episodes?season=' + i
            # get response.text
            response_text = request_url(season_url)
            # parse response.text by creating a BeautifulSoup Object, and assign this object to html_soup
            html_soup = BeautifulSoup(response_text,'html.parser')

            episode_lists = html_soup.find('div', class_='list detail eplist')
            # find eoisode id
            episode_id_lists = [item.get('data-const') for item in episode_lists.find_all('div',class_='hover-over-image zero-z-index')]
            # combine url_head and review links
            episode_review_link_lists = [ url_head + item + '/reviews' for item in episode_id_lists ]
            # print(episode_review_link_lists)
        # set same id
        episode_id = [i for _ in range(len(episode_review_link_lists))]
        # print(episode_id)
        
        # get episode name
        episode_name_lists = episode_lists.find_all('strong')
        episode_names = [item.find('a').text for item in episode_name_lists]
        # print(episode_names)

        # get year
        episode_year_lists = episode_lists.find_all('div',class_='airdate')
        # split year number by Regex
        episode_years = [re.split('\s',item.text.strip())[-1] for item in episode_year_lists ]
        # print(episode_years)
        
        # extend lists: | name | season | review link | year |
        name_episode.extend(episode_names)
        season_num.extend(episode_id)
        review_link.extend(episode_review_link_lists)
        year.extend(episode_years)
    
    # (3) store data by pandas
    tv_series_data = pd.DataFrame({
        'name': name_episode,
        'season': season_num,
        'review-link': review_link,
        'year': year
    })
    print(tv_series_data.info())
    # save data into data.csv
    tv_series_data.to_csv('data.csv')
    print('[Finished] saved data.csv')

def scrape_all_reviews():
    '''
    The method is to scrape all reviews info from data.csv
    '''
    # use pandas to read data.csv 
    data_csv = pd.read_csv('data.csv')
    # print(data_csv['name'])

    # lists to store review 
    review_score_lists = [] # store scores
    review_comment_lists = [] # store comments
    num_episode = [] # store episode num
    score_positive_negative = [] # 1 for positive, 0 for negative
    # scrape each review link
    for index, review_url in enumerate(data_csv['review-link']):
        # get response.text
        response_text = request_url(review_url)
        # parse response.text by creating a BeautifulSoup Object, and assign this object to html_soup
        html_soup = BeautifulSoup(response_text,'html.parser')

        # get all reviews of this link (avoid "Warning:spoilers")
        review_lists = html_soup.find_all('div',class_='lister-item mode-detail imdb-user-review collapsable')
        # print(review_lists[0])
        
        # find review score (>= 8.0 is positive, otherwise is negative.) in <span>
        # review_score_lists = [item.find('span',class_='rating-other-user-rating').find('span').text for item in star_mark_lists]
        # print(review_score_lists[0])
        for review_item in review_lists:
            # check if the review has scores or not
            if review_item.find_all('div',class_='ipl-ratings-bar'):
                star_mark = review_item.find('div',class_='ipl-ratings-bar')
                # if there is review scores, then store it
                review_score_lists.append(int(star_mark.find('span',class_='rating-other-user-rating').find('span').text))
                
                # if score >= 8 is positive (1 mark), otherwise negative (0 mark)
                if review_score_lists[-1] >= 8:
                    score_positive_negative.append(1)
                else:
                    score_positive_negative.append(0)

                # store comments
                review_comment_lists.append(review_item.find('div',class_='text show-more__control').text)
                # store episode num
                num_episode.append(index)

        # print(review_score_lists)  
        # print(review_comment_lists) 

    # save pandas dataframe
    reviews_data = pd.DataFrame({
        'num-episode' : num_episode,
        'review-score' : review_score_lists,
        'positive-negative' : score_positive_negative,
        'review-comment' : review_comment_lists
    })
    print(reviews_data.info())

    # save data into review_data.csv
    reviews_data.to_csv('review_data.csv')
    print('[Finished] store data in review_data.csv ')
    # reorder data by 'positive-negative' 
    order_data = reviews_data.sort_values(by='positive-negative',ascending=True)
    order_data.to_csv('review_data_ordered.csv')
    print('[Finished] ordered data in review_data.csv ')


if __name__ == '__main__':
    while True:
        tv_url = download_menu()
        if tv_url != None:
            # run scrape seasons
            start_scrape_seasons(tv_url)
        
    
