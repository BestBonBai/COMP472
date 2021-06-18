## Assignment 2

### Required Environment
- python 3.7
- conda 
    - `conda active py37`

### lists of libaries
- matplotlib
- numpy
- sys
- math (for module calculation)
- copy (for deepcopy)
- re (for regex)

- requests (for scrape)
- lxml
- bs4 import BeautifulSoup

- pandas 
- matplotlib.pyplot 
- tqdm (for progressbar)

- nltk
nltk.download('punkt')
nltk.download('stopwords')
- nltk.tokenize import word_tokenize
- nltk.corpus import stopwords

### File Structures
```
|---[A2]
    |-----scrape_tv_series.py
    |-----run_model.py
    |-----data.csv
    |-----review_data.csv
    |-----review_data_ordered.csv
    |-----remove.txt
    |-----model.txt
    |-----result.txt
    |-----frequency-model.txt
    |-----frequency-result.txt
    |-----smooth-model.txt
    |-----smooth-result.txt
    |-----length-model.txt
    |-----length-result.txt
    |-----[images]
        |-------task_2_1_graph.jpg
        |-------task_2_2_graph.jpg
        |-------task_2_3_graph.jpg
    |-----Expectation-of-Originality-40053833.pdf
    |-----Expectation-of-Originality-40079344.pdf
    |-----Expectation-of-Originality-40000000.pdf
```

### Run Instructions:

#### first method:
1. cd into `A2` folder
2. run `scrape_tv_series.py` using `python3 scrape_tv_series.py`
3. following the input requirements in the terminal
    1. input number `1` or `2` to scrape your favorite tv series, that will save in `data.csv`: 
        - [1]: The 100, [2]: Type URL by myself, [3]: Scrape reviews in `data.csv`, [9]: Exit
    2. input number `3` to scrape reviews in `data.csv` (if you have already has `data.csv` file)
        - when finished, `review_data.csv` and `review_data_ordered.csv` will be saved.
    3. input number `9` to exit.
4. run `run_model.py` using `python3 run_model.py`
5. following the input requirements in the terminal, after finishing the preprocessing
    1. input number `1`, `2` or `3` to choose different task
        - [1 : Task 2.1, 2 : Task 2.2, 3 : Task 2.3, 9 : Exit]
    2. After finished a Task, you have two methods to do other task:
        - input number `9` to `exit` and repeat the step 4 above
        - input number `1` , `2` or `3` to do another task.
6. check the graph images in `images` folder
