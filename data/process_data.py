# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
	"""
	Picks up the data from messages and categories csv files and merges them into a dataframe
	"""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df


def clean_data(df):
	"""
	Cleans the merged df so that ML algorithm can be implemented on the data.
	"""
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[len(x)-1:len(x)])
    
    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
        
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
        
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
	"""
	Saves the cleaned dataframe to a sql database
	"""
    link='sqlite:///'+database_filename
    engine = create_engine(link)
    df.to_sql('msg_ctg', engine, index=False, if_exists='replace')


def main():
	"""
	Calls the various functions to load, clean and save data respectively
	"""
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()