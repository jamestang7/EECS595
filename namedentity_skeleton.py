import re


# For Programming Problem 1, we will use regular expressions to replace certain types of named entity substrings with special tokens. 
#
# Please implement the ner() function below, and feel free to use the re library. 
# DO NOT modify any function definitions or return types, as we will use these to grade your work. However, feel free to add new functions to the file to avoid redundant code.
#
# *** Don't forget to additionally submit a README_1 file as described in the assignment. ***


# Description: Transforms a string into a string with special tokens for specific types of named entities.
# Input: Any string.
# Output: The input string, with the below types of named entity substrings replaced by special tokens (<expression type>: "<token>").
# - Times: "TIME"
# - Dates: "DATE"
# - Email addresses: "EMAIL_ADDRESS"
# - Web addresses: "WEB_ADDRESS"
# - Dollar amounts: "DOLLAR_AMOUNT"
#
# Sample input => output: “she spent $149.99 and bought a nice microphone from www.bestdevices.com yesterday” => “she spent DOLLAR_AMOUNT and bought a nice microphone from WEB_ADDRESS DATE”
def ner(input_string):
    # TODO: implement the transformation of input_string
    dollar_pattern = re.compile('\$?[0-9]+\.[0-9]+(\sdollar[s]?|\scent[s]?)?')
    web_pattern = re.compile('www\.\S+')
    email_pattern = re.compile('\S+\@\S+\.\S+')
    date_pattern = re.compile('([a-zA-Z]+day|[tT]omorrow)')
    time_pattern =re.compile('((?:Jan(?:uary)?)|(?:Feb(?:ruary))|(?:Mar(?:ch)?)|(?:Apr(?:il)?)|(?:May)|(?:Jun(?:e)?)|(?:Jul(?:y)?)|(?:Aug(?:ust)?)|(?:Sep(?:tember))|(?:Oct(?:ober)?)|(?:Nov(?:vember)?)|(?:Dec(?:ember)?))(\.|,|/|-|\s)([0-9]?|[12][0-9]|30|31)(st|nd|rd|th)(\.|,|/|-|\s)([0-9]+)')
    map = {time_pattern:'TIME',
           date_pattern:"DATE",
           email_pattern:"EMAIL_ADDRESS",
           web_pattern:"WEB_ADDRESS",
           dollar_pattern:"DOLLAR_AMOUNT"}
    for pattern in [dollar_pattern,web_pattern,email_pattern,date_pattern,time_pattern]:
        input_string = pattern.sub(map[pattern],input_string)

                             
                             
                            
    return input_string # Feel free to modify this line if necessary

# GRADING: We will be importing and repeatedly calling your ner function from a separate script with various test case strings. For example (not exact):
# str1 = ner('she spent $149.99 and bought a nice microphone from www.bestdevices.com yesterday')
# if str1 == 'she spent DOLLAR_AMOUNT and bought a nice microphone from WEB_ADDRESS DATE':
#    correct = True