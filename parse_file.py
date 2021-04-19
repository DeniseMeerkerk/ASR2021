# http://lands.let.ru.nl/cgn/doc_Dutch/topics/version_1.0/formats/text/plk.htm

import re


def read_lines(filename):
    """
        Input: filename of a file formatted in the .plk format
        Output: list of four tuples: utterance, utterance type denoted by interpunction sign, start time of utterance, end time of utterance
    """
    
    # Initialization
    sentence = "  "
    conv = []
    start_t = 0
    end_t = 0
    
    # Need special file encoding to work
    with open(filename, encoding="ISO-8859-1") as f:
        for line in f:
            # For a word line, this produces a list with the different information for a single word
            # For a change of utterance line, this produces a list with one single string with all the information (as that info is deliminated by spaces and not by tabs).
            spl = line.split("\t")  
            if spl[0].split(" ")[0] == "<au": #We use the made split and check if the first part is a change of utterance line. If it is, we wrap up
                time = re.findall(r"tb=\"(.*)\"", line)
                
                # Set end time (as we found the end of the utterance
                end_t = float(time[0])
                
                # Produces four tuple: sentence, punctuation sign, starting time, ending time
                conv.append((sentence[:-1], sentence[-2], start_t, end_t))
                
                # Reset starting time and sentence
                start_t = float(time[0])
                sentence = ""
            else:
                sentence += spl[0] + " " # Add the read word (always first thing on the line) and add a space to make it legible
    return conv[1:] # We return from 1 onwards as the first one is empty (the first line of the file is a change of utterance, so the code produces one empty sentence).

#print(read_lines("fn000006.plk"))
