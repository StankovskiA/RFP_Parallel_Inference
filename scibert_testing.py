from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from heapq import heappush, heappop
from tika import parser
from tqdm import tqdm
import pandas as pd
import threading
import argparse
import logging
import csv
import os
import re

logging.basicConfig(filename='rfp_script.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GITHUB_REGEX = r'(?:https?://(?:www\.)?)?github\.com\s*/\s*[a-zA-Z0-9_. -]+\s*/\s*[a-zA-Z0-9_.-]+'
CODE_GOOGLE_REGEX = r'(?:https?://(?:www\.)?)?code\.google\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+'
GITLAB_REGEX = r'(?:https?://(?:www\.)?)?gitlab\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+'
REPO_REGEXES = [GITHUB_REGEX, GITLAB_REGEX]

SENTENCE_LEN_MAX_LIMIT = 250 # Maximum length of a sentence for analysis
SENTENCE_LEN_MIN_LIMIT = 30 # Maximum length of a sentence for analysis
FOOTNOTE_NUM_LIMIT = 30 # Numbers higher than this are not considered as footnotes
UNIQUE_NUM_LIMIT = 10 # Maximum size of unique numbers in a sentence - candidates as footnotes

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModelForSequenceClassification.from_pretrained("oeg/SciBERT-Repository-Proposal")

def read_pdf_list(pdf_path):
    try:
        raw = parser.from_file(pdf_path)
        list_pdf_data = raw['content'].split('\n\n')
        # delete empty lines
        list_pdf_data = [x for x in list_pdf_data if x != '']

        return list_pdf_data

    except FileNotFoundError:
        print(f"PDF file not found at path: {pdf_path}")
        return []
    except Exception as e:
        print(f"An error occurred while reading the PDF: {str(e)}")
        return []
    
def find_top_sentences(sentences, top_k=5):
    top_sentences = []  # Using a min heap to efficiently keep track of top sentences

    for sentence in sentences:
        # Remove sentences with more than 1 link
        if len(re.findall(r'(http?://\S+)', sentence)) > 1:
            continue
        
        try:
            inputs = tokenizer(sentence, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            scores = softmax(logits, dim=1).detach().numpy()[0]
            positive_score = scores[1]
            
            # If the heap is not full or the current sentence has a higher score than the smallest score in the heap
            if len(top_sentences) < top_k or positive_score > top_sentences[0][0]:
                heappush(top_sentences, (positive_score, sentence))
                
                # If the heap size exceeds top_k, remove the smallest element
                if len(top_sentences) > top_k:
                    heappop(top_sentences)
        except Exception as e:
            continue

    # Extract sentences from the heap in descending order of score
    sorted_sentences = [sentence for _, sentence in sorted(top_sentences, key=lambda x: x[0], reverse=True)]
    
    return sorted_sentences

# Look for github links
def get_repo_links(text):
    found_links = []
    for regex in REPO_REGEXES:
        found_links += re.findall(regex, text)
    return found_links

def extract_footnotes(pdf_list):
    LINK_REGEX = r'\b(\d+)\s*(https?://(?:www\.)?(?:github\.com|gitlab\.com)/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)\b'

    matches_dict = {}
    for sentence in pdf_list:
        matches = re.findall(LINK_REGEX, sentence)
        sentence_matches = {number: link for number, link in matches}
        matches_dict.update(sentence_matches)

    return matches_dict

# Look for github links that follow a number
def extract_link_by_number(sentence, target_number):
    LINK_REGEX = re.escape(target_number) + r'\s*(https?://(?:www\.)?(?:github\.com|gitlab\.com)/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+))'

    match = re.search(LINK_REGEX, sentence)

    if match:
        # print(f"Got link {match.group(1)} for number {target_number}")
        return match.group(1)
    else:
        return None   
    
# Get sentences that have number
def get_sentences_with_footnote(footnotes, sentences, best_sentences):
    sentences_with_footnote = []
    for footnote in footnotes:
        for sentence in sentences:
            if str(footnote) in sentence \
                and '.com' in sentence \
                and ('github' in sentence or 'gitlab' in sentence or 'code.google' in sentence) \
                and sentence not in best_sentences:
                    link = extract_link_by_number(sentence, str(footnote))
                    sentences_with_footnote.append((sentence, link))
                
    return sentences_with_footnote

def clean_final_sentence(sentence):
    # Remove newline characters
    sentence = sentence.replace('\n', ' ')
    
    # Use regular expression to replace multiple spaces with a single space
    sentence = re.sub(r'\s+', ' ', sentence)
    
    # Remove leading and trailing spaces
    sentence = sentence.strip()
    
    # Remove extra spaces in links
    sentence = sentence.replace(". com", ".com").replace("/ ", "/").replace(" /", "/")
    
    # Replace badly read h�ps with https
    sentence = sentence.replace("h�p", "http")
    
    # Replace word breaks
    sentence = sentence.replace("- ", "-")
    
    return sentence

def extract_references(pdf_list):
    references, not_references = {}, []
    skip_until = None
    pattern = r'^\[\d{1,2}\]\s'

    for id, paragraph in enumerate(pdf_list):
        if len(paragraph.replace(" ", '').replace('\n','')) < 10:
            continue
        
        if skip_until is not None and id < skip_until:
            continue
        else:
            skip_until = None
            
        # Check if the paragraph begins with number in a bracket
        if re.match(pattern, paragraph):
            next_paragraph_id = None
            # Search at most 10 next paragraphs until the next paragraph that begins with brackets
            for i in range(1, 10):
                if id+i < len(pdf_list) and pdf_list[id+i][0] == '[':
                    next_paragraph_id = id+i
                    break
            
            next_paragraph_id = next_paragraph_id if next_paragraph_id is not None else id+3 # Most likely the last reference
            
            # Concatenate the paragraphs until the next paragraph that begins with brackets
            paragraph = ' '.join(pdf_list[id:next_paragraph_id])
            
            # Remove newline characters
            paragraph = paragraph.replace('\n', ' ')
            
            # Use regular expression to replace multiple spaces with a single space
            paragraph = re.sub(r'\s+', ' ', paragraph)
            
            # Remove leading and trailing spaces
            paragraph = paragraph.strip()
            
            # Skip the next paragraphs
            skip_until = next_paragraph_id
              
            paragraph = clean_final_sentence(paragraph)
            
            # for x in range(id, next_paragraph_id):
            #     print(f"Skipping sentence: {pdf_list[x]}")
            
            # The paragraph is a reference in the format of [number] Author, Title, Journal, Year
            # Use the [number] as key and the rest as value for the references dictionary
            references[f"[{paragraph.split(']')[0][1:]}]"] = paragraph.replace(f"[{paragraph.split(']')[0][1:]}] ", '').strip()
        else:
            not_references.append(paragraph)
            
    return references, not_references

def extract_full_sentences(pdf_list):
    full_sentences = []
    not_final_sentences = []
    for paragraph in pdf_list:
        if len(paragraph.replace(" ", '').replace('\n','')) < 10:
            continue

        # Split sentences with period and capital letter
        paragraph = re.sub(r'(\.\s)([A-Z])', r'\1\n \2', paragraph) 
        
        if '\n' in paragraph:
            paragraph = clean_final_sentence(paragraph)
            
            new_paragraph = []
            for sentence in paragraph.replace('.', '..').split('. '):
                if not sentence and len(sentence) < 1:
                    continue
                
                # Last item may have two periods
                sentence = sentence.replace('..', '.')
                
                # If sentence begins with a capital letter and ends with a period, add to new_paragraph
                try:
                    if sentence[0].isupper() and sentence[-1] == '.' and len(sentence) > 20:
                        full_sentences.append(sentence)
                    else:
                        new_paragraph.append(sentence)
                except Exception as e:
                    pass
                    
            if new_paragraph:
                not_final_sentences.extend(new_paragraph)
        elif paragraph[0].isupper() and paragraph[-1] == '.' and len(paragraph) > 20:
            full_sentences.append(paragraph)
            paragraph = ''
        elif re.match(r'^\[\d{1,2}\]\s', paragraph) and paragraph[-1] == '.' and len(paragraph) > 20: 
            full_sentences.append(paragraph)
            paragraph = ''
        else:
            not_final_sentences.append(paragraph)
                                
    return full_sentences, not_final_sentences

def combine_split_sentences(not_final_sentences):
    final_sentences, uncombined_sentences = [], []
    # Try to combine split sentences
    skip_next = False
    for id, paragraph in enumerate(list(not_final_sentences)):
        if skip_next:
            skip_next = False
            continue
        
        # Check if first letter is capital, and first letter of next paragraph is lowercase and last is period
        if paragraph[-1] not in ['.', ':'] and paragraph[0].isupper():
            if id < len(not_final_sentences) - 1 and not_final_sentences[id+1][-1] == '.':
                sentence = paragraph + ' ' + not_final_sentences[id+1]
                
                # Use regular expression to replace multiple spaces with a single space
                sentence = re.sub(r'\s+', ' ', sentence)
                
                # Remove word break hyphen if its not a link, else just replace the space
                sentence = sentence.replace('- ', '')
                
                # Clean the sentence
                sentence = clean_final_sentence(sentence)
                
                final_sentences.append(sentence)               
                skip_next = True
            else:
                uncombined_sentences.append(paragraph)
        else:
            uncombined_sentences.append(paragraph)
                
    # Replaces word breaks in the sentences          
    combined_not_final_sentences = '#JOIN#'.join(uncombined_sentences).replace('-#JOIN# ', '').split('#JOIN#')
    combined_not_final_sentences = ' '.join(combined_not_final_sentences).replace('. ', '.. ').split('. ')    
    combined_not_final_sentences = [clean_final_sentence(s) for s in combined_not_final_sentences if len(s) > 20]
        
    return final_sentences, combined_not_final_sentences

def get_sentences(pdf_path):
    # Read the pdf file with Tika
    pdf_list = read_pdf_list(pdf_path)
    
    # Extract reference sentences and non-reference sentences
    references, not_final_sentences = extract_references(pdf_list)

    # Extract full sentences which begin with a capital letter and end with a period
    final_sentences, not_final_sentences = extract_full_sentences(not_final_sentences)

    combined_final, not_final_sentences = combine_split_sentences(not_final_sentences)   
    
    final_sentences.extend(combined_final)    
        
    footnotes = extract_footnotes(final_sentences + not_final_sentences + list(references.values()))
        
    return references, footnotes, final_sentences + not_final_sentences

def process_files(thread_id, file_paths, output_path):
    for id, file in enumerate(file_paths):
        logging.info(f"Thread {thread_id} processing file {id+1} of {len(file_paths)}")
        references, footnotes, sentences = get_sentences(file)
        best_sentences = find_top_sentences(sentences)
        
        link, final_sentence = '', ''
        all_footnotes = []
        reference_numbers = []
        for sentence in best_sentences:
            # Look for github links
            repo_links = get_repo_links(sentence)
            if repo_links:
                link = repo_links[0]
                final_sentence = sentence
                # print(f'Found {link} in {sentence}')
                break
            else:
                # Use regular expression to find numbers attached to words
                # Ensure non-duplication
                square_brackets = re.findall(r'\[\d+\]', sentence)
                if square_brackets:
                    reference_numbers.extend(square_brackets)
                
                numbers = list(set(re.findall(r'(\[\d+\]|\d+)\S*\b', sentence)))
                
                # Remove numbers greater than 30
                numbers = [num for num in numbers if '[' not in num and 0 < int(num) <= FOOTNOTE_NUM_LIMIT]
                
                # If more than 5 numbers, unlikely to be footnotes
                numbers = numbers if len(numbers) <= 5 else []
                
                # Use regular expression to find special characters used as footnotes
                extra_chars = list(set(re.findall(r'[†‡*]', sentence)))
                all_footnotes.extend(numbers+extra_chars)

        # Remove duplicates in all_footnotes while keeping order
        all_footnotes = list(dict.fromkeys(all_footnotes))

        if not link and reference_numbers: # No link found in best matches, look for references or footnotes
            # print(f'No link found in best matches')
            if reference_numbers:
                # print(f'Reference numbers found: {reference_numbers}')
                for ref in reference_numbers:
                    if ref in references:
                        repo_links = get_repo_links(references[ref])
                        if repo_links:
                            link = repo_links[0]
                            final_sentence = references[ref]
                            # print(f'Found {link} in {ref} from references')
                        break
                    
        if not link and all_footnotes:
            # Look for footnotes in the footnotes dictionary
            for f in all_footnotes:
                if f in footnotes:
                    link = footnotes[f]
                    final_sentence = footnotes[f]
                    # print(f'Found {link} in {f} from footnotes dictionary')
                    break

            sentences_with_footnote = get_sentences_with_footnote(all_footnotes, sentences, best_sentences)

            # Look for link attached to number in sentence
            if not link:
                for sentence_with_number, footnote_link in sentences_with_footnote:
                    # print(f"Sentence with number: {sentence_with_number}")
                    if footnote_link:
                        link = footnote_link
                        final_sentence = sentence_with_number
                        # print(f'Found {link} in {sentence_with_number} from footnote')
                        break
                
            # Look for sentence with number
            if not link:
                for sentence_with_number, footnote_link in sentences_with_footnote:
                    repo_links = get_repo_links(sentence_with_number)
                    if repo_links:
                        link = repo_links[0]
                        final_sentence = sentence_with_number
                        # print(f'Found {link} in {sentence_with_number} from footnote')
                        break
            
        if link:
            # If link ends with a dot remove it
            if link[-1] == '.':
                link = link[:-1]
            
            # If the link ends with .git remove it
            if link[-4:] == '.git':
                link = link[:-4]

        with open(output_path, "a", newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([file, link])
            logging.info(f"Thread {thread_id} finished processing file {id+1} of {len(file_paths)}")

import os
import re

def get_latest_versions(main_folder_path):
    # Dictionary to store the latest version of each folder
    latest_versions = {}

    # Iterate through all subdirectories inside the main folder
    for root, dirs, _ in os.walk(main_folder_path):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            
            # Extract version information using regular expression
            match = re.match(r'.*v(\d+)$', folder)
            if match:
                version = int(match.group(1))
                # Update the latest version if the current version is greater
                latest_versions[folder_path] = max(version, latest_versions.get(folder_path, 0))

    # Form the paths to the latest version PDF files
    pdf_paths = [os.path.join(folder_path, folder.replace(".pdf", ""), f"{folder}.pdf") for folder, version in latest_versions.items()]

    return pdf_paths

def main(folder_path: str, output: str, first_write: bool) -> None:
    if first_write:
        with open(output, "w", newline='') as csvfile:
            line = ["DOI", "Link"]
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(line)
        existing_dois = []
    else:
        df = pd.read_csv(output)
        existing_dois = set(df['DOI'])
        
    filenames = get_latest_versions(folder_path)
       
    # Get filenames, excluding those with existing DOIs
    # filenames = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
    
    filenames = [filename for filename in filenames if filename not in existing_dois]   
    
    # Split files among threads
    files_per_thread = len(filenames) // 5
    file_chunks = [filenames[i:i + files_per_thread] for i in range(0, len(filenames), files_per_thread)]
    
    # Create and start threads
    threads = []
    for i, files in enumerate(file_chunks, start=1):
        thread = threading.Thread(target=process_files, args=(i, files, output))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()        

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Description of your script.")

    # Add command-line arguments
    argparser.add_argument("--path", type=str, help="Path to the folder containing the PDF files", required=True)
    argparser.add_argument("--output", type=str, help="Name of output file", required=True)
    argparser.add_argument("--first_write", type=bool, default=True, required=False)

    # Parse command-line arguments
    args = argparser.parse_args()

    # Call the main function with parsed arguments
    main(folder_path=args.path, output=args.output, first_write=args.first_write)
