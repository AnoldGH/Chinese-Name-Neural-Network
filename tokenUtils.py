import tokenToolKit

# Parse and extract all chinese characters from a file and output to another
def extract_characters_to(input: str, output: str, encoding='utf-8'):
    characters = set()
    
    with open(input, 'r', encoding=encoding) as file:
        for line in file.readlines():
            for ch in line.strip():
                characters.add(ch)
                
    with open(output, 'w', encoding=encoding) as file:
        for ch in characters:
            file.write(ch)
            file.write('\n')

def extract_characters_to(inputs: tuple, output: str, encoding='utf-8'):
    characters = set()
    
    for input in inputs:
        with open(input, 'r', encoding=encoding) as file:
            for line in file.readlines():
                for ch in line.strip():
                    characters.add(ch)
                
    with open(output, 'w', encoding=encoding) as file:
        for ch in characters:
            file.write(ch)
            file.write('\n')

def extract_tokens_to(inputs: tuple, output: str, token_src: str, encoding='utf-8'):
    tokens = set()
    
    tokenizer = tokenToolKit.TrieTokenizer()
    tokenizer.insertFromFile(token_src)
    
    for input in inputs:
        with open(input, 'r', encoding=encoding) as file:
            for line in file.readlines():
                for token in tokenizer.tokenize(line.strip()):
                    tokens.add(token)
                
    with open(output, 'w', encoding=encoding) as file:
        for token in tokens:
            file.write(token)
            file.write('\n')
            
# Random Generator
def build_weighted_tables(inputs: tuple, encoding='utf-8'):
    freq_table = dict()
    for input in inputs:
        with open(input, 'r', encoding=encoding) as file:
            for line in file.readlines():
                token, freq = line.split()
                if freq_table.get(token) is None:
                    freq_table[token] = float(freq)
                else: freq_table[token] += float(freq)
                
    tokens_list, weights_list = list(), list()
    for token, weight in freq_table.items():
        tokens_list.append(token)
        weights_list.append(weight)
    return tokens_list, weights_list


if __name__ == '__main__':
    given_name_corpus = 'chinese_names_corpus.txt'
    family_name_corpus = 'chinese_family_names_corpus.txt'
    family_name_frequency = 'chinese_family_names_frequency.txt'
    output = 'characters.txt'
    
    inputs = (given_name_corpus, family_name_corpus)
    extract_tokens_to(inputs, output, family_name_corpus)
    token_list, weights_list = build_weighted_tables((family_name_frequency, ))
    print(token_list)
    print(weights_list)