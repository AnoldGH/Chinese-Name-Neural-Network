class Tokenizer:
    def tokenize(self, sentence: str):
        pass
    def get_tokens(self):
        return self.tokens

class TrieTokenizer(Tokenizer):
    class TrieNode:
        def __init__(self, char, isLeaf=False) -> None:
            self.char = char
            self.isLeaf = isLeaf
            self.children = {}
            
        def hasChild(self, char: str):
            return char in self.children
        
        def getChild(self, char: str):
            return self.children[char]
    
    def __init__(self, tokenFile=None) -> None:
        self.root = self.TrieNode("")
        
        if tokenFile is not None:
            with open(tokenFile, 'r') as file:
                self.tokens = set(file.readlines().split())
        else: self.tokens = set()
        
        for token in self.tokens:
            self.insert(token)
    
    def insert(self, word: str):
        curr = self.root
        
        for char in word:
            if char not in curr.children:
                curr.children[char] = self.TrieNode(char)
            curr = curr.children[char]
            
        curr.isLeaf = True
        
    def insertFromFile(self, filename: str):
        with open(filename, 'r', encoding='utf-8') as file:
            for token in file.readlines():
                self.insert(token.strip())
                self.tokens.add(token.strip())
        
    def tokenize(self, sentence: str):
        tokens = []
            
        i = 0
        while (i < len(sentence)):
            char = sentence[i]
            
            if self.root.hasChild(char):
                next_token = self.dfs(i, sentence, self.root, '')
                if next_token is None: next_token = char
            else: next_token = char
            
            tokens.append(next_token)
            i += len(next_token)
        
        return tokens
                
    def dfs(self, loc: int, sentence: str, curr: TrieNode, result: str):
        # End conditions
        if loc >= len(sentence) \
            or not curr.hasChild(sentence[loc]):
                if curr.isLeaf:
                    return result
                else: return None

        curr_char = sentence[loc]
        
        # Trying to find a longer match
        match = self.dfs(loc + 1, sentence, \
            curr.getChild(curr_char), result + curr_char)
        
        if match is not None:
            return match
        
        # Else, try to return current match
        else: 
            if curr.isLeaf: 
                return result
            else: return None
        
        
if __name__ == '__main__':
    
    tokenizer = TrieTokenizer()
    tokenizer.insertFromFile('special_tokens.txt')
    
    sentence = "前郭尔罗斯蒙古族自治县国有林总场"
    print(tokenizer.tokenize(sentence))