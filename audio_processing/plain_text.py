import difflib
import string
import re
class PlainText:
    def __init__(self,file_path,file_type='txt'):
        pattern = re.compile(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ\s]')
        if file_type =='txt':
            with open(file_path, 'r', encoding='latin-1') as file:
                self.raw_text=file.read()
            self.text = [pattern.sub('', word) for word in self.raw_text.split()]
        else:
            self.text=None
    
    def word_lookup(self,words,sensitivity):
        for word in words:
            word_found=False
            for word_in_text in self.text:
                if difflib.SequenceMatcher(None, word, word_in_text).ratio() > sensitivity:
                    word_found=True
                    break
            if(word_found is True):
                break
        return word_found
    
    def _normalize_text(self,text):
        # Convert to lowercase
        text = text.lower()
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def sentence_lookup(self,sentences):
        raw_text_normilized=self._normalize_text(self.raw_text)
        sentence_found=False
        for sentence in sentences:
            sentence_normalized=self._normalize_text(sentence)
            if sentence_normalized in raw_text_normilized:
                sentence_found=True
                break
        return sentence_found

    def __call__(self):
        return self.text