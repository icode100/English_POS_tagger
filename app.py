import pickle
import gradio as gr
from collections import defaultdict,Counter

class HMM:
    def __init__(self, dataset):
        # data processing
        self.state_tags = defaultdict(set)
        self.sentences = []
        self.state_seq = []
        self.vocab = set()
        self.tags = set()
        self.tag_count = defaultdict(int)
        self.word_count = defaultdict(int)
        self.tag_zipped_words = []
        from copy import deepcopy
        for item in dataset:
            self.sentences.append(item['sentence'])
            self.state_seq.append(item['labels'])
            self.tag_zipped_words.append(deepcopy(list(zip(item['sentence'], item['labels']))))
            for word, count in list(Counter(item['sentence']).items()):
                self.word_count[word] += count
            for tag, count in list(Counter(item['labels']).items()):
                self.tag_count[tag] += count
            self.vocab = self.vocab.union(set(item['sentence']))
            self.tags = self.tags.union(set(item['labels']))
            for i, word in enumerate(item['sentence']):
                self.state_tags[word].add(item['labels'][i])
        
        # parameters
        self.A = defaultdict(lambda: defaultdict(float))  # transition matrix
        self.B = defaultdict(lambda: defaultdict(float))  # emission matrix
        self.PI = defaultdict(float)

        for item in self.state_seq:
            for i_1, i in zip(item, item[1:]):
                self.A[i_1][i] += 1

        for item in self.A.keys():
            for key in self.A[item].keys():
                self.A[item][key] = self.A[item][key] / self.tag_count[item]

        self.PI = self.A['<s>']
        for item in self.tag_zipped_words:
            for obs, state in list(item):
                self.B[state][obs] += 1
        for item in self.B.keys():
            for key in self.B[item].keys():
                self.B[item][key] /= self.tag_count[item]

    def forward(self, sentence):
        sentence = sentence.strip().split()
        memo = defaultdict(lambda: defaultdict(tuple))
        w = sentence[0]
        for tags in self.state_tags[w]:
            memo[0][tags] = (self.PI[tags],'<s>')
        for i in range(1,len(sentence)):
            w = sentence[i]
            tags = self.state_tags[w]
            for tag in tags:
                emission = self.B[tag][w]
                memo[i][tag] = (-1e9,'')
                for t,(prior,path) in memo[i-1].items():
                    transition = self.A[t][tag]
                    curr_prob = transition * emission * prior
                    if curr_prob>memo[i][tag][0]:
                        memo[i][tag] = (curr_prob,f'{path},{tag}')
        n = len(sentence)
        res = ''
        check = -1e9
        for t,(prior,path) in memo[n-1].items():
            if prior>check:
                check = prior
                res = path
        return res

    def __getstate__(self):
        """Prepare for pickling."""
        state = self.__dict__.copy()
        # Convert defaultdicts to dicts for pickling
        state['state_tags'] = dict(state['state_tags'])
        state['A'] = {k: dict(v) for k, v in state['A'].items()}
        state['B'] = {k: dict(v) for k, v in state['B'].items()}
        state['PI'] = dict(state['PI'])
        return state

    def __setstate__(self, state):
        """Reconstruct from pickle."""
        self.__dict__.update(state)
        # Convert dicts back to defaultdicts if needed
        self.state_tags = defaultdict(set, self.state_tags)
        self.A = defaultdict(lambda: defaultdict(float), {k: defaultdict(float, v) for k, v in self.A.items()})
        self.B = defaultdict(lambda: defaultdict(float), {k: defaultdict(float, v) for k, v in self.B.items()})
        self.PI = defaultdict(float, self.PI)

with open('hmm_model.pkl', 'rb') as f:
    model = pickle.load(f)

def pos(sen):
    return model.forward(sen)

demo = gr.Interface(fn=pos, inputs="text", outputs="text")
demo.launch(share=True)   