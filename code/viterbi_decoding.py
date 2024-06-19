from collections import defaultdict
def viterbi(sentence,A,B,PI,state_tags):
    sentence = sentence.strip().split()
    memo = defaultdict(lambda: defaultdict(tuple))
    w = sentence[0]
    for tags in state_tags[w]:
        memo[0][tags] = (PI[tags],'<s>')
    for i in range(1,len(sentence)):
        w = sentence[i]
        tags = state_tags[w]
        for tag in tags:
            emission = B[tag][w]
            memo[i][tag] = (-1e9,'')
            for t,(prior,path) in memo[i-1].items():
                transition = A[t][tag]
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
    