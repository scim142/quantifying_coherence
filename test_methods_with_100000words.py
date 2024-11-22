import numpy as np

def find_masked_letter(word, words):
    #word is a 5 letter string
    start = word[:2]
    end = word[-2:]
    h1 = [] #the list of letters * that match the pattern __* where __ are the first two letters of word
    h2 = []#the list of letters * that match the pattern *__ where __ are the last two letters of word
    for word1 in words: #iterate through list of words
        i = 0
        while start in word1[i:-1]: #add letter after start to h2 whenever it occurs in word1
            i = word1[i:-1].index(start) + 2 + i
            h1.append(word1[i])

        i = 1
        while end in word1[i:]: #add letter before end to h2 whenever it occurs in word1
            i = word1[i:].index(end) + 2 + i
            h2.append(word1[i - 3]) 

    h1Letters = set(h1)
    h2Letters = set(h2)
    letters = h1Letters.intersection(h2Letters) #common letters from both matchings

    h1Counts = {} #will be a dictionary of the form *: number of appearances of * in h1Letters
    s1 = 0
    for a in letters:
        s1 += h1.count(a)
        h1Counts[a] = h1.count(a)

    s2 = 0
    h2Counts = {} #will be a dictionary of the form *: number of appearances of * in h2Letters
    for a in letters:
        s2 += h2.count(a)
        h2Counts[a] = h2.count(a)

    h1Probs = {a:h1Counts[a]/s1 for a in h1Counts}
    h2Probs = {a:h2Counts[a]/s2 for a in h1Counts}

    geoms = {a: np.sqrt(h1Probs[a] * h2Probs[a]) for a in h1Counts} #p^* for assymtric loss function with f
    odds = {a: 10000 if (1-h1Probs[a])*(1-h2Probs[a]) == 0 else np.sqrt(h1Probs[a] * h2Probs[a] / (1-h1Probs[a])/(1-h2Probs[a])) for a in h1Counts} #p^* for assymtric loss function with f
    means = {a: (h1Probs[a] + h2Probs[a]) / 2 for a in h1Counts} ##p^* for either loss function with fo

    guesses = np.array([max(geoms, key=geoms.get), max(odds, key=odds.get), max(means, key=means.get)]) #the most likely letter for each loss function
    return guesses == word[2] #for each method, whether or not that method correctly predicted the middle letter


def find_accuracy_on_file(file):
    with open(file, 'r') as f:
        words = f.read().split("\n")
        num = 0
        right = np.array([0, 0, 0])
        for word in words: #could be rewriten to run in O(n) rather than O(n^2) time for larger word lists
            if len(word) == 5:
                num += 1
                right += find_masked_letter(word, words)
        return (right / num)


if __name__ == "__main__":
    print(find_accuracy_on_file('coherent_code/10000words.txt'))


