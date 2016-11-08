from collections import Counter
import numpy as np
from itertools import product, permutations
from queue import PriorityQueue
from time import time
from sklearn.metrics import accuracy_score

start_mark = "<s>"
end_mark = "<\s>"
dummy_value = 1e-10

def combine_params(params_dict):
    keys, values = params_dict.keys(), params_dict.values()
    return [dict(zip(keys, items)) for items in product(*values)]


def _fit_and_score(model, params, corpus, valid):
    print params,
    model.reset(**params)
    score = model.fit(corpus).score_corpus(valid)
    print score
    return score


def isclose(a, b, rel_tol=1e-05, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class GridSearch:
    def __init__(self, model, grid):
        self.model = model
        self.grid = grid
        self.best_model = None
        self.best_params = None

    def fit(self, corpus, valid=None):
        if valid is None:
            valid = corpus

        model = self.model
        all_params = self.grid
        outs = [_fit_and_score(model, learning_params, corpus, valid)
                for learning_params in all_params]

        best_score = None
        for score, learning_params in zip(outs, all_params):
            if (best_score is None) or score < best_score:
                self.best_params = learning_params
                best_score = score

        model.reset(**self.best_params)
        self.best_model = model.fit(corpus)
        return self

    def predict(self, texte):
        return self.best_model.predict_corpus(texte)

    def score(self, texte):
        return self.best_model.score_corpus(texte)


class Ngram_model:
    def __init__(self, n, lissage=None, delta=None, lambdas=None):
        self.reset(n, lissage, delta, lambdas)

    def reset(self, n, lissage=None, delta=None, lambdas=None):
        self.n = n
        if lissage is None:
            self.prob = self.__prob
        elif lissage.upper() == "LAPLACE":
            self.prob = self.__prob_laplace
            if (delta is None) or (not (0 <= delta <= 1)):
                raise Exception("delta should be a value between 0 and 1")
            delta *= 1.0
        elif lissage.upper() == "INTERPOLATION":
            self.prob = self.__prob_interpolation
            if ((lambdas is None) or (not isinstance(lambdas, (list, tuple, np.ndarray)))
                or (len(lambdas) != n) or (not isclose(np.sum(lambdas),1.0))):
                raise Exception("lambdas should be a list or a tuple of n coefficients that sum to 1")
        else:
            raise Exception("lissage must be None if you don't want to do a lissage. "
                            "Otherwise, it should be 'LAPLACE' or 'INTERPOLATION'")
        self.lissage = lissage
        self.delta = delta
        self.lambdas = lambdas

    def fit(self, corpus):
        temp = [x for line in corpus for x in line]
        self.nb_total_words = len(temp)
        self.vocab_size = len(set(temp))
        del temp

        corpus2 = [(([start_mark] * (self.n - 1)) + line + ([end_mark] * (self.n - 1)))
                   for line in corpus]
        self.counts_in_corpus = Counter([" ".join(line[i: i + ng])
                                         for line in corpus2
                                         for ng in range(1, self.n+1)
                                         for i in range(len(line) - ng + 1)])
        return self

    def __prob(self, x, history=""):
        assert len(x.split(" ")) == 1, "x should be a single word"
        assert history == "" or (history != "" and len(history.split(" ")) < self.n), "The history size should be less our n-gram size"

        if (history is None) or (history == ""):
            if self.counts_in_corpus[x] == 0:
                return dummy_value
            return (self.counts_in_corpus[x] * 1.0) / self.nb_total_words
        else:
            if self.counts_in_corpus[history + " " + x] == 0:
                return 0
            return (self.counts_in_corpus[history + " " + x] * 1.0) / self.counts_in_corpus[history]

    def __prob_laplace(self, x, history=""):
        assert len(x.split(" ")) == 1, "x should be a single word"
        assert history == "" or (history != "" and len(history.split(" ")) < self.n), "The history size should be less our n-gram size"
        if (history is None) or (history == ""):
            return ((self.counts_in_corpus[x] + self.delta) /
                    (self.nb_total_words + self.vocab_size*self.delta))
        else:
            return ((self.counts_in_corpus[history + " " + x] + self.delta) /
                    (self.counts_in_corpus[history] + self.vocab_size*self.delta))

    def __prob_interpolation(self, x, history=""):
        assert len(x.split(" ")) == 1, "x should be a single word"
        assert history == "" or (history != "" and len(history.split(" ")) < self.n), "The history size should be less our n-gram size"

        hist_list = history.split(" ")
        len_hist = len(hist_list)
        if history == "":
            return self.__prob(x)*self.lambdas[-1]
        return np.sum([self.__prob(x, " ".join(hist_list[i:])) * self.lambdas[i + (self.n - len_hist - 1)]
                       for i in range(len_hist+1)])

    def log_prob(self, x, history=""):
        assert self.prob(x, history) <= 1 or self.prob(x, history) >= 0, "{}: {}, {}, {}".format(self.prob(x, history), x, history, self.lissage)
        return np.log(self.prob(x, history))

    def predict(self, phrase):
        texte = (([start_mark] * (self.n - 1)) + phrase + ([end_mark] * (self.n - 1)))
        log_prob = np.sum([self.log_prob(el, history=" ".join(texte[i - self.n + 1:i]))
                           for i, el in enumerate(texte) if i >= self.n - 1])
        return log_prob

    def predict_sequence(self, phrase):
        log_prob = np.sum([self.log_prob(el, history=" ".join(phrase[i - self.n + 1:i]))
                           for i, el in enumerate(phrase) if i >= self.n - 1])
        for i in range(self.n-1):
            log_prob += self.log_prob(phrase[i], history=" ".join(phrase[:i]))
        return log_prob

    def predict_corpus(self, corpus):
        log_prob = np.sum([self.predict(line) for line in corpus])
        return log_prob

    def score(self, phrase):
        n = np.sum([True for _ in phrase])
        return np.exp((-1.0/n) * self.predict(phrase))

    def score_sequence(self, phrase):
        n = np.sum([True for _ in phrase])
        return np.exp((-1.0/n) * self.predict_sequence(phrase))

    def score_corpus(self, corpus):
        n = np.sum([True for line in corpus for _ in line])
        return np.exp((-1.0/n) * self.predict_corpus(corpus))

    def find_order_slow(self, list_of_words):
        if len(list_of_words) > 9:
            return []
        best_option = []
        best_score = np.inf
        for option in permutations(list_of_words):
            score = self.score_sequence(list(option))
            # print option, score, self.predict_sequence(list(option))
            if best_score > score:
                best_score = score
                best_option = option

        return best_option

    def find_order(self, list_of_words):
        l = len(list_of_words)
        best_so_far, best_so_far_cost = [], -np.inf

        dw = dict(zip(list_of_words, range(l)))
        pairwise = np.zeros((l, l))
        for choice, history in product(range(l), repeat=2):
            pairwise[choice, history] = self.log_prob(list_of_words[choice], list_of_words[history])

        if self.n == 3:
            tripletwise = np.zeros((l, l, l))
            for choice, hist1, hist2 in product(range(l), repeat=3):
                tripletwise[choice, hist1, hist2] = self.log_prob(list_of_words[choice],
                                                                  list_of_words[hist2] + " " + list_of_words[hist1])
        q = PriorityQueue(-1)
        q.put_nowait((0, []))
        t = time()
        while not q.empty():
            partial_cost, partial_solution = q.get_nowait()
            current_len = len(partial_solution)

            if current_len == l and partial_cost > best_so_far_cost:
                best_so_far = partial_solution
                best_so_far_cost = partial_cost

            options = list_of_words + []
            _ = [options.remove(x) for x in partial_solution]
            for word in options:
                if current_len == 0:
                    cost = self.log_prob(word)
                elif current_len == 1 and self.n == 3:
                    cost = partial_cost + self.log_prob(word, partial_solution[-1])
                elif self.n == 2:
                    cost = partial_cost + pairwise[dw[word], dw[partial_solution[-1]]]
                else:
                    cost = partial_cost + tripletwise[dw[word], dw[partial_solution[-1]], dw[partial_solution[-2]]]

                if cost > best_so_far_cost:
                    q.put_nowait((cost, partial_solution + [word]))
        print "Completed in {:.3f} seconds".format(time() - t)
        return best_so_far

if __name__ == "__main__":
    train_proportion = 1

    params_grid_laplace = combine_params({
        "n": [2, 3],
        "lissage": ["laplace"],
        "delta": [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1]
    })
    params_grid_interpol = [{
        "n": 1,
        "lissage": "interpolation",
        "lambdas": [1]
    }]
    for n in [2, 3]:
        temp = np.random.randint(1, 1000, size=(20, n))*1.0
        params_grid_interpol += combine_params({
            "n": [n],
            "lissage": ["interpolation"],
            "lambdas": (temp.T / temp.sum(axis=1)).T
        })

    # for lissage_avec_laplace in [True, False]:
    for lissage_avec_laplace in [False]:
        if lissage_avec_laplace:
            params_grid = params_grid_laplace
            print "#"*40, "\n### {:32s} ###\n".format("Lissage de Laplace"), "#"*40
        else:
            params_grid = params_grid_interpol
            print "#"*40, "\n### {:32s} ###\n".format("Lissage par interpolation"), "#"*40

        # Construction of n_grams models
        with open("corpus_small.txt") as fichier_corpus:
            corpus = [line[:-1].split(" ") for line in fichier_corpus.readlines()]
            n_train = int(train_proportion * len(corpus))
            corpus_train, corpus_valid = corpus[:n_train], corpus[n_train:]

        # construction du modele avec le lissage de laplace
        grid = GridSearch(Ngram_model(1), params_grid)
        grid.fit(corpus_train, corpus_train)

        print grid.best_params
        with open("listes_en_desordre_solutions.txt") as fichier:
            temp = fichier.read().split("\n\n")[:]
            phrases = [[x.split(": ")[-1].split(" ") for x in el.split("\n")[:2]] for el in temp]
            phrases = sorted(phrases, cmp=(lambda a, b: len(a[0]) - len(b[0])))
            phrases = phrases[:-4]

        y_true, y_predict = ([], [])
        for phrase_desordre, solution in phrases:
            phrase_desordre = [x for x in phrase_desordre if x != '']
            solution = [x for x in solution if x != '']
            print "{:12}".format("WORDLIST:"), phrase_desordre
            print "{:12}".format("ORDERED:"), solution
            y_true += [" ".join(solution)]
            pred = grid.best_model.find_order(phrase_desordre)
            print "{:12}".format("PREDICTION:"), pred, pred == solution
            y_predict += [" ".join(pred)]
            print

        print "Accuracy score: ", accuracy_score(y_true, y_predict)






# {'n': 3, 'lissage': 'interpolation', 'lambdas': array([ 0.68114818,  0.29480217,  0.02404965])}