import numpy as np


class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()


class Useless(Model):

    def __init__(self):
        super().__init__()
        self.reference_example = None
        self.reference_label = None

    def fit(self, X, y):
        self.num_input_features = X.shape[1]
        # Designate the first training example as the 'reference' example
        # It's shape is [1, num_features]
        self.reference_example = X[0, :]
        # Designate the first training label as the 'reference' label
        self.reference_label = y[0]
        self.opposite_label = 1 - self.reference_label

    def predict(self, X):
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')
        # Perhaps fewer features are seen at test time than train time, in
        # which case X.shape[1] < self.num_input_features. If this is the case,
        # we can simply 'grow' the rows of X with zeros. (The copy isn't
        # necessary here; it's just a simple way to avoid modifying the
        # argument X.)
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        # Or perhaps more features are seen at test time, in which case we will
        # simply ignore them.
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        # Compute the dot products between the reference example and X examples
        # The element-wise multiply relies on broadcasting; here, it's as if we first
        # replicate the reference example over rows to form a [num_examples, num_input_features]
        # array, but it's done more efficiently. This forms a [num_examples, num_input_features]
        # sparse matrix, which we then sum over axis 1.
        dot_products = X.multiply(self.reference_example).sum(axis=1)
        # dot_products is now a [num_examples, 1] dense matrix. We'll turn it into a
        # 1-D array with shape [num_examples], to be consistent with our desired predictions.
        dot_products = np.asarray(dot_products).flatten()
        # If positive, return the same label; otherwise return the opposite label.
        same_label_mask = dot_products >= 0
        opposite_label_mask = ~same_label_mask
        y_hat = np.empty([num_examples], dtype=np.int)
        y_hat[same_label_mask] = self.reference_label
        y_hat[opposite_label_mask] = self.opposite_label
        return y_hat


class Majority(Model):

    def __init__(self):
        super().__init__()
        # TODO: Initializations etc. go here.
        self.highest_occurance = None

    def fit(self, X, y):
        self.num_input_features = X.shape[1]

        # Designate the first training example as the 'reference' example
        # It's shape is [1, num_features]
        zeros = 0
        ones = 0
        for label in y:
            if label == 1:
                ones += 1
            else:
                zeros += 1
        if(ones > zeros):
            self.highest_occurance = 1
        elif(ones < zeros):
            self.highest_occurance = 0
        else:
            self.highest_occurance = np.random.random_integers(0,2)
    def predict(self, X):
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')
        num_examples, num_input_features = X.shape
        y_hat = np.full([num_examples], self.highest_occurance)
        return y_hat

class Perceptron(Model):

    def __init__(self, a, b):
        super().__init__()
        self.w = None 
        self.rate = a
        self.iterations = b

    def fit(self, X, y):
        self.num_input_features = X.shape[1]
        self.w = np.zeros(X.shape[1])
        y_hat = np.zeros(y.shape)
        self.calculate(X,y, y_hat)

    def calculate(self, X, y, y_hat):
        for i in range(0, self.iterations):
            counter = 0
            for examples in X:
                num_examples, num_input_features = X.shape
                if num_input_features < self.num_input_features:
                    X = X.copy()
                    X._shape = (num_examples, self.num_input_features)

                if num_input_features > self.num_input_features:
                    X = X[:, :self.num_input_features]
                dot_products = examples.multiply(self.w).sum(axis=1)
                compare = 1
                if y[counter] == 0:
                    compare = -1
                if compare != np.sign(dot_products):
                    self.w = self.w  + (examples.multiply(self.rate*compare))
                counter = counter + 1
                

    def predict(self, X):
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')
        
        y_hat = np.zeros(X.shape[0])
        counter = 0
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)

        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        
        for examples in X:
            dot_products = examples.multiply(self.w).sum(axis=1)
            sign = np.sign(dot_products)
            if sign < 0 or dot_products == 0:
                sign = 0
            y_hat[counter] = sign
            counter = counter + 1

        return y_hat
