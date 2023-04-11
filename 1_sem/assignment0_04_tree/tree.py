import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    prob = np.sum(y, axis=0)/len(y)
    entropy = -np.sum(prob*np.log(prob+EPS))
    return entropy
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """
    prob = np.sum(y, axis=0)/len(y)
    gini = 1. - np.sum(prob*prob)
    return gini
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    return np.mean(np.absolute(y - np.median(y)))

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """
    return np.mean(np.absolute(y - np.median(y)))


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.predict = None 
        self.left_child = None
        self.right_child = None
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split 
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with
        threshold : float
            Threshold value to perform split
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset
        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """
        X_left = list()
        X_right = list()
        y_left = list()
        y_right = list()
        for i in range(X_subset.shape[0]):
                if (X_subset[i, feature_index] < threshold):
                    X_left.append(X_subset[i])
                    y_left.append(y_subset[i])
                elif (X_subset[i, feature_index] >= threshold):
                    X_right.append(X_subset[i])
                    y_right.append(y_subset[i])
        X_left = np.array(X_left)
        X_right = np.array(X_right)
        y_left = np.array(y_left)
        y_right = np.array(y_right)
        
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with
        threshold : float
            Threshold value to perform split
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset
        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold
        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        y_left = list()
        y_right = list()
        for i in range(X_subset.shape[0]):
                if (X_subset[i, feature_index] < threshold):
                    y_left.append(y_subset[i])
                elif (X_subset[i, feature_index] >= threshold):
                    y_right.append(y_subset[i])
        y_left = np.array(y_left)
        y_right = np.array(y_right)

        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset
        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with
        threshold : float
            Threshold value to perform split
        """
        best_index = 0
        best_threshold = 0
        best_error = np.inf

        if X_subset.shape[0] == 0: 
            return 0, 0
        
        for feature_index in range(X_subset.shape[1]):
            for threshold in np.unique(X_subset[:, feature_index]): 
                y_left, y_right = self.make_split_only_y(feature_index, threshold, X_subset, y_subset)

                if (y_left.shape[0] < self.min_samples_split or y_right.shape[0] < self.min_samples_split):
                    continue

                error_left = self.criterion(y_left)
                error_right = self.criterion(y_right)
                error = (error_left * y_left.shape[0] + error_right * y_right.shape[0]) / y_subset.shape[0]
                if (best_error > error):
                    best_index = feature_index
                    best_threshold = threshold
                    best_error = error
            
        return best_index, best_threshold


    
    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset
        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """
        feature_index, threshold = self.choose_best_split(X_subset, y_subset)
        new_node = Node(feature_index, threshold)
        if (X_subset.shape[0] == 1 or self.depth == self.max_depth):
            self.depth -= 1
            new_node.predict = y_subset
            return new_node
        else:
            (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)
            self.depth += 1
            new_node.left_child = self.make_tree(X_left, y_left)
            self.depth += 1
            new_node.right_child = self.make_tree(X_right, y_right)
            self.depth -= 1
            return new_node

        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on
        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)

    
    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for
        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """

        y_predicted = []
        self.criterion, self.classification = self.all_criterions[self.criterion_name]

        for i in range(X.shape[0]):
            root = self.root
            while root.left_child :
                if (X[i][root.feature_index] < root.value):
                    root = root.left_child
                else:
                    root = root.right_child
            
            if (self.classification):
                y_predicted.append(np.argmax(np.sum(root.predict, axis=0)))
            else:
                y_predicted.append(np.mean(root.predict))
        
        return y_predicted



    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for
        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        y_predicted_proba = []
        for i in range(X.shape[0]):
            root = self.root
            while root.left_child: 
                if (X[i][root.feature_index] < root.value):
                    root  = root.left_child
                else:
                    root = root.right_child
            if (len(root.predict) == 0):
                y_predicted_proba.append(list(np.squeeze(np.zeros([1, self.n_classes]))))
            else:
                y_predicted_proba.append(list(np.sum(root.predict, axis=0)/len(root.predict)))
        
        return np.array(y_predicted_proba)
