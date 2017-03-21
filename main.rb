require "./k_nearest_neighbour_classifier.rb"
require "./iris_instance.rb"

DEFAULT_TRAINING_SET_FILE = "iris-training.txt"
DEFAULT_TEST_SET_FILE     = "iris-test.txt"
DEFAULT_K_VALUE           = 1

training_set_file = ARGV[0].nil? ? DEFAULT_TRAINING_SET_FILE : ARGV[0]
test_set_file = ARGV[1].nil? ? DEFAULT_TEST_SET_FILE : ARGV[1]
k_value = ARGV[2].nil? ? DEFAULT_K_VALUE : ARGV[2].to_i

classifier = KNearestNeighbourClassifier.new(training_set_file, test_set_file, k_value)

classifier.read_data
classifier.estimate_ranges
classifier.classify_all

classifier.show_result
