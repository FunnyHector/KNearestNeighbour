require "./k_nearest_neighbour_classifier.rb"
require "./iris_instance.rb"

DEFAULT_TRAINING_SET_FILE = "iris-training.txt"
DEFAULT_TEST_SET_FILE     = "iris-test.txt"
DEFAULT_K_VALUE           = 1
DEFAULT_NUM_CLUSTER       = 3

# set parameters
training_set_file = ARGV[0].nil? ? DEFAULT_TRAINING_SET_FILE : ARGV[0]
test_set_file     = ARGV[1].nil? ? DEFAULT_TEST_SET_FILE : ARGV[1]
k_value           = ARGV[2].nil? ? DEFAULT_K_VALUE : ARGV[2].to_i
num_cluster       = ARGV[3].nil? ? DEFAULT_NUM_CLUSTER : ARGV[3].to_i

# prompt the user, ask if k-means clustering needs to be done
puts "Do you want to do k-means clustering? (Y/N)"

user_input = STDIN.gets.strip.upcase
while user_input != "Y" && user_input != "N"
  puts "Please only type \"Y/N\":"
  user_input = STDIN.gets.strip.upcase
end

@do_cluster = { Y: true, N: false }.fetch(user_input.to_sym)

# show the parameters used
puts "Running K-Nearest Neighbours with:"
puts "  K = #{k_value}"
puts "  Num of clusters: #{num_cluster}" if @do_cluster

classifier = KNearestNeighbourClassifier.new(training_set_file, test_set_file, k_value, num_cluster, @do_cluster)

classifier.read_data
classifier.estimate_ranges
classifier.cluster_training_set if @do_cluster
classifier.classify_test_set

classifier.show_result
