require "./k_nearest_neighbour_classifier.rb"
require "./iris_instance.rb"

# define constants
DEFAULT_TRAINING_SET_FILE = "iris-training.txt".freeze
DEFAULT_TEST_SET_FILE     = "iris-test.txt".freeze
DEFAULT_K_VALUE           = 1
DEFAULT_NUM_CLUSTER       = 3

# define helper methods
def read_file(file)
  File.readlines(file).reject { |line| line.strip.empty? }.map do |line|
    values = line.split
    IrisInstance.new(values[0].to_f, values[1].to_f, values[2].to_f, values[3].to_f, values[4])
  end
rescue StandardError => e
  abort("Error occurred when reading \"#{file}\". Exception message: #{e.message}")
end

def get_clustering_or_not_from_user
  puts "Do you want to do k-means clustering? (Y/N)"

  user_input = STDIN.gets.strip.upcase
  while user_input != "Y" && user_input != "N"
    puts "Please only type \"Y/N\":"
    user_input = STDIN.gets.strip.upcase
  end

  user_input
end

def get_k_value_from_user
  puts "Please enter a K-value (1 - 10):"

  user_input = STDIN.gets.strip.to_i
  while user_input < 1 || user_input > 10
    puts "Please enter an allowed K-value (1 - 10):"
    user_input = STDIN.gets.strip.to_i
  end

  user_input
end

def get_num_of_cluster_from_user
  puts "Please enter a value for number of clusters (greater than 0):"

  user_input = STDIN.gets.strip.to_i
  while user_input < 1
    puts "Please enter an allowed int (greater than 0):"
    user_input = STDIN.gets.strip.to_i
  end

  user_input
end

# set parameters
training_set_file = ARGV[0].nil? ? DEFAULT_TRAINING_SET_FILE : ARGV[0]
test_set_file     = ARGV[1].nil? ? DEFAULT_TEST_SET_FILE : ARGV[1]

# read in the training file & test file
training_set = read_file(training_set_file)
test_set = read_file(test_set_file)

# ask the user for k-value
k_value = get_k_value_from_user

# ask the user if k-means clustering needs to be done
user_input = get_clustering_or_not_from_user
do_cluster = { Y: true, N: false }.fetch(user_input.to_sym)
num_cluster = get_num_of_cluster_from_user if do_cluster

# show the parameters used
puts "Running K-Nearest Neighbours with:"
puts "  K = #{k_value}"
puts "  Num of clusters: #{num_cluster}" if do_cluster

# initialise the classifier and run magic!
classifier = KNearestNeighbourClassifier.new(training_set, test_set, k_value, num_cluster, do_cluster)

classifier.estimate_ranges
classifier.cluster_training_set if do_cluster
classifier.classify_test_set

# print the result, and write the result into output.txt
result = classifier.result
File.write("./output.txt", result)
puts result
puts "\n=======================================================\n"
puts "\"output.txt\" is generated."
