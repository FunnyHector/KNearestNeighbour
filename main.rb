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

def get_KNN_or_clustering_from_user
  puts "Do you want to do K-Nearest Neighbours or K-Means clustering? (choose 1/2)"
  puts "  1. K-Nearest Neighbours"
  puts "  2. K-Means clustering"

  user_input = STDIN.gets.strip.to_i
  while user_input != 1 && user_input != 2
    puts "Please only type \"1/2\":"
    user_input = STDIN.gets.strip.to_i
  end

  user_input
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

# =========== Here we go ===================

# set parameters
training_set_file = ARGV[0].nil? ? DEFAULT_TRAINING_SET_FILE : ARGV[0]
test_set_file     = ARGV[1].nil? ? DEFAULT_TEST_SET_FILE : ARGV[1]

# read in the training file & test file
training_set = read_file(training_set_file)
test_set = read_file(test_set_file)

# ask the user which one to do, K-Nearest Neighbours or K-Means clustering
choice = get_KNN_or_clustering_from_user

# if do KNN
if choice == 1
  # ask the user for k-value
  k_value = get_k_value_from_user

  # show the parameters used
  puts "Running K-Nearest Neighbours with:"
  puts "  K = #{k_value}"

  # initialise the classifier and run magic!
  classifier = KNearestNeighbourClassifier.new(training_set, test_set, k_value, nil, false)
  classifier.estimate_ranges
  classifier.classify_test_set

  # print the result, and write the result into output.txt
  result = classifier.result
  File.write("./sample_output_KNN.txt", result)
  puts result
  puts "\n=======================================================\n"
  puts "\"sample_output_KNN.txt\" is generated."

  # if do clustering
elsif choice == 2
  # ask the user the number of clusters
  num_cluster = get_num_of_cluster_from_user

  # show the parameters used
  puts "Running K-Means Clustering with:"
  puts "  Num of clusters: #{num_cluster}"

  # initialise the classifier and run magic!
  classifier = KNearestNeighbourClassifier.new(training_set, test_set, nil, num_cluster, true)
  classifier.estimate_ranges
  classifier.cluster_training_set

  # print the result, and write the result into output.txt
  result = classifier.result
  File.write("./sample_output_clustering.txt", result)
  puts result
  puts "\n=======================================================\n"
  puts "\"sample_output_clustering.txt\" is generated."
else
  abort("Unknown choice: \"#{choice}\". Abort.")
end
