require "./k_nearest_neighbour_classifier.rb"
require "./iris_instance.rb"

DEFAULT_TRAINING_SET_FILE = "iris-training.txt".freeze
DEFAULT_TEST_SET_FILE     = "iris-test.txt".freeze
DEFAULT_K_VALUE           = 1
DEFAULT_NUM_CLUSTER       = 3

# set parameters
training_set_file = ARGV[0].nil? ? DEFAULT_TRAINING_SET_FILE : ARGV[0]
test_set_file     = ARGV[1].nil? ? DEFAULT_TEST_SET_FILE : ARGV[1]
k_value           = ARGV[2].nil? ? DEFAULT_K_VALUE : ARGV[2].to_i
num_cluster       = ARGV[3].nil? ? DEFAULT_NUM_CLUSTER : ARGV[3].to_i

# read in the training file
begin
  training_set = File.readlines(training_set_file).reject { |line| line.strip.empty? }.map do |line|
    values = line.split
    IrisInstance.new(values[0].to_f, values[1].to_f, values[2].to_f, values[3].to_f, values[4])
  end
rescue StandardError => e
  abort("Error occurred when reading training data. Exception message: #{e.message}")
end

# read in the test file
begin
  test_set = File.readlines(test_set_file).reject { |line| line.strip.empty? }.map do |line|
    values = line.split
    IrisInstance.new(values[0].to_f, values[1].to_f, values[2].to_f, values[3].to_f, values[4])
  end
rescue StandardError => e
  abort("Error occurred when reading test data. Exception message: #{e.message}")
end

# prompt the user, ask if k-means clustering needs to be done
puts "Do you want to do k-means clustering? (Y/N)"

user_input = STDIN.gets.strip.upcase
while user_input != "Y" && user_input != "N"
  puts "Please only type \"Y/N\":"
  user_input = STDIN.gets.strip.upcase
end

do_cluster = { Y: true, N: false }.fetch(user_input.to_sym)

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
